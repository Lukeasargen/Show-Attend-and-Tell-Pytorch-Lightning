from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import corpus_gleu
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import MultiStepLR, ReduceLROnPlateau, ExponentialLR, CosineAnnealingWarmRestarts, OneCycleLR
from torchvision import models
from torchvision.transforms import Normalize

from util import LabelSmoothing


class FlattenShuffle(nn.Module):
    """ Flatten and Shuffle the encoder output.
        Change the shape from (B, C, H, W) to (B, H*W, C).
    """
    def __init__(self):
        super(FlattenShuffle, self).__init__()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b, c, h*w)  # Flatten the feature maps to 1 dimension
        # 0=batch, 1=channels, 2=locations
        return x.permute(0, 2, 1)


def get_encoder(args):
    """ Return a torchvision model with an output volume for SAT. """
    # args.encoder_arch is a string
    if callable(models.__dict__[args.encoder_arch]):
        m = models.__dict__[args.encoder_arch](pretrained=args.pretrained)

        # Only freeze the weights if the model is pretrained
        if args.pretrained:
            for param in m.parameters():
                param.requires_grad = False

        # Get model features before pooling and linear layers
        if "resnet" in args.encoder_arch or "resnext" in args.encoder_arch:
            layers = list(m.children())[:-2]  # Remove pooling and fc
        elif "shufflenet" in args.encoder_arch:
            layers = list(m.children())[:-1]  # Remove fc
        elif "squeezenet" in args.encoder_arch:
            layers = list(m.children())[:-1]  # Remove classifer
        elif "densenet" in args.encoder_arch:
            layers = list(m.children())[:-1]  # Remove classifer
        elif "mobilenet_v2" in args.encoder_arch:
            layers = list(m.children())[:-1]  # Remove classifer
        elif "mobilenet_v3" in args.encoder_arch:
            layers = list(m.children())[:-2]  # Remove pooling and classifer
        elif "mnasnet" in args.encoder_arch:
            layers = list(m.children())[:-1]  # Remove pooling and classifer
        else:
            raise ValueError("Encoder not supported : {}".format(args.encoder_arch))

        # Pass a fake iamge through the model to see what the annotation dimensions are
        fake_img = torch.zeros(1, 3, args.input_size, args.input_size)
        yhat = nn.Sequential(*layers)(fake_img)
        _, final_dim, final_size, _ = yhat.shape

        if args.encoder_dim is not None and args.encoder_dim!=final_dim:
            # This conv forces the number of features to be encoder_dim
            # Does not match the pretrained encoders in paper since this always requires training
            layers.append(nn.Conv2d(final_dim, args.encoder_dim, kernel_size=1, stride=1, bias=True))
        else:
            # Store the encoder_dim back in the hparams
            args.encoder_dim = final_dim
        
        # TODO : pool->conv or conv->pool? pool will change the features locally before the conv op
        # For now conv->pool, bc conv will take the actual features as input rather than pooled features
        if args.encoder_size is not None and args.encoder_size!=final_size:
            if args.encoder_size < final_size:
                # Going to a smaller map uses average pooling
                layers.append(nn.AdaptiveAvgPool2d((args.encoder_size, args.encoder_size)))
            elif args.encoder_size > final_size:
                # Average pooling uses nearest neighbor for upsampling, but sometimes it interpolates
                # To avoid not knowning what upsampling happens, force a behavior with nn.Upsample
                # Setting align_corners=False is the default, but it is set explicitly to suppress warnings
                layers.append(nn.Upsample((args.encoder_size, args.encoder_size), mode="bilinear", align_corners=False))
        else:
            # Store the encoder_size back in the hparams
            args.encoder_size = final_size

        # Layer that flattens the feature maps from 2d to 1d and then
        # reshapes so that there are L feature locations along dim=1, and
        # dim=2 is the representation with size D
        layers.append(FlattenShuffle())
        
        # First layer of the model, input is [0,1] normalized images
        norm = Normalize(args.mean, args.std, inplace=True)

        # Sequential model returns shape (batch, encoder_size**2, encoder_dim)
        return nn.Sequential(norm, *layers)
    raise ValueError("Unknown model arg: {}".format(args.encoder_arch))


class InitLSTM(nn.Module):
    """ Sec 3.1.2 "predicted by an average of the annotation vectors feed through two separate MLPs" """
    def __init__(self, encoder_dim, decoder_dim, decoder_layers, dropout, bias=True):
        super(InitLSTM, self).__init__()
        self.decoder_dim = decoder_dim
        self.decoder_layers = decoder_layers
        self.init_h = nn.Linear(encoder_dim, decoder_dim*decoder_layers, bias=bias)
        self.init_c = nn.Linear(encoder_dim, decoder_dim*decoder_layers, bias=bias)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, annotations):
        # Mean over the locations (dim=1) to get a vector of length encoder_dim
        mean = self.dropout(annotations.mean(1))
        # shape = (decoder_layers, batch, decoder_dim)
        init_h = self.init_h(mean).reshape(self.decoder_layers, mean.shape[0], self.decoder_dim)
        init_c = self.init_c(mean).reshape(self.decoder_layers, mean.shape[0], self.decoder_dim)
        return init_h, init_c


class SoftAttention(nn.Module):
    """ SAT citation points here -> Bahdanau et al. (2014). Sec 3.1 Equations 5, 6
        Formula is clear here: https://paperswithcode.com/method/additive-attention
    """
    def __init__(self, attention_dim, encoder_dim, hidden_dim):
        super(SoftAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim, bias=False)
        self.decoder_att = nn.Linear(hidden_dim, attention_dim, bias=False)
        self.f_att = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, annotations, decoder_hidden):
        """ Returns the context and the alpha (for doubly sotchastic loss and visualizations) """
        # Convert encoder_dim to attention_dim features
        att_enc = self.encoder_att(annotations)  # att_enc.shape = (batch, locations, attention_dim)
        # Convert hidden_dim to attention_dim features, expaned dim=1 for braodcasting to all locations(dim=1)
        att_dec = self.decoder_att(decoder_hidden).unsqueeze(1)   # att_dec.shape = (batch, 1, attention_dim)
        # Take all the features and predict and attention value for each location(dim=1)
        att = self.f_att(torch.tanh(att_enc+att_dec))  # att.shape = (batch, attention_dim, 1)
        # Softmax over the all locations(dim=1)
        alpha = F.softmax(att, dim=1)  # alpha.shape = (batch, attention_dim, 1)
        # Apply the alphas to the annotations, then sum each feature map over all locations(dim=1)
        zt = (annotations*alpha).sum(dim=1)  # zt.shape = (batch, encoder_dim)
        return zt, alpha.reshape(alpha.shape[0], -1) # Remove last dim of alpha before return


class DeepOutput(nn.Module):
    """ Sec 3.1.2 Equation 7 - Deep Output Layer """
    def __init__(self, args):
        super(DeepOutput, self).__init__()
        self.deep = args.deep_output
        self.dropout = nn.Dropout(p=args.dropout)
        if self.deep:
            self.hidden = nn.Linear(args.decoder_dim, args.embed_dim, bias=False)
            self.context = nn.Linear(args.encoder_dim, args.embed_dim, bias=False)
            self.output = nn.Linear(args.embed_dim, args.vocab_size)
        else:
            self.output = nn.Linear(args.decoder_dim, args.vocab_size)

    def forward(self, prev_embed, hidden, context):
        if self.deep:
            x = prev_embed + self.hidden(hidden) + self.context(context)
            logit = self.output(self.dropout(torch.tanh(x)))
        else:
            logit = self.output(self.dropout(hidden))
        return logit  # logit.shape = (batch, vocab_size)


class SAT(pl.LightningModule):
    """ Show, Attend, and Tell (SAT)
    Note:
    -all image encoding is done in self.encoder using convolutional networks
    -all text decoding is done in this LightningModule (ie, beam search)
    """
    def __init__(self, **kwargs):
        super(SAT, self).__init__()
        self.save_hyperparameters()
        self.scheduler = None  # Set in configure_optimizers()
        self.opt_init_lr = None  # Set in configure_optimizers()

        # Smoothing of 0 is just regular cross entropy
        assert 0 <= self.hparams.label_smoothing < (self.hparams.vocab_size-1)/self.hparams.vocab_size
        self.criterion = LabelSmoothing(self.hparams.label_smoothing)

        # Keep these, a nice list to have around
        self.special_idxs = [self.stoi("<PAD>"), self.stoi("<START>"), self.stoi("<END>")]

        # Call the function to get the encoder architecture.
        self.encoder = get_encoder(self.hparams)

        # This is the size of the flattened encoding annotations.
        # attention_dim is the same as L in the paper.
        self.hparams.attention_dim = self.hparams.encoder_size**2

        # Sec 3.1.2 - Matrix E is the embedding matrix of shape (vocab_size, embed_dim).
        # This is initialized randomly and is trainable by default
        self.embedding = nn.Embedding(
            num_embeddings=self.hparams.vocab_size,
            embedding_dim=self.hparams.embed_dim,
            max_norm=self.hparams.embed_norm,
            padding_idx=self.stoi("<PAD>")
        )
        if self.hparams.pretrained_embedding is not None:
            # Load the pretrained embedding matrix into the embedding layer
            embedding_matrix = np.load(self.hparams.pretrained_embedding)
            self.embedding.weight = nn.Parameter(torch.tensor(embedding_matrix, dtype=torch.float32))

        # Sec 3.1.2 - Intialize the hidden states based on the mean annotations and two separate MLPs.
        self.init_lstm = InitLSTM(self.hparams.encoder_dim, self.hparams.decoder_dim,
                                self.hparams.decoder_layers, self.hparams.dropout, bias=True)

        # Sec 3.1.2 Equations 1, 2, 3 - Apply the LSTM update rules.
        # Per the paper, input_size is encoder_dim+embed_dim (D+m) and the hidden_size is decoder_dim (n).
        self.lstm = nn.LSTM(
            input_size=self.hparams.embed_dim+self.hparams.encoder_dim,
            hidden_size=self.hparams.decoder_dim,
            num_layers=self.hparams.decoder_layers,
            bias=True
        )

        # Sec 3.1.2 Equations 4, 5, 6 - Soft Attention Module
        self.attention = SoftAttention(
            attention_dim=self.hparams.attention_dim,
            encoder_dim=self.hparams.encoder_dim,
            hidden_dim=self.hparams.decoder_dim
        )

        # Sec 4.2.1 - predict gating scalar \beta from previous hidden state
        # This is outside of the attention module bc the context vector is needed for deep output
        self.beta = nn.Sequential(
            nn.Linear(self.hparams.decoder_dim, self.hparams.encoder_dim, bias=True),
            nn.Sigmoid()
        )
        fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.beta[0].weight)
        # self.beta[0].bias.data.fill_(1/fan_in)

        # Sec 3.1.2 - Deep Output for word prediction
        self.deep_output = DeepOutput(self.hparams)

    def stoi(self, s):
        return int(self.hparams.vocab_stoi.get(s, self.hparams.vocab_stoi['<UNK>']))

    def itos(self, i):
        return str(self.hparams.vocab_itos.get(int(i), "<UNK>"))

    def decode_seq(self, seq, remove_special=False):
        """ Convert a list of int into a list of str. """
        keep_token = lambda x: not (remove_special and x in self.special_idxs)
        # Using str() just in case
        return [str(self.itos(t)) for t in seq if keep_token(t)]

    @torch.no_grad()
    def caption(self, img_tensor, beamk=3, max_gen_length=32, temperature=1.0, 
                sample_method="beam", decoder_noise=None,
                rescore_method=None, rescore_reward=0.5,
                return_all=False):
        """ Caption method for better code readability.
            Input : img tensor with shape [B, C, H, W].
            Output : lists. captions, scores, alphas, and perplexity
            beamk : beam width
            max_gen_length :  maximum length
            temperature : scaling factor before softmax
            sample_method : beam=Beam Search, m=Multinomial Sampling,
            rescore_method : None, LN=Length Normalization, WR=Word Reward, BAR=Bounded Adaptive-Reward
            rescore_reward : should be tuned on a dev set
            decoder_noise : base variance for noise injected to hidden state during decoding
            return_all : True returns a list of batch size filled with lists of beamk size
        """
        self.eval()  # Freeze all parameters and turn off dropout
        return self.forward(img_tensor, beamk, max_gen_length, temperature,
                            sample_method, decoder_noise,
                            rescore_method, rescore_reward,
                            return_all)

    def forward(self, img, beamk=3, max_gen_length=32, temperature=1.0, 
                sample_method="beam", decoder_noise=None,
                rescore_method=None, rescore_reward=0.5,
                return_all=False):
        """ Inference Method Only. Uses beam search to create a caption. """
        assert sample_method in ["beam", "multinomial", "topk"]

        # Keep this arg since beamk get's overwritten as sequences are completed
        beamk_arg = beamk

        # Makes single values a list so it can be indexed at each step
        if not isinstance(temperature, list):
            temperature = [temperature]

        # Create empty return lists, will become the length of the batch
        captions, cap_scores, cap_alphas, cap_perplexity = [], [], [], []

        # Encode the batch
        annotations = self.encoder(img)

        # TODO : batch decoding with beam search?
        # Loop over the batch on image at a time
        for idx in range(img.shape[0]):

            beamk = beamk_arg  # Reset to beamk args

            # Treat each image as a batch of beamk size
            annots = annotations[idx, :]  # (attention_dim, encoder_dim))
            annots = annots.expand(beamk, *annots.shape)  # (beamk, attention_dim, encoder_dim)

            # Init the beam batch
            h, c = self.init_lstm(annots)  # (beamk, decoder_dim)

            # Keep top econded sequences
            # top_preds.shape = (step, beamk)
            top_preds = torch.LongTensor([[self.stoi("<START>")]*beamk]).to(self.device)

            # Keep the -1/log sum
            # top_scores.shape = (beamk)
            top_scores = torch.zeros(beamk, dtype=torch.float).to(self.device)

            # Keep the alphas for visualization
            # alphas.shape = (step, beamk, attention_dim)
            alphas = torch.zeros(1, beamk, self.hparams.attention_dim).to(self.device)

            # Extend these lists once the <END> is predicted or max_gen_length is reached
            finished_captions = []
            finished_alphas = []
            finished_scores = []
            finished_perplexity  = []

            step = 0
            while True:
                # Get the current sampling temperature
                current_temperature = temperature[step % len(temperature)]

                # Get the last predictions
                k_prev_words = top_preds[step]

                # Forward pass through the model
                embed_prev_words = self.embedding(k_prev_words)
                zt, alpha = self.attention(annots, h[-1])
                beta = self.beta(h[-1])
                h_in = torch.cat([embed_prev_words, beta*zt], dim=1).unsqueeze(0)

                """ My own idea inspired by "Scheduled Sampling for Sequence Prediction with Recurrent Neural Network"
                In the paper, it is suggested that exploring the hidden space during
                training is beneficial. I think it will also be beneficial to explore
                more during inference decoding to increase generation diversity.
                To achieve this, I add noise to the hidden state or input.
                The noise decreases with generation length.
                The motivation is to start the generation in a different region in
                the hidden space and then finished the caption using the learned
                language model. By decreasing the noise, I hope the model will
                use the learned language model to finish the captions in an
                understandable way.
                I did a quick evalutaion to see where to add the noise.
                Bleu4 score. decoder_noise=0.1.
                No Noise : 25.77
                Noise on Hidden : 25.63 (decoder_noise=0.2 : 25.06)
                Noise on Input : 23.63
                Noise on Input and Hidden : 23.28
                It's apparent that noise on hidden is best to not degrade bleu4 score.
                """
                if decoder_noise is not None and decoder_noise!=0.0:
                    # h_in += torch.randn(h_in.size(), device=h_in.device) * decoder_noise/(step+1)
                    h += torch.randn(h.size(), device=h.device) * decoder_noise/(step+1)

                _, (h, c) = self.lstm(h_in, (h, c))
                logit = self.deep_output(embed_prev_words, h[-1], zt)

                # Use the log probability as the score
                scores = F.log_softmax(logit/current_temperature, dim=1)
                
                # TODO : mask the <START> and <PAD> token, right?
                scores[:, [self.stoi("<START>"), self.stoi("<PAD>")]] = float('-inf')

                # if no_unk:
                # scores[:, self.stoi("<UNK>")] -= 100 # float('-inf')

                if step==0:
                    # Additionally mask the <END> and <UNK> on the first step
                    scores[:, [self.stoi("<END>"), self.stoi("<UNK>")]] = float('-inf')
                    # Extract the top beamk words for the first image since
                    # the initial predicitons are all the same across the beam batch
                    top_scores, pred_idx = torch.topk(scores[0], beamk)
                    # Extend each top sequence with the top predictions
                    top_preds = torch.cat([top_preds, pred_idx.unsqueeze(0)], 0)
                    # Extend the alphas
                    alphas = torch.cat([alphas, alpha.unsqueeze(0)], 0)
                else:
                    # Compute the scores for all possible next words
                    # This adds the parent scores along the batch dimension
                    seq_scores = scores + top_scores.unsqueeze(1)

                    # The scores are all update. It's time to selected which seq to continue.
                    # Each method produces pred_idx, which is the hypotheses for this step
                    # NOTE : don't forget to FLATTEN the scores when selecting the index
                    if sample_method=="beam":
                        # Basic beam search considers all the scores without regard for parent node
                        # Flatten the scores so all scores are in dim=0
                        _, pred_idx = torch.topk(seq_scores.reshape(-1), beamk, dim=0)
                    elif sample_method=="multinomial":
                        # Take the softmax over each sample so the best word for each parent has a high probability
                        # Multiplying by 20 is like sharpening the probalities, increases the relative logits
                        score_probs = F.softmax(20*seq_scores/step, dim=1)
                        pred_idx = torch.multinomial(score_probs.reshape(-1), beamk)
                    elif sample_method=="topk":
                        # Choose topk from each beam and then randomly sample
                        topk = 5
                        # Take the topk samples from each beam
                        _, candidate_idxs = torch.topk(seq_scores, topk, dim=1)
                        # This adjustment corrects for the vocab_size of each successive sequence
                        # Add the adjustment and flatten the candidates
                        adj_idx = torch.tensor([i*self.hparams.vocab_size for i in range(beamk)]).unsqueeze(1).to(candidate_idxs)
                        candidate_idxs = (candidate_idxs+adj_idx).reshape(-1)
                        # Multinomial sampling
                        candidate_scores = seq_scores.reshape(-1)[candidate_idxs]
                        candidate_probs = F.softmax(candidate_scores/step, dim=0)
                        # Uniform sampling
                        # candidate_probs = torch.ones(candidate_idxs.numel())
                        choice_idx = torch.multinomial(candidate_probs, beamk)
                        pred_idx = candidate_idxs[choice_idx]

                    # Update the top_scores by taking selected scores on the flatten scores
                    top_scores = seq_scores.reshape(-1)[pred_idx]

                    # Which beam idx to keep, // floor divide
                    keep_seq_idxs = torch.div(pred_idx, self.hparams.vocab_size, rounding_mode='floor')

                    # Which vocab index to add to the seq, % remainder
                    # unsequeeze dim=0 so it can stack on top_preds
                    keep_vocab_idxs = torch.remainder(pred_idx, self.hparams.vocab_size).unsqueeze(0)

                    # Update the sequences by taking the top beamk scores and cat the word indexes
                    # top_preds[:,keep_seq_idxs] = take every timestep for the top beamk scores
                    top_preds = torch.cat([top_preds[:,keep_seq_idxs], keep_vocab_idxs], 0)
                    alphas = torch.cat([alphas[:,keep_seq_idxs], alpha.unsqueeze(0)[:,keep_seq_idxs]], 0)

                    # Update other variables for the selected top beamk scores
                    h, c = h[:, keep_seq_idxs], c[:, keep_seq_idxs]
                    annots = annots[keep_seq_idxs]
                
                # Now the top_preds,top_scores,alphas,h,c,annots are updated with the latest sequences
                # Check if any of the sequences predicted the <END> token and remove them
                complete = top_preds[step+1]==self.stoi("<END>")

                # Define this function here so it has access to the step and top_scores variables
                def rescore(s, method, reward):
                    if method=="LN":
                        # Length Normalization
                        return s/step
                    if method=="WR":                        
                        # Word Reward
                        return s + reward*step
                    if method=="BAR":
                        # Bounded Adaptive-Reward - https://www.aclweb.org/anthology/D18-1342.pdf Sec 4.2.2
                        average_beam_prob = -torch.mean(top_scores)
                        return s + reward*average_beam_prob
                    # No rescoring
                    return s

                if any(complete):
                    # Slice the sequence by complete and take everything except the <START> and <END> token 
                    finished_captions.extend([top_preds[:,complete][:,i][1:-1].tolist() for i in range(top_preds[:,complete].shape[1])])
                    finished_alphas.extend([alphas[:,complete][:,i][1:-1].cpu() for i in range(alphas[:,complete].shape[1])])
                    # Get the scores divide by the length
                    finished_scores.extend([rescore(top_scores[complete][i], rescore_method, rescore_reward).tolist() for i in range(top_scores[complete].shape[0])])
                    finished_perplexity.extend([torch.exp(-top_scores[complete][i]/step).tolist() for i in range(top_scores[complete].shape[0])])

                    # Reduce the beam batch to only the incomplete sequences
                    incomplete = torch.logical_not(complete)
                    top_preds = top_preds[:,incomplete]
                    alphas = alphas[:,incomplete]
                    top_scores = top_scores[incomplete]
                    h, c = h[:, incomplete], c[:, incomplete]
                    annots = annots[incomplete]

                    # Update the beamk value
                    beamk = sum(incomplete)

                    if beamk==0: break  # Leave once all sequence end

                # Add the incomplete sequences to the output
                if step >= max_gen_length:
                    finished_captions.extend([top_preds[:,i][1:-1].tolist() for i in range(top_preds.shape[1])])
                    finished_alphas.extend([alphas[:,i][1:-1].cpu() for i in range(alphas.shape[1])])
                    finished_scores.extend([rescore(top_scores[i], rescore_method, rescore_reward).tolist() for i in range(top_scores.shape[0])])
                    finished_perplexity.extend([torch.exp(-top_scores[i]/step).tolist() for i in range(top_scores.shape[0])])                        
                    break

                step += 1  # All updates are complete, iterate the step count

            # End of beam search

            # Append to the output lists
            if return_all:
                # Sort by scores before returning
                score_index = [[finished_scores[i], i] for i in range(len(finished_scores))]
                score_index.sort(reverse=True)
                score_index = [x[1] for x in score_index]
                captions.append([finished_captions[i] for i in score_index])
                cap_alphas.append([finished_alphas[i] for i in score_index])
                cap_scores.append([finished_scores[i] for i in score_index])
                cap_perplexity.append([finished_perplexity[i] for i in score_index])         
            else:
                best_idx = finished_scores.index(max(finished_scores))
                captions.append(finished_captions[best_idx])
                cap_alphas.append(finished_alphas[best_idx])
                cap_scores.append(finished_scores[best_idx])
                cap_perplexity.append(finished_perplexity[best_idx])

        # End of the img batch loop
        
        # Return these lists, which are the length of the batch
        return captions, cap_scores, cap_alphas, cap_perplexity

    def train_batch(self, batch, epsilon=0):
        """ Forward passes a batch to output logits.
            Set epsilon=1 at validation to tune temperature scaling.
        """
        # Unpack, pl puts the data on the correct device already
        img, encoded_captions, lengths = batch

        # Encode the images into annotation vectors
        # annotations.shape = (batch, attention_dim, encoder_dim)
        annotations = self.encoder(img)

        # Repeat the annotations to match the number of target captions
        annotations = annotations.repeat_interleave(lengths.size(1), dim=0)
        # The design of my dataset ouputs dim=1 as the number of captions
        # We must flatten the num_captions dimension to match the annotations
        encoded_captions = encoded_captions.reshape(-1, encoded_captions.size(2))
        # Same with lengths
        lengths = lengths.reshape(-1)
        # Shift the caption to the left to get the targets. The target is the next word
        targets = encoded_captions[:, 1:]

        # Initialize the lstm states with the annotations as input
        # states.shape = (batch, decoder_dim)
        h, c = self.init_lstm(annotations)

        # Allocate empty tensors for predictions and attention alphas
        bs, caplen = encoded_captions.shape  # Get the shape of the captions
        # Use caplen-1 bc the targets are shifted left 1
        # This is used to calculate the softmax of the token prediction
        logits = torch.zeros(bs, caplen-1, self.hparams.vocab_size).to(self.device)
        # This stores the attention alphas to compute the doubly stochastic loss
        alphas = torch.zeros(bs, caplen-1, self.hparams.attention_dim).to(self.device)

        # Now we can step through each time step
        # NOTE : the batch dimension will reduce as the captions are completed
        for step in range(caplen-1):
            # Boolean vector indicating which captions have not ended at this step 
            incomplete_idxs = lengths>step
            if not any(incomplete_idxs): break  # All the captions are done

            # On the first few steps to start off with a reasonable sequence
            # then uniformly sample between [0,1] and compare to epsilon
            # NOTE : hard coded value
            if step<=2 or torch.rand(1) <= epsilon:
                # Get the actual next word embedding
                embed_prev_words = self.embedding(encoded_captions[incomplete_idxs, step])
            else:
                # Get the argmax of the previous logits
                idxs = torch.argmax(logits[incomplete_idxs, step-1, :], dim=1)
                # Get the predicted next word embedding
                embed_prev_words = self.embedding(idxs)

            # Compute the attention context vector from the annotations and the previous hidden state
            # zt.shape, alpha.shape = (batch, encoder_dim)
            zt, alpha = self.attention(annotations[incomplete_idxs], h[-1, incomplete_idxs])            
            # Save the alpha values
            alphas[incomplete_idxs, step, :] = alpha

            # Compute the gating scalar beta from the previous hidden state
            # beta.shape = (batch, encoder_dim). Notice this matches zt so we can do the element wise product
            beta = self.beta(h[-1, incomplete_idxs].squeeze(1))  # squeeze to remove the length dim

            # Prepare to enter the lstm with shape (1, batch, embed_dim+enocder_dim)
            h_in = torch.cat([embed_prev_words, beta*zt], dim=1).unsqueeze(0)

            # Compute the new hidden states
            _, (h[:, incomplete_idxs], c[:, incomplete_idxs]) = self.lstm(h_in, (h[:, incomplete_idxs], c[:, incomplete_idxs]))

            # Compute the word logits
            logit = self.deep_output(embed_prev_words, h[-1, incomplete_idxs], zt)
            logits[incomplete_idxs, step, :] = logit.clone().float()  # clone because float() is an inplace operation

        # End of the loop

        # Pack the logits and the targets
        logits_packed = pack_padded_sequence(logits, lengths.tolist(), batch_first=True, enforce_sorted=False)
        targets_packed = pack_padded_sequence(targets, lengths.tolist(), batch_first=True, enforce_sorted=False)

        # Return all these tensors
        return logits_packed, targets_packed, alphas

    def training_step(self, batch, batch_idx):
        """ Method used only during training. """
        # NOTE : there are hard coded values here
        # For teacher force, epsilon=1 is forced, epsilon=0 uses predictions
        # As epsilon goes down, there is less chance of teacher forcing
        # Epsilon is a tensor bc then I don't have to change tensorboard metric logging code
        if self.hparams.decoder_tf is not None:
            if self.hparams.decoder_tf=="always":
                epsilon = torch.tensor(1)
            elif self.hparams.decoder_tf=="linear":
                # Linear decays down to decoder_tf_min teacher forcing by the final epoch
                epsilon = torch.tensor(1 - (1-self.hparams.decoder_tf_min)*self.current_epoch/self.hparams.epochs)
            elif self.hparams.decoder_tf=="inv_sigmoid":
                # Shift the 50% point to b, change the slope with g
                # l is from the final epsilon, used to set b
                l = -torch.log(torch.tensor(self.hparams.decoder_tf_min/(1-self.hparams.decoder_tf_min)))
                g = 5.0
                b = (1/((l/g)+1))*self.hparams.epochs
                epsilon = 1/(1+torch.exp(torch.tensor((g/b)*(self.current_epoch-b))))
            elif self.hparams.decoder_tf=="exp":
                # Exponential decays down to decoder_tf_min teacher forcing by the final epoch
                epsilon = torch.exp(torch.log(torch.tensor(self.hparams.decoder_tf_min))/self.hparams.epochs)**self.current_epoch
        else:
            epsilon = torch.tensor(0)  # No teacher forcing

        if self.global_step==self.hparams.encoder_finetune_after and self.hparams.encoder_finetune_after>=0:
            for param in self.encoder.parameters():
                param.requires_grad = True

        # Forward pass
        logits_packed, targets_packed, alphas = self.train_batch(batch, epsilon)

        # Loss calculation
        loss = self.criterion(logits_packed.data, targets_packed.data)
        # Sec 4.2.1 Equation 14 - doubly stochastic loss
        loss += self.hparams.att_gamma*((1-alphas.sum(dim=1))**2).mean()

        pred = torch.argmax(logits_packed.data, dim=1)
        acc = torch.sum(pred==targets_packed.data)/pred.shape[0]

        # Create metrics dict
        metrics = {
            "loss": loss,
            "accuracy": float(acc),
            "epsilon_tf": float(epsilon),
        }

        # Update tensorboard for each train step
        for k, v in metrics.items():
            key = "{}/train".format(k)
            val = metrics[k]
            if torch.is_tensor(val):
                val = val.item()
            self.logger.experiment.add_scalar(key, val, global_step=self.global_step)

        # Update the lr during warmup
        """ Code from https://pytorch-lightning.readthedocs.io/en/latest/common/optimizers.html
            Except I didn't override optimizer_step() bc that would break gradient accumulation.
        """
        if self.trainer.global_step < self.hparams.lr_warmup_steps:
            opt = self.optimizers()
            lr_scale = min(1, float(self.trainer.global_step+1)/self.hparams.lr_warmup_steps)
            for pg, init_lr in zip(opt.param_groups, self.opt_init_lr):
                pg['lr'] = lr_scale*init_lr
        elif self.trainer.global_step > 0:
            # Step these schedulers every batch
            if type(self.scheduler) in [CosineAnnealingWarmRestarts, OneCycleLR]:
                self.scheduler.step()

        return metrics

    def training_epoch_end(self, outputs):
        # Calculating epoch metrics
        for k in outputs[0].keys():
            key = "{}/train_epoch".format(k)
            vals = [x[k] for x in outputs]
            val = sum(vals) / len(vals) if len(vals)!=0 else 0
            self.logger.experiment.add_scalar(key, val, global_step=self.current_epoch+1)

        # Log lr, group 0 is a decoder module
        lr = self.optimizers().param_groups[0]['lr']
        self.logger.experiment.add_scalar('Learning Rate', lr, global_step=self.current_epoch+1)

        # Step these schedulers every epoch
        if type(self.scheduler) in [MultiStepLR, ExponentialLR]:
            self.scheduler.step()

    def score_captions(self, captions, encoded_captions, lengths, perplexities=None):
        # Remove <PAD> <START> <END>
        references = [[c[1:l] for c,l in zip(refs, lengths[i])] for i,refs in enumerate(encoded_captions.tolist())]

        # Same input for all these metrics, nice
        bleu1 = corpus_bleu(references, captions, weights=(1, 0, 0, 0))
        bleu2 = corpus_bleu(references, captions, weights=(0.5, 0.5, 0, 0))
        bleu3 = corpus_bleu(references, captions, weights=(0.33, 0.33, 0.33, 0))
        bleu4 = corpus_bleu(references, captions, weights=(0.25, 0.25, 0.25, 0.25))
        gleu = corpus_gleu(references, captions)

        # TODO : does this idea work? Check correlation with other metrics in evaluation
        # Average the embedding vectors of the sequences and use
        # the maximum cosine similarity with the references
        cossims = torch.zeros(encoded_captions.shape[0], dtype=torch.float).to(self.device)
        for i in range(encoded_captions.shape[0]):
            # Caption mean embedding
            cv = self.embedding(torch.LongTensor(captions[i]).to(self.device)).mean(0).unsqueeze(0)
            rvs = torch.zeros(encoded_captions.shape[1], dtype=torch.float).to(self.device)
            for j, l in enumerate(lengths[i]):
                ec = encoded_captions[i][j][1:l]  # Slice without START and END
                # Reference mean embedding
                rv = self.embedding(ec).mean(0).unsqueeze(0)
                rvs[j] = F.cosine_similarity(rv, cv)
            # Take the maximum similarity
            cossims[i] = max(rvs)
        # Average of the best similarities
        cosine_similarity = cossims.mean()

        # Create metrics dict
        metrics = {
            "bleu1": bleu1, "bleu2": bleu2, "bleu3": bleu3, "bleu4": bleu4,
            "cosine_similarity": cosine_similarity.item(),
            "gleu": gleu,
        }
        if type(perplexities)==list: metrics.update({"perplexity": sum(perplexities)/len(perplexities)})
        return metrics

    def val_batch(self, batch, beamk=3, max_gen_length=32, temperature=0.5,
                sample_method="beam", decoder_noise=None,
                rescore_method=None, rescore_reward=0.5):
        img, encoded_captions, lengths = batch  # Unpack
        captions, scores, alphas, perplexities = self.caption(img, beamk, max_gen_length,
            temperature, sample_method, decoder_noise, rescore_method, rescore_reward, return_all=False)
        metrics = self.score_captions(captions, encoded_captions, lengths, perplexities)
        return metrics

    def validation_step(self, batch, batch_idx):
        """ Method used only for validation. """
        return self.val_batch(batch, beamk=self.hparams.val_beamk,
                            max_gen_length=self.hparams.val_max_len,
                            temperature=1.0, rescore_method="LN")

    def validation_epoch_end(self, outputs): 
        # Calculate epoch metrics 
        for k in outputs[0].keys():
            key = "{}/val_epoch".format(k)
            vals = [x[k] for x in outputs]
            try:  # precision and recall fail for short captions
                val = sum(vals) / len(vals)
            except:
                val = 0
            if self.current_epoch!=0:  # Don't log pl's "sanity check"
                self.logger.experiment.add_scalar(key, val, global_step=self.current_epoch+1)
            # Use self.log() for the checkpoint callbacks
            if k==self.hparams.save_monitor: self.log(k, val)
            if k==self.hparams.early_stop_monitor: self.log(k, val)
            if k==self.hparams.plateau_monitor: plateau_val = val  # For plateau scheduler

        # Step plateau scheduler
        if self.trainer.global_step >= self.hparams.lr_warmup_steps:
            if type(self.scheduler) in [ReduceLROnPlateau]:
                self.scheduler.step(plateau_val)

    def configure_optimizers(self):

        """ https://discuss.pytorch.org/t/weight-decay-in-the-optimizers-is-a-bad-idea-especially-with-batchnorm/16994/3 """
        def add_weight_decay(modules, weight_decay, lr):
            decay = []
            no_decay = []
            for m in modules:
                for name, param in m.named_parameters():
                    if param.requires_grad:
                        if len(param.shape) == 1:  # Bias and bn parameters
                            no_decay.append(param)
                        else:
                            decay.append(param)
            return [{'params': no_decay, 'lr': lr, 'weight_decay': 0.0},
                    {'params': decay, 'lr': lr, 'weight_decay': weight_decay}]

        # Get the parameters
        decoder_modules = [self.init_lstm, self.lstm, self.attention, self.beta, self.deep_output]
        params = add_weight_decay(decoder_modules, self.hparams.weight_decay, self.hparams.decoder_lr)
        if self.hparams.embedding_lr>0:
            # I don't think embeddings use weight_decay because they might be normalized
            params += [{'params': self.embedding.parameters(), 'lr': self.hparams.embedding_lr, 'weight_decay': 0.0}]

        # If either of these are true, then there is finetuning
        if self.hparams.encoder_finetune_after>0 and self.hparams.encoder_lr>0:
            params += add_weight_decay([self.encoder], self.hparams.weight_decay, self.hparams.encoder_lr)

        # NOTE : decoder_lr is provide as the required lr argument
        if self.hparams.opt == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.hparams.decoder_lr, momentum=self.hparams.momentum, nesterov=self.hparams.nesterov)
        elif self.hparams.opt == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.hparams.decoder_lr, betas=(self.hparams.adam_b1, self.hparams.adam_b2))
        elif self.hparams.opt == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=self.hparams.decoder_lr, betas=(self.hparams.adam_b1, self.hparams.adam_b2))

        # Keep a copy of the initial lr for each group because this will get overwritten during warmup steps
        self.opt_init_lr = [pg['lr'] for pg in optimizer.param_groups]

        if self.hparams.scheduler == 'step':
            self.scheduler = MultiStepLR(
                optimizer,
                milestones=self.hparams.milestones,
                gamma=self.hparams.lr_gamma)
        elif self.hparams.scheduler == 'plateau':
            self.scheduler = ReduceLROnPlateau(
                optimizer,
                mode='max',
                factor=self.hparams.lr_gamma,
                patience=self.hparams.plateau_patience,
                min_lr=self.hparams.min_lr)
        elif self.hparams.scheduler == 'exp':
            self.scheduler = ExponentialLR(
                optimizer,
                gamma=self.hparams.lr_gamma)
        elif self.hparams.scheduler == 'cosine':
            # I thought to divide by accumulate, but cosine is stepped every epoch so no need
            adj_steps = self.hparams.epochs*self.hparams.train_loader_len - self.hparams.lr_warmup_steps
            t0 = self.hparams.cosine_iterations
            tm = self.hparams.cosine_multi
            """ Adjust the t0 to end with a low learning rate
            Get number of restarts for specified t0 and tm, then update t0
            Under estimate the number of restarts and over estimate the t0 size
            Add the accumulate steps just in case of unfortunate rounding
            """
            if tm!=1:
                # Use the sum of geometric sequence solved for n to get the number of restarts
                # geometric sum = t0*(1-tm**n)/(1-tm)
                restarts = (torch.log(torch.tensor(1-(adj_steps*(1-tm)/t0)))/torch.log(torch.tensor(tm))).floor()
                # Divide steps by the geometric sum to get t0
                if restarts==0.0:
                    t0 = adj_steps+self.hparams.accumulate
                else:
                    t0 = ((adj_steps+self.hparams.accumulate)/((1-tm**restarts)/(1-tm))).ceil()
            else:
                restarts = torch.tensor(adj_steps/t0).floor()
                if restarts==0.0:
                    t0 = adj_steps+self.hparams.accumulate
                else:
                    t0 = ((adj_steps+self.hparams.accumulate)/restarts).ceil()
            self.scheduler = CosineAnnealingWarmRestarts(
                optimizer,
                T_0=int(t0),
                T_mult=int(tm),
                eta_min=self.hparams.min_lr)
        elif self.hparams.scheduler == 'one_cycle':
            self.hparams.lr_warmup_steps = 0  # Set this to zero
            self.scheduler = OneCycleLR(
                optimizer,
                self.opt_init_lr,
                epochs=self.hparams.epochs,
                steps_per_epoch=self.hparams.train_loader_len,
                pct_start=self.hparams.one_cycle_pct,
                cycle_momentum=False,
                div_factor=self.hparams.one_cycle_div,
                final_div_factor=self.hparams.one_cycle_fdiv)           

        return optimizer
