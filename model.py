from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.gleu_score import corpus_gleu
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import models
from torchvision.transforms import Normalize


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

        # Only freeze the weights if the model is pretrained and no fine tuning
        if args.pretrained and not args.encoder_finetune:
            for param in m.parameters():
                param.requires_grad = False

        # TODO : change back to the previous extractors 
        # Get model features before pooling and linear layers
        # Save final_dim as input to 1x1 conv to transform to encoder_dim features
        # Save final_size to see if !=args.encoder_size, used in AdaptiveAvgPool2d
        final_size = 7  # All models use 7, except squeeznets which if set in the if block
        if "resnet" in args.encoder_arch or "resnext" in args.encoder_arch:
            layers = list(m.children())[:-2]  # Remove pooling and fc
            final_dim = m.fc.in_features
        elif "shufflenet" in args.encoder_arch:
            layers = list(m.children())[:-1]  # Remove fc
            final_dim = m.fc.in_features
        elif "squeezenet" in args.encoder_arch:
            layers = list(m.children())[:-1]  # Remove classifer
            final_dim = 512
            final_size = 13
        elif "densenet" in args.encoder_arch:
            layers = list(m.children())[:-1]  # Remove classifer
            final_dim = m.classifier.in_features
        elif "mobilenet_v2" in args.encoder_arch:
            layers = list(m.children())[:-1]  # Remove classifer
            final_dim = m.classifier[1].in_features
        elif "mobilenet_v3" in args.encoder_arch:
            layers = list(m.children())[:-2]  # Remove pooling and classifer
            final_dim = m.classifier[0].in_features
        elif "mnasnet" in args.encoder_arch:
            layers = list(m.children())[:-1]  # Remove pooling and classifer
            final_dim = m.classifier[1].in_features
        else:
            raise ValueError("Encoder not supported : {}".format(args.encoder_arch))

        if args.encoder_dim is not None and args.encoder_dim!=final_dim:
            # This conv forces the number of features to be encoder_dim
            # Does not match the pretrained encoders in paper since this always requires training
            layers.append( nn.Conv2d(final_dim, args.encoder_dim, kernel_size=1, stride=1, bias=True) )
        else:
            # Store the encoder_dim back in the hparams
            args.encoder_dim = final_dim
        
        # TODO : pool->conv or conv->pool? pool will change the features locally before the conv op
        # For now conv->pool, bc conv will take the actual features as input rather than pooled features
        if args.encoder_size is not None and args.encoder_size!=final_size:
            if args.encoder_size < final_size:
                # Going to a smaller map uses average pooling
                layers.append( nn.AdaptiveAvgPool2d((args.encoder_size, args.encoder_size)) )
            elif args.encoder_size > final_size:
                # Average pooling uses nearest neighbor for upsampling, but sometimes it interpolates
                # To avoid not knowning what upsampling happens, force a behavior with nn.Upsample
                # Setting align_corners=False is the default, but it is set explicitly to suppress warnings
                layers.append( nn.Upsample((args.encoder_size, args.encoder_size), mode="bilinear", align_corners=False) )    
        else:
            # Store the encoder_size back in the hparams
            args.encoder_size = final_size

        # Layer that flattens the feature maps from 2d to 1d and then
        # reshapes so that there are L feature locations along dim=1, and
        # dim=2 is the representation with size D
        layers.append( FlattenShuffle() )
        
        # First layer of the model, input is [0,1] normalized images
        norm = Normalize(args.mean, args.std, inplace=True)

        # Returns (batch, encoder_size**2, encoder_dim)
        return nn.Sequential(norm, *layers)
    raise ValueError("Unknown model arg: {}".format(args.encoder_arch))


class InitLSTM(nn.Module):
    """ Sec 3.1.2 "predicted by an avergage of the annotation vectors feed through two separate MLPs" """
    def __init__(self, encoder_dim, decoder_dim, decoder_layers, dropout):
        super(InitLSTM, self).__init__()
        self.decoder_dim = decoder_dim
        self.decoder_layers = decoder_layers
        self.init_h = nn.Linear(encoder_dim, decoder_dim*decoder_layers)
        self.init_c = nn.Linear(encoder_dim, decoder_dim*decoder_layers)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, annotations):
        # Mean over the locations (dim=1) to get a vector of length encoder_dim
        mean = annotations.mean(1)
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
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(hidden_dim, attention_dim)
        self.f_att = nn.Linear(attention_dim, 1)

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
    def __init__(self, encoder_dim, decoder_dim, embed_dim, vocab_size, dropout):
        super(DeepOutput, self).__init__()
        self.hidden = nn.Linear(decoder_dim, embed_dim)
        self.context = nn.Linear(encoder_dim, embed_dim)
        self.output = nn.Linear(embed_dim, vocab_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, prev_embed, hidden, context):
        # logit.shape = (batch, embed_dim)
        logit = prev_embed + self.hidden(hidden) + self.context(context)
        # torch.cat([prev_embed, self.hidden(hidden), self.context(context)], dim=1)
        logit = self.dropout(torch.tanh(logit))
        return self.output(logit)


class SAT(pl.LightningModule):
    """ Shot, Attend, and Tell (SAT)
    Notes:
    -all image encoding is done in self.encoder using convolutional networks
    -all text decoding is done in this LightningModule (ie, beam search)
    """
    def __init__(self, **kwargs):
        super(SAT, self).__init__()
        self.save_hyperparameters()
        self.scheduler = None  # Set in configure_optimizers()

        # Keep these, a nice list to have around
        self.special_idxs = [self.stoi("<PAD>"), self.stoi("<START>"), self.stoi("<END>")]

        # Call the function to get the encoder architecture.
        self.encoder = get_encoder(self.hparams)

        # This is the size of the flattened encoding annotations.
        # attention_dim is the same as L in the paper.
        self.hparams.attention_dim = self.hparams.encoder_size**2

        # Sec 3.1.2 - Matrix E is the embedding matrix of shape (vocab_size, embed_dim).
        self.embedding = nn.Embedding(
            num_embeddings=self.hparams.vocab_size, embedding_dim=self.hparams.embed_dim,
            max_norm=self.hparams.embed_norm,
            padding_idx=self.stoi("<PAD>")
        )

        # Sec 3.1.2 - Intialize the hidden states based on the mean annotations and two separate MLPs.
        self.init_lstm = InitLSTM(self.hparams.encoder_dim, self.hparams.decoder_dim,
                                self.hparams.decoder_layers, self.hparams.dropout)

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
        # This is outside of the attention module bc the zt vector is needed for deep output
        self.beta = nn.Linear(self.hparams.decoder_dim, self.hparams.encoder_dim)

        # Sec 3.1.2 - Deep Output for word prediction
        self.deep_output = DeepOutput(self.hparams.encoder_dim, self.hparams.decoder_dim,
                                    self.hparams.embed_dim, self.hparams.vocab_size, self.hparams.dropout)

    def stoi(self, s):
        return int(self.hparams.vocab_stoi.get(s, self.hparams.vocab_stoi['<UNK>']))

    def itos(self, i):
        return str(self.hparams.vocab_itos.get(int(i), "<UNK>"))

    def decode_seq(self, seq, remove_special=False):
        """ Convert a list of int into a list of str. """
        keep_token = lambda x: not (remove_special and x in self.special_idxs)
        # Using str() just in case
        return [self.itos(t) for t in seq if keep_token(t)]

    def caption(self, img_tensor, beamk=3, max_gen_length=32, temperature=0.5, 
                rescore_method=None, rescore_reward=0.5, return_all=False):
        """ Caption method for better code readability.
            Input img as a batch [B, C, H, W].    
            Output is a list of strings with len()=B.
            rescore_method : None, LN=Length Normalization, WR=Word Reward, BAR=Bounded Adaptive-Reward
            rescore_reward : should be tuned on a dev set
        """
        return self.forward(img_tensor, beamk, max_gen_length,temperature, rescore_method, rescore_reward, return_all)

    def forward(self, img, beamk=3, max_gen_length=32, temperature=0.5, 
                rescore_method=None, rescore_reward=0.5, return_all=False):
        """ Inference Method Only. Use beam search to create a caption. """
        self.eval()  # Freeze all parameters and turn off dropout

        beamk_arg = beamk  # Keep this arg since beamk get's overwritten

        # Makes single values a list so it can be indexed at each step
        if not isinstance(temperature, list):
            temperature = [temperature]

        # Create empty return lists, will become the length of the batch
        captions, cap_scores, cap_alphas, cap_perplexity = [], [], [], []

        with torch.no_grad():
            # Encode the batch
            annotations = self.encoder(img)

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
                top_preds = torch.LongTensor([[self.hparams.vocab_stoi["<START>"]]*beamk]).to(self.device)

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
                    beta = torch.sigmoid(self.beta(h[-1]))
                    h_in = torch.cat([embed_prev_words, beta*zt], dim=1).unsqueeze(0)
                    _, (h, c) = self.lstm(h_in, (h, c))
                    logit = self.deep_output(embed_prev_words, h[-1], zt)

                    # Use the log probability as the score
                    scores = F.log_softmax(logit/current_temperature, dim=1)
                    
                    # TODO : mask the <START> and <PAD> token, right?
                    scores[:, [self.stoi("<START>"), self.stoi("<PAD>")]] = float('-inf')

                    if step==0:
                        # Additionally mask the <END> on the first step
                        scores[:, self.stoi("<END>")]= float('-inf')
                        # Extract the top beamk words for the first image since
                        # the initial predicitons are all the same across the beam batch
                        top_scores, pred_idx = torch.topk(scores[0], beamk)
                        # Extend each top sequence with the top predictions
                        top_preds = torch.cat([top_preds, pred_idx.unsqueeze(0)], 0)
                        # Extend the alphas
                        alphas = torch.cat([alphas, alpha.unsqueeze(0)], 0)
                    else:
                        # Compute the scores for all possible next words
                        scores = scores + top_scores.unsqueeze(1)

                        # Flatten the scores so all scores are in dim=0
                        scores = scores.reshape(-1)
                        probs, pred_idx = torch.topk(scores, beamk, dim=0)
                        
                        # This is the index of which sequence in the beam batch with the top beamk scores
                        # Need this to select which sequence to cat the word and to sum the score
                        keep_idxs = pred_idx//self.hparams.vocab_size

                        # Update the top_scores by taking the top beamk scores
                        top_scores = scores[pred_idx]

                        # Update the sequences by taking the top beamk scores and cat the word indexes
                        # top_preds[:,keep_idxs] = take every timestep for the top beamk scores
                        # pred_idx%self.hparams.vocab_size = vocab_size index rather than the flattened beam batch index
                        top_preds = torch.cat([top_preds[:,keep_idxs], (pred_idx%self.hparams.vocab_size).unsqueeze(0)], 0)
                        alphas = torch.cat([alphas[:,keep_idxs], alpha.unsqueeze(0)[:,keep_idxs]], 0)
 
                        # Update other variables for the selected top beamk scores
                        h, c = h[:, keep_idxs], c[:, keep_idxs]
                        annots = annots[keep_idxs]
                    
                    # Now the top_preds,top_scores,alphas,h,c,annots are updated with the latest sequences
                    # Check if any of the sequences predicted the <END> token and remove them
                    complete = top_preds[step+1]==self.stoi("<END>")

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

    def training_step(self, batch, batch_idx):
        """ Method used only during training. """
        # Unpack, pl puts the data on the correct device already
        img, encoded_captions, lengths = batch

        # Encode the images into annotation vectors
        # annotations.shape = (batch, attention_dim, encoder_dim)
        annotations = self.encoder(img)

        # The design of my dataset ouputs dim=1 as the number of captions, but
        # since in training this is always size 1, we can remove dim=1 
        encoded_captions = torch.squeeze(encoded_captions, dim=1)
        # Same with lengths, except remove the batch dimension so it is just a 1D vector
        lengths = torch.squeeze(lengths)
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
            mask_idxs = lengths>step
            if not any(mask_idxs): break  # All the captions are done

            # TODO : teacher forcing schedule

            # On the fist step get the start tokens, for step>0 check decoder_tf
            if step==0 or self.hparams.decoder_tf:
                # Get the actual next word embedding
                embed_prev_words = self.embedding(encoded_captions[mask_idxs, step])
            else:
                # Get the argmax of the logits
                idxs = torch.argmax(logits[mask_idxs, step-1, :], dim=1)
                # Get the predicted next word embedding
                embed_prev_words = self.embedding(idxs)

            # Compute the attention context vector from the annotations and the previous hidden state
            # zt.shape, alpha.shape = (batch, encoder_dim)
            zt, alpha = self.attention(annotations[mask_idxs], h[-1, mask_idxs])            
            # Save the alpha values
            alphas[mask_idxs, step, :] = alpha

            # Compute the gating scalar beta from the previous hidden state
            # beta.shape = (batch, encoder_dim). Notice this matches zt so we can do the element wise product
            beta = torch.sigmoid(self.beta(h[-1, mask_idxs].squeeze(1)))  # squeeze to remove the length dim

            # Prepare to enter the lstm with shape (1, batch, embed_dim+enocder_dim)
            h_in = torch.cat([embed_prev_words, beta*zt], dim=1).unsqueeze(0)

            # Compute the new hidden states
            _, (h[:, mask_idxs], c[:, mask_idxs]) = self.lstm(h_in, (h[:, mask_idxs], c[:, mask_idxs]))

            # Compute the word logits
            logit = self.deep_output(embed_prev_words, h[-1, mask_idxs], zt)
            logits[mask_idxs, step, :] = logit.clone().float()  # clone because float() is an inplace operation

        # End of the loop

        # Use the logits to compute the word prediction loss
        logits_packed = pack_padded_sequence(logits, lengths.tolist(), batch_first=True, enforce_sorted=False).data
        targets_packed = pack_padded_sequence(targets, lengths.tolist(), batch_first=True, enforce_sorted=False).data

        loss = F.cross_entropy(logits_packed, targets_packed)
        # Sec 4.2.1 Equation 14 - doubly stochastic loss
        loss += self.hparams.att_gamma*((1-alphas.sum(dim=1))**2).mean()

        pred = torch.argmax(logits_packed, dim=1)
        acc = torch.sum(pred==targets_packed)/pred.shape[0]
        # Create metrics dict
        metrics = {
            "loss": loss,
            "accuracy": acc,
        }

        # Update tensorboard for each train step
        for k, v in metrics.items():
            key = "{}/train".format(k)
            val = metrics[k].detach().item()
            self.logger.experiment.add_scalar(key, val, global_step=self.global_step)
        return metrics

    def training_epoch_end(self, outputs):
        # Calculating epoch metrics
        for k in outputs[0].keys():
            key = "{}/train_epoch".format(k)
            val = torch.stack([x[k] for x in outputs]).mean().detach().item()
            self.logger.experiment.add_scalar(key, val, global_step=self.current_epoch+1)

        # Log lr
        lr = self.optimizers().param_groups[0]['lr']
        self.logger.experiment.add_scalar('Learning Rate', lr, global_step=self.current_epoch+1)
    
    def score_captions(self, encoded_captions, lengths, captions, perplexities=None):
        # Remove <PAD> <START> <END>
        references = [[c[1:l] for c,l in zip(refs, lengths[i])] for i,refs in enumerate(encoded_captions.tolist())]

        # Same input for all these metrics, nice
        bleu1 = corpus_bleu(references, captions, weights=(1, 0, 0, 0))
        bleu2 = corpus_bleu(references, captions, weights=(0.5, 0.5, 0, 0))
        bleu3 = corpus_bleu(references, captions, weights=(0.33, 0.33, 0.33, 0))
        bleu4 = corpus_bleu(references, captions, weights=(0.25, 0.25, 0.25, 0.25))
        gleu = corpus_gleu(references, captions)

        # TODO : does this idea work?
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
            "gleu": gleu,
            "cosine_similarity": cosine_similarity.item()
        }
        if type(perplexities)==list: metrics.update({"perplexity": sum(perplexities)/len(perplexities)})
        return metrics

    def val_batch(self, batch, beamk=3, max_gen_length=32, temperature=0.5, rescore_method=None, rescore_reward=0.5):
        img, encoded_captions, lengths = batch  # Unpack
        captions, scores, alphas, perplexities = self.caption(img, beamk, max_gen_length, temperature, rescore_method, rescore_reward)
        metrics = self.score_captions(encoded_captions, lengths, captions, perplexities)
        return metrics

    def validation_step(self, batch, batch_idx):
        """ Method used only for validation. """
        return self.val_batch(batch, beamk=self.hparams.val_beamk, max_gen_length=self.hparams.val_max_len, rescore_method="LN")

    def validation_epoch_end(self, outputs): 
        # Calculate epoch metrics 
        for k in outputs[0].keys():
            key = "{}/val_epoch".format(k)
            vals = [x[k] for x in outputs]
            try:  # precision and recall fail for short captions
                val = sum(vals) / len(vals)
            except:
                val = 0
            self.logger.experiment.add_scalar(key, val, global_step=self.current_epoch+1)
            # Use self.log() for the checkpoint callbacks
            if k==self.hparams.save_monitor: self.log(k, val)
            if k==self.hparams.early_stop_monitor: self.log(k, val)
            if k==self.hparams.plateau_monitor: plateau_val = val  # For plateau scheduler

        # Step scheduler
        if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
            self.scheduler.step(plateau_val)
        elif type(self.scheduler) in [torch.optim.lr_scheduler.MultiStepLR, torch.optim.lr_scheduler.ExponentialLR]:
            self.scheduler.step()

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

        decoder_modules = [self.init_lstm, self.lstm, self.attention, self.beta, self.deep_output]
        if self.hparams.weight_decay != 0:
            params = add_weight_decay(decoder_modules, self.hparams.weight_decay, self.hparams.decoder_lr)
            # I don't think embeddings use weight_decay
            params += [{'params': self.embedding.parameters(), 'lr': self.hparams.decoder_lr}]
            params += add_weight_decay([self.encoder], self.hparams.weight_decay, self.hparams.encoder_lr)
        else:
            params = [{'params': self.embedding.parameters(), 'lr': self.hparams.decoder_lr}]
            params += [{'params': m.parameters(), 'lr': self.hparams.decoder_lr} for m in decoder_modules]
            params += [{'params': self.encoder.parameters(), 'lr': self.hparams.encoder_lr}]

        # NOTE : decoder_lr is provide as the required lr argument
        if self.hparams.opt == 'sgd':
            optimizer = torch.optim.SGD(params, lr=self.hparams.decoder_lr, momentum=self.hparams.momentum, nesterov=self.hparams.nesterov)
        elif self.hparams.opt == 'adam':
            optimizer = torch.optim.Adam(params, lr=self.hparams.decoder_lr, betas=(self.hparams.adam_b1, self.hparams.adam_b2))
        elif self.hparams.opt == 'adamw':
            optimizer = torch.optim.AdamW(params, lr=self.hparams.decoder_lr, betas=(self.hparams.adam_b1, self.hparams.adam_b2))

        if self.hparams.scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.milestones, gamma=self.hparams.lr_gamma)
        elif self.hparams.scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=self.hparams.lr_gamma, patience=self.hparams.plateau_patience)
        elif self.hparams.scheduler == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.lr_gamma)

        return optimizer
