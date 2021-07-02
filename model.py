from nltk.metrics.scores import accuracy as corpus_accuracy
from nltk.metrics.scores import precision, recall
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.chrf_score import corpus_chrf
from nltk.translate.gleu_score import corpus_gleu
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F, modules
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
        b, c, s1, s2 = x.shape
        x = x.reshape(b, c, s1*s2)  # Flatten the feature maps to 1 dimension
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

        # Get model features before pooling and linear layers
        # Save final_dim as input to 1x1 conv to transform to encoder_dim features
        if "resnet" in args.encoder_arch or "resnext" in args.encoder_arch:
            layers = list(m.children())[:-2]  # Remove pooling and fc
            final_dim = m.fc.in_features
        elif "shufflenet" in args.encoder_arch:
            layers = list(m.children())[:-1]  # Remove fc
            final_dim = m.fc.in_features
        elif "squeezenet" in args.encoder_arch:
            layers = list(m.children())[:-1]  # Remove classifer
            final_dim = 512  # Always 512
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

        # Combine into a sequential model
        norm = Normalize(args.mean, args.std, inplace=True)
        # avg->conv or conv->avg? Avg will change the features locally before the conv op
        layers.append( nn.AdaptiveAvgPool2d((args.encoder_size, args.encoder_size)) )

        if args.encoder_dim is not None and args.encoder_dim!=final_dim:
            # This conv forces the number of features to be encoder_dim
            # Does not match the paper since this always requires training
            layers.append( nn.Conv2d(final_dim, args.encoder_dim, kernel_size=1, stride=1, bias=True) )
        else:
            # Store the encoder dimensions back in the hparams
            args.encoder_dim = final_dim

        # Layer that flattens the feature maps from 2d to 1d and then
        # reshapes so that there are L feature locations along dim=1, and
        # dim=2 is the representation with size D
        shuffle = FlattenShuffle()
        
        # Returns (batch, encoder_size**2, encoder_dim)
        return nn.Sequential(norm, *layers, shuffle)
    raise ValueError("Unknown model arg: {}".format(args.encoder_arch))


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
        # Softamax over the all locations(dim=1)
        alpha = F.softmax(att, dim=1)  # alpha.shape = (batch, attention_dim, 1)
        # Apply the alphas to the annotations, then sum each feature map over all locations(dim=1)
        zt = (annotations*alpha).sum(dim=1)  # zt.shape = (batch, encoder_dim)
        return zt, alpha.reshape(alpha.shape[0], -1)


class SAT(pl.LightningModule):
    """ Shot, Attend, and Tell (SAT)
    Notes:
    -all image encoding is done in self.encoder using convolutional networks
    -all text decoding is done in this module (beam search, metrics)
    """
    def __init__(self, **kwargs):
        super(SAT, self).__init__()
        self.save_hyperparameters()
        self.scheduler = None  # Set in configure_optimizers()

        # Keep these, a nice list to have around
        self.special_idxs = [self.stoi("<PAD>"), self.stoi("<START>"), self.stoi("<END>")]

        # This is the size of the flattened encoding annotations.
        # attention_dim is the same as L in the paper.
        self.hparams.attention_dim = self.hparams.encoder_size**2

        # hidden_dim is the decoder_dum times the decoder_layers
        # self.hparams.hidden_dim = self.hparams.decoder_dim*self.hparams.decoder_layers

        # Call the function to get the encoder architecture.
        self.encoder = get_encoder(self.hparams)

        # self.decoder = Decoder(self.hparams)

        # Dropout is used before intializing the lstm and before projecting to the vocab.
        self.dropout = nn.Dropout(p=self.hparams.dropout)

        # Sec 3.1.2 - Matrix E is the embedding matrix of shape (vocab_size, embed_dim).
        self.embedding = nn.Embedding(
            num_embeddings=self.hparams.vocab_size, embedding_dim=self.hparams.embed_dim,
            max_norm=self.hparams.embed_norm,
            padding_idx=self.stoi("<PAD>")
        )

        # Sec 3.1.2 - Intialize the hidden states based on the mean annotations.
        # Use 2 separate linear layers for hidden and cell state.
        self.init_h = nn.Linear(self.hparams.encoder_dim, self.hparams.decoder_dim)
        self.init_c = nn.Linear(self.hparams.encoder_dim, self.hparams.decoder_dim)

        # Sec 3.1.2 Equations 1, 2, 3 - Apply the LSTM update rules.
        # Per the paper, input_size is encoder_dim+embed_dim (D+m) and
        # the hidden_size is decoder_dim (n).
        self.lstm = nn.LSTMCell(
            input_size=self.hparams.embed_dim+self.hparams.encoder_dim,
            hidden_size=self.hparams.decoder_dim,
            bias=True
        )
        # TODO : nn.LSTM() to add layers, change the size of the init_lstm linear layers
        # self.lstm = nn.LSTM(
        #     input_size=self.hparams.embed_dim+self.hparams.encoder_dim,
        #     hidden_size=self.hparams.decoder_dim,
        #     num_layers=self.hparams.decoder_layers,
        #     bias=True,
        #     batch_first=True,
        #     dropout=self.hparams.dropout
        # )

        # Sec 3.1.2 Equations 4, 5, 6 - Soft Attention Module
        self.attention = SoftAttention(
            attention_dim=self.hparams.attention_dim,
            encoder_dim=self.hparams.encoder_dim,
            hidden_dim=self.hparams.decoder_dim
        )

        # Sec 4.2.1 - predict gating scalar \beta from previous hidden state
        self.beta = nn.Linear(self.hparams.decoder_dim, self.hparams.encoder_dim)

        # Sec 3.1.2 - Deep Output for word prediction
        # TODO : change to use Equation 7
        self.deep_output = nn.Linear(self.hparams.decoder_dim, self.hparams.vocab_size)

    def stoi(self, s):
        return int(self.hparams.vocab_stoi.get(s, self.hparams.vocab_stoi['<UNK>']))

    def itos(self, i):
        return str(self.hparams.vocab_itos.get(int(i), "<UNK>"))

    def decode_seq(self, seq, remove_special=False):
        """ Convert a list of int into a list of str. """
        keep_token = lambda x: not (remove_special and x in self.special_idxs)
        # Using str() just in case
        return [self.itos(t) for t in seq if keep_token(t)]

    def init_lstm(self, annotations):
        """ Sec 3.1.2 "predicted by an avergage of the
            annotation vectors feed through two separate MLPs" """
        # Mean over the locations (dim=1) to get a vector of length encoder_dim
        mean = self.dropout(annotations.mean(1))
        # TODO : Add torch.tanh()? rn these are unbounded
        return self.init_h(mean), self.init_c(mean)
        # return torch.tanh(self.init_h(mean)), torch.tanh(self.init_c(mean))

    def caption(self, img_tensor, beamk=3, max_gen_length=32, temperature=0.5, return_all=False,
                rescore_method=None, rescore_reward=0.5):
        """ Caption method for better code readability.
            Input img as a batch [B, C, H, W].    
            Output is a list of strings with len()=B.
            rescore_method : None, LN=Length Normalization, WR=Word Reward, BAR=Bounded Adaptive-Reward
            rescore_reward : should be tuned on a dev set
        """
        return self.forward(img_tensor, beamk, max_gen_length,temperature, return_all, rescore_method, rescore_reward)

    def forward(self, img, beamk=3, max_gen_length=32, temperature=0.5, return_all=False,
                rescore_method=None, rescore_reward=0.5):
        """ Inference Method Only. Use beam search to create a caption. """
        self.eval()  # Freeze all parameters and turn off dropout

        beamk_arg = beamk  # Keep this arg since beamk get's overwritten

        # Makes single values a list so it can be indexed at each step
        if not isinstance(temperature, list):
            temperature = [temperature]

        captions, cap_scores, cap_alphas = [], [], []  # Create empty return lists, will become the length of the batch

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
                top_scores = torch.zeros(beamk, dtype=torch.float)

                # Keep the alphas for visualization
                # alphas.shape = (step, beamk, attention_dim)
                alphas = torch.zeros(1, beamk, self.hparams.attention_dim).to(self.device)

                # Extend these lists once the <END> is predicted or max_gen_length is reached
                finished_captions = []
                finished_scores = []
                finished_alphas = []

                step = 0
                while True:
                    # Get the current sampling temperature
                    current_temperature = temperature[step % len(temperature)]

                    # Get the last predictions
                    k_prev_words = top_preds[step]

                    # Forward pass through the model
                    embeddings = self.embedding(k_prev_words)
                    zt, alpha = self.attention(annots, h)
                    beta = torch.sigmoid(self.beta(h))
                    h, c = self.lstm(torch.cat([embeddings, beta*zt], dim=1), (h, c))
                    logit = self.deep_output(h)

                    # Use the log probability as the score
                    scores = F.log_softmax(logit/current_temperature, dim=1)
                    
                    # TODO : mask the <START> and <PAD> token, right?
                    scores[:, [self.stoi("<START>"), self.stoi("<PAD>")]] = float('-inf')

                    if step==0:
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
                        h, c = h[keep_idxs], c[keep_idxs]
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

                        # Reduce the beam batch to only the incomplete sequences
                        incomplete = torch.logical_not(complete)
                        top_preds = top_preds[:,incomplete]
                        alphas = alphas[:,incomplete]
                        top_scores = top_scores[incomplete]
                        h, c = h[incomplete], c[incomplete]
                        annots = annots[incomplete]

                        # Update the beamk value
                        beamk = sum(incomplete)

                        if beamk==0: break  # Leave once all sequence end

                    # Add the incomplete sequences to the output
                    if step >= max_gen_length:
                        finished_captions.extend([top_preds[:,i][1:-1].tolist() for i in range(top_preds.shape[1])])
                        finished_alphas.extend([alphas[:,i][1:-1].cpu() for i in range(alphas.shape[1])])
                        finished_scores.extend([rescore(top_scores[i], rescore_method, rescore_reward).tolist() for i in range(top_scores.shape[0])])
                        break

                    step += 1  # All updates are complete, iterate the step count

                # End of beam search

                if return_all:
                    # Sort by scores before returning
                    score_index = [[finished_scores[i], i] for i in range(len(finished_scores))]
                    score_index.sort(reverse=True)
                    score_index = [x[1] for x in score_index]
                    captions.append([finished_captions[i] for i in score_index])
                    cap_scores.append([finished_scores[i] for i in score_index])
                    cap_alphas.append([finished_alphas[i] for i in score_index])                
                else:
                    best_idx = finished_scores.index(max(finished_scores))
                    captions.append(finished_captions[best_idx])
                    cap_scores.append(finished_scores[best_idx])
                    cap_alphas.append(finished_alphas[best_idx])

            # End of the img batch loop
        
        # Return these lists, which are the length of the batch
        return captions, cap_scores, cap_alphas

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
        # Convert the token indexes into embedding vectors
        # embeddings.shape = (batch, max_cap_length, embed_dim)
        embeddings = self.embedding(encoded_captions)

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
            mask_idxs = lengths>step  # Squeeze to reduce to 1D
            if not any(mask_idxs): break  # All the captions are done

            # Compute the attention contex vector from the annotations and the previous hiden state
            # zt.shape, alpha.shape = (batch, encoder_dim)
            zt, alpha = self.attention(annotations[mask_idxs], h[mask_idxs])
            # Save the alpha values
            alphas[mask_idxs, step, :] = alpha

            # Compute the gating scalar beta from the previous hidden state
            # beta.shape = (batch, encoder_dim). Notice this matches zt so we can do the element wise product
            beta = torch.sigmoid(self.beta(h[mask_idxs, :]))

            # Compute the new hidden states
            h[mask_idxs], c[mask_idxs] = self.lstm(
                torch.cat([embeddings[mask_idxs, step, :], beta*zt], dim=1),  # dim=0 is batch, so concatenate along dim=1
                (h[mask_idxs], c[mask_idxs])
            )

            # Compute the word logits
            logit = self.deep_output(self.dropout(h[mask_idxs]))
            logits[mask_idxs, step, :] = logit

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
            if k=="loss": avg_loss = val  # For plateau scheduler

        # Log lr
        lr = self.optimizers().param_groups[0]['lr']
        self.logger.experiment.add_scalar('Learning Rate', lr, global_step=self.current_epoch+1)

        # Step scheduler
        if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
            self.scheduler.step(avg_loss)
        elif type(self.scheduler) in [torch.optim.lr_scheduler.MultiStepLR, torch.optim.lr_scheduler.ExponentialLR]:
            self.scheduler.step()
    
    def validation_step(self, batch, batch_idx):
        """ Method used only for validation. """
        img, encoded_captions, lengths = batch  # Unpack
        captions, scores, alphas = self.caption(img, beamk=self.hparams.val_beamk, max_gen_length=self.hparams.val_max_len, rescore_method="LN")

        # Remove <PAD> <START> <END>
        references = [[c[1:l] for c,l in zip(refs, lengths[i])] for i,refs in enumerate(encoded_captions.tolist())]

        # Same input for all these metrics, nice
        bleu1 = corpus_bleu(references, captions, weights=(1, 0, 0, 0))
        bleu2 = corpus_bleu(references, captions, weights=(0.5, 0.5, 0, 0))
        bleu3 = corpus_bleu(references, captions, weights=(0.33, 0.33, 0.33, 0))
        bleu4 = corpus_bleu(references, captions, weights=(0.25, 0.25, 0.25, 0.25))
        gleu = corpus_gleu(references, captions)
        gleu = corpus_gleu(references, captions)

        # Convert to list of strings for the CHRF (Character n-gram F-score)
        captions_str, references_str = [], []
        for refs, c in zip(references, captions):
            str_cap = self.decode_seq(c, remove_special=True)
            for r  in refs:
                str_ref = self.decode_seq(r, remove_special=True)
                references_str.append(str_ref)
                captions_str.append(str_cap)
                # compute the accuracy and keep the highest one

        # recall is considered beta times as important as precision
        chrf = corpus_chrf(references_str, captions_str, beta=3)

        # acc = corpus_accuracy(references_str, captions_str)  # Hmm, this doesn't tell much

        captions_set = set([x for c in captions for x in c])
        references_set = set(encoded_captions.reshape(-1).tolist()) # - set(self.special_idxs)
        prec = precision(references_set, captions_set)
        reca = recall(references_set, captions_set)

        # TODO : Not sure how to do the loss against multiple targets
        # Create metrics dict
        metrics = {
            # "accuracy": acc,
            "chrf": chrf,
            "bleu1": bleu1,
            "bleu2": bleu2,
            "bleu3": bleu3,
            "bleu4": bleu4,
            "gleu": gleu,
            "precision": prec,
            "recall": reca,
        }
        return metrics

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

        decoder_modules = [self.init_h, self.init_c, self.lstm, self.attention, self.beta, self.deep_output]
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
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=self.hparams.lr_gamma, patience=self.hparams.patience, verbose=True)
        elif self.hparams.scheduler == 'exp':
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.hparams.lr_gamma)

        return optimizer
