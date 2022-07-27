
# Motivation

I am fascinated by image captioning because it combines the knowledge of computer vision and natural language processing.

# Main Paper

Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhudinov, R., Zemel, R. &amp; Bengio, Y.. (2015). Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. *Proceedings of the 32nd International Conference on Machine Learning*, in *Proceedings of Machine Learning Research* 37:2048-2057. [[arXiv](https://arxiv.org/abs/1502.03044)]

Paper notes:
- The image is passed through as CNN encoder to create annotations with L locations and D features at each location.
- The LSTM decoder is initialized by passing the mean of the annotation features into separate linear projections for the hidden and cell state.
- At each step, an additive attention mechanism creates a weight for L locations in the annotations. This weight can be thought of as the relative importance of the location. In Soft Attention, the annotations and attention weights are multiplied at each location and then the summation over each weighted feature map creates a single context vector of size D features.
- The context vector and the previous word embedding are concatenated together as input to the LSTM.
- A deep output layer projects the previous word embedding, hidden state, and context vector into the embedding dimension. These vectors are summed and passed through a projection to create logits over the vocab size.
- The model is trained end-to-end using a cross entropy loss of the word prediction at each step.
- An additional doubly stochastic loss is applied. This takes the L2 difference of the sum of attention weights over all timesteps and 1. "This can be interpreted as encouraging the model to pay equal attention to every part of the image over the course of generation." (Sec 4.2.1)
- Use dropout and early stopping on Bleu score.

Original Source Code. kelvinxu, arctic-captions (https://github.com/kelvinxu/arctic-captions)


# Setup

Create a new conda environment
```
conda create --name pytorch python=3.9
```

```
conda activate pytorch
```

Get pytorch installed. Command generated here: https://pytorch.org/
```
conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
```

Requirements
```
pip install -r requirements.txt
```

Create the interactive notebook kernel:

```
conda install ipykernel jupyter
```

```
python -m ipykernel install --user --name pytorch --display-name "pytorch"
```


# Dataset

## MS COCO (Microsoft Common Objects in Context)

### Links
- http://images.cocodataset.org/zips/train2014.zip (12.7 GB)
- http://images.cocodataset.org/zips/val2014.zip (6.25 GB)
- http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip (137 MB)

I downsampled the images to 256 to speed up the image loading pipeline. You can see how that is done in [https://github.com/Lukeasargen/GarbageML/blob/main/resize.py](https://github.com/Lukeasargen/GarbageML/blob/main/resize.py).


Use [preprocess.ipynb](preprocess.ipynb) to build a tokenized version of the dataset. It saves the image path paired with captions in a single json file. The captions are tokenized and padded. It also saves a dictionary mapping word strings to token indexes called "vocab_stoi".

### Splits
|  | train | restval | val | test | train+restval | total |
|-|-|-|-|-|-|-|
| images | 82783 | 30504 | 5000 | 5000 | 113287 | 123287 |
| captions | 414113 | 152634 | 25010 | 25010 | 566747 | 616767 |

### Captions Counts
- 122959 images have 5 captions
- 324 images have 6 captions
- 4 images have 7 captions

### Captions Lengths (Log Scale)

![caption_length_histogram](/data/readme/caption_length_histogram.png)


In preprocessing, I combine train and restval and just call it train.
Additionally, all images will have 5 captions in order to use the built-in pytorch collate_fn.

Here are 2 test images. The captions have been decoded using a vocab size of 6400, so unknown words are decode as "UNK".

![test_samples](/data/readme/test_samples.png)


# Model

## Encoder

You can use any of these torchvision models. Just get the name string correct for the --encoder_arch argument.
```
resnet18, resnet34, resnet50, resnet101, resnet152,
resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2,
squeezenet1_0, squeezenet1_1,
densenet121, densenet169, densenet201, densenet161,
shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5*, shufflenet_v2_x2_0*
mobilenet_v2, mobilenet_v3_large, mobilenet_v3_small
mnasnet0_5, mnasnet0_75*, mnasnet1_0, mnasnet1_3*

*supported but no pretrained version available
```

The code that modifies the torchvision models is in get_encoder() in model.py. Below is a simplified snippet that does the same things, check the full function for details.
```
from torchvision import models
def get_encoder(args):
    m = models.__dict__[args.encoder_arch](pretrained=args.pretrained)
    layers = list(m.children())[:-1]  # Remove some layers
    # final_dim and final_size are set by which arch is used
    layers.append(nn.Conv2d(final_dim, args.encoder_dim, kernel_size=1, stride=1, bias=True))
    if args.encoder_size < final_size:
        layers.append(nn.AdaptiveAvgPool2d((args.encoder_size, args.encoder_size)))
    elif args.encoder_size > final_size:
        layers.append(nn.Upsample((args.encoder_size, args.encoder_size), mode="bilinear", align_corners=False))    
    layers.append(FlattenShuffle())
    norm = Normalize(args.mean, args.std, inplace=True)
    return nn.Sequential(norm, *layers)
```


## Decoder

The decoder is built during the SAT init where it assigns nn.Modules as attributes to the SAT pl.LightningModule class. Here is the minimal forward pass with a greedy decoder.
```
annotations = self.encoder(img)
h, c = self.init_lstm(annotations)
prev_words = torch.LongTensor([[self.stoi("<START>")]*img.shape[0]]).to(self.device)
for step in range(caplen):
    embed_prev_words = self.embedding(prev_words)
    zt, alpha = self.attention(annotations, h[-1])
    beta = torch.sigmoid(self.beta(h[-1]))
    h_in = torch.cat([embed_prev_words, beta*zt], dim=1).unsqueeze(0) # dim=0 is the sequence length, which is 1 in greedy decoding
    _, (h, c) = self.lstm(h_in, (h, c))
    logit = self.deep_output(embed_prev_words, h[-1], zt)
    probs = F.softmax(logit, dim=1)
    prev_words = torch.argmax(probs, dim=1)
```


# Training

The model is trained end-to-end with label smoothing cross entropy loss on the word outputs and a doubly stochastic loss (Sec 4.2.1 Equation 14).
```
loss = LabelSmoothing()(logits_packed, targets_packed)
loss += self.hparams.att_gamma*((1-alphas.sum(dim=1))**2).mean()
```

## Pytorch Lightning Utilities

The pl.Trainer implements these utilities and callbacks:
- **Gradient Clipping** - uses `nn.utils.clip_grad_norm_` and `nn.utils.clip_grad_value_`
- **Validation Interval** - `check_val_every_n_epoch`
- **Early Stopping** - `callbacks.early_stopping.EarlyStopping`
- **Model Checkpointing** - `callbacks.model_checkpoint.ModelCheckpoint`
- **Mixed Precision** - setting `precision=16` uses the native amp in PyTorch


# Training Methodology

I ran a bunch of small experiments on a subset of MSCOCO to help me understand some characteristics of the model.

## Note : The experiments 0-4 use the wrong bleu4 calculation. I forgot to remove the padding tokens. Because the model never outputs padding tokens the scores are very low. This is fixed for experiments >4.

## Experiment 0 (Adam vs Adamw) and Experiment 1 (Dropout)

- v15: --opt=adam --dropout=0.2
- v16: --opt=adamw --dropout=0.2
- v17: --opt=adam --dropout=0.5

![experiment_0_to_1](/data/readme/experiment_0_to_1.png)

Using weight decay with adamw (v16) makes no difference from regular adam (v15). Using more dropout (v17) helped this small model.

### Results: Use opt=adam and dropout=0.5.

## Experiment 2 (Pretrain and Finetune), Experiment 3 (Lower lr), and Experiment 4 (reduce encoder_dim)

- v18: --encoder_lr=4e-3 --decoder_lr=4e-3
- v19: --pretrained --decoder_lr=4e-3
- v20: --pretrained --encoder_finetune --encoder_lr=4e-3 --decoder_lr=4e-3 
- v21: --pretrained --encoder_finetune --encoder_lr=4e-3 --decoder_lr=1e-3
- v22: --pretrained --encoder_finetune --encoder_lr=1e-5 --decoder_lr=1e-3
- v23: --pretrained --encoder_dim=256 --encoder_lr=1e-5 --decoder_lr=1e-3
- v24: --pretrained --encoder_finetune --encoder_dim=256 --encoder_lr=1e-5 --decoder_lr=1e-3

![experiments_2_to_4](/data/readme/experiments_2_to_4.png)

Training from scratch (v18) is a terrible idea. A pretrained encoder does better (v19) and if you fine tune the encoder (v20) the bleu score improves. Also, lowering the learning rate on the decoder (v21) and encoder (v22) boosts performs best on this size model (hidden size of 128 and one layer).

Reducing the features by fine tuning a randomly initialized 1x1 convolution decreases the validation score (v23). If you reduce the features and fine tune the encoder (v24), you can achieve results close to a poorly fine tuned encoder (v21, where the learning rate was too high). I will note that the reduced dimensions of the output took less time (-14%) to complete the same number of steps. It also used less gpu memory.

*On fine tuning*: I believe fine tuning the encoder helps by allowing the encoder to become invariant to the new training augmentations. For example, with a frozen encoder, the decoder will be forced to learn to be invariant to the data augmentations. However, if the encoder is fine tuned, the signals from the decoder can force the encoder to learn this invariance and produce more useful features. Additionally, the pretraining dataset may not have been covered the captioning dataset in the input space and the output features will have weak representations.

*On reducing feature dimension*: Adding a new 1x1 convolution usually underperforms the pretrained features. This new layer always requires training, so performance is driven by the training data. For the most part, these features are prone to overfitting on a smaller dataset. This possibly explains why these features underperform against the features transferred from imagenet pretraining.

### Results: Use lower learning rates and finetune a pretrained encoder.

## Code improvements for experiments >4: The primary fix was to remove special tokens when computing the bleu score.

The model architecture was also improved to follow the deep output layer (Equation 7) and support a Multilayer LSTM. Adding a second layer did not help this small subset of 32k training samples (Bleu4 dropped from 18.82 from 1 layer to 18.26 for 2 layers). I set the embedding normalization. I made a bucket sampler to feed batches of similar length sequences. Additionally, I did a quick run with the single layer model and increased the attention gamma in the loss from 1 to 2; the bleu4 dropped from 18.82 to 16.41, so a attention gamma of 1 is fine. The final improvement was teacher forcing with scheduling.

## Experiment 5 (Teacher Forcing Schedule)

The value in the parentheses is a hard coded value I changed. It represents a multiplication factor, where factor*epochs is the location that the sigmoid hits 0.5. So 1.0 means at the final epoch the epsilon value is 0.5. 

- v39: bleu4=16.72, --decoder_tf=inv_sigmoid(0.5) --dropout=0.1
- v40: bleu4=20.28, --decoder_tf=inv_sigmoid(0.7) --dropout=0.0
- v41: bleu4=20.65, --decoder_tf=inv_sigmoid(0.8) --dropout=0.2
- v42: bleu4=21.79, --decoder_tf=inv_sigmoid(1.0) --dropout=0.5
- v43: bleu4=22.06, --decoder_tf=inv_sigmoid(1.2) --dropout=0.5
- v44: bleu4=20.27, --decoder_tf=always --dropout=0.5
- v45: bleu4= 8.11, --decoder_tf=None --dropout=0.5

![experiments_5_bleu4](/data/readme/experiments_5_bleu4.png)

Use v45 as a control because there was no teacher forcing. The first attempts (v39, v40, v41) did so badly I thought reducing dropout would help increase the gradient signals. It turns out that dropout was not a controlling value, as v44 always used teacher forcing and it under performs compared to the scheduled v42 and v43. The main takeaway is simply using more teacher forcing for longer improved validation bleu. 

Given how poorly the decoder does without teacher forcing, I did not test the linear and exponential schedules. Linear and exponential drop epsilon much quicker than inverse sigmoid. I think all methods would benefit from more "warm up" steps with always teacher forcing.

### Results: Use more teacher forcing for longer.

## Experiment 6 (Optical Transformations and Fine Tuning)

- v58: bleu4=23.06, --aug_optical_strength=0.0
- v60: bleu4=23.87, --aug_optical_strength=0.0 --encoder_finetune
- v61: bleu4=21.90, --aug_optical_strength=0.2
- v62: bleu4=23.71, --aug_optical_strength=0.2 --encoder_finetune

![experiments_6_bleu4](/data/readme/experiments_6_bleu4.png)

Fine tuning the encoder (v60, v62) is beneficial regardless of the input transformations. Further, with this set of 180k image-caption pairs, the performance with fine tuning was about the same for both transformation strengths. Fine tuning the shufflenet_v2_x0_5 encoder took on average 32% longer to train.

### Results: Fine tune the encoder if possible. Different image augmentations do not substantially change performance when fine tuning.

## Experiment 7 (Label Smoothing)

This experiment is about decoder performance. The encoder is frozen and the input transformations are just horizontal flip, color jitter, and gaussian noise.

- v63: bleu4=22.34, label_smoothing=0.0
- v64: bleu4=22.58, label_smoothing=0.05
- v65: bleu4=22.74, label_smoothing=0.1
- v66: bleu4=22.82, label_smoothing=0.2
- v67: bleu4=23.18, label_smoothing=0.4
- v68: bleu4=22.68, label_smoothing=0.6

![experiments_7_bleu4](/data/readme/experiments_7_bleu4.png)

A higher smoothing appears to be beneficial at higher learning rates. This is seen in the bleu4 chart early in training (before step 3k). It also looks like higher smoothing does not level off or decrease score near the end of training, the slope of line is still positive.

### Results: Any label smoothing outperforms one-hot labels. This model is robust to most smoothing values. Smooth labels help the model continue to make improvements late in training.

## Experiment 8 (Teacher Forcing and Learning Rate Schedule)

- v70: bleu4=23.79, --decoder_tf=always --scheduler=plateau
- v71: bleu4=24.08, --decoder_tf=inv_sigmoid (match v70 schedule)
- v72: bleu4=24.61, --decoder_tf=inv_sigmoid --scheduler=plateau
- v73: bleu4=22.88, --decoder_tf=inv_sigmoid --scheduler=exp --lr_gamma=0.9
- v74: bleu4=24.54, --decoder_tf=inv_sigmoid --scheduler=cosine

![experiments_8_bleu4](/data/readme/experiments_8_bleu4.png)

Always teacher forcing (v70) underperforms inv_sigmoid (v71) with the same learning rate schedules. Using a plateau scheduler and inv_sigmoid (v72) had the best bleu4 score. An exponential schedule (v73) did worst. The cosine annealing with restarts (v74) had an issue where it ended training on a high learning rate. I modified the schedule creation to always end on the lowest learning rate possible.

### Results: Inverse Sigmoid is best and use cosine or plateau schedule.

### Results: You do not need to normalize pretrained embeddings, but you should normalize randomly initialized embeddings. Bleu score is ambiguous, however the pretrained embeddings have lower perplexity.

## Summary of Experiments
- adamw with a little weight decay
- dropout pretty high. over 0.1 at minimum
- pretrained encoder and finetune with a lr around 1e-5
- decoder lr is related by batch size. with batch=160, start with lr=2e-3. if you increase the batch, you can increase the lr
- always use --deep_output. without it the decoder ignores the image
- 1 layer lstm is fine.
- --bucket_sampler saves time
- keep att_gamma at 1.0
- teacher forcing with inv_sigmoid schedule
- weak image transformations. not too much cropping
- label smoothing between 0.1 to 0.4 is beneficial
- cosine or plateau lr schedule. make sure the lr is at least 1000 times less at the end of training
- pretrained glove embeddings (non-normalized, embedding_lr=1e-5) or randomly initialized embeddings (normalized, embedding_lr=2e-2)

## Things not tested that just kind of work
- lr_warmup_steps with a few hundred steps
- --grad_clip=norm --clip_value=5.0


# Sampling a Caption

The steps for training match inference steps with slight modification to how the RNN/LSTM decoder outputs are interpreted. The output of the decoder is a probability distribution and it must be sampled to get the predicted word. The simplest method is to pick the most likely word; this is called __greedy search__. An extension of this idea is called __beam search__. During beam search, a breadth-first search is done by tracking several candidate sequences and at each step the __topk most likely are choosen__. __Greedy__ and __beam__ search always selected the most likely words, which can lead to common phrases being oversampled in the generation. I decided to also try to sample from the whole distribution using __torch.multinomial__. The results were poor at first, but I rescaled the probabilities to make a sharper distribution and the outcomes have a lot more unique words and still follow language rules.

The best way to understand the sampling process is to look in model.py in the forward method. Nearly every line is commented with an explanation.

## Results

[WIP]

models with size 128, 256, 512

# Example Predictions

## Good Predictions

![](/data/readme/predictions/COCO_val2014_000000063602_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000293810_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000366630_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000324250_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000069946_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000234902_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000301837_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000442942_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000462026_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000576566_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000100848_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000369541_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000135210_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000458567_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000036501_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000548538_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000571990_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000177015_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000519611_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000264347_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000147915_result.jpg)

![](/data/readme/predictions/test_COCO_val2014_000000296236.png)

![](/data/readme/predictions/test_COCO_val2014_000000275202.png)

![](/data/readme/predictions/test_COCO_val2014_000000373846.png)

## Inaccurate Predictions

![](/data/readme/predictions/COCO_val2014_000000558235_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000006896_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000058492_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000100594_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000273772_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000337705_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000002753_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000082933_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000042526_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000067122_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000422200_result.jpg)

## Funny Predictions

![](/data/readme/predictions/COCO_val2014_000000306664_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000209868_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000136117_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000408364_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000079369_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000045434_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000039009_result.jpg)

![](/data/readme/predictions/COCO_val2014_000000001425_result.jpg)

![](/data/readme/predictions/test_COCO_val2014_000000024343.png)

![](/data/readme/predictions/test_COCO_val2014_000000235836.png)

![](/data/readme/predictions/test_COCO_val2014_000000001573.png)

![](/data/readme/predictions/test_COCO_val2014_000000262175.png)

![](/data/readme/predictions/test_COCO_val2014_000000553678.png)

![](/data/readme/predictions/test_COCO_val2014_000000504811.png)


# References

Main Paper
- Xu, K., Ba, J., Kiros, R., Cho, K., Courville, A., Salakhudinov, R., Zemel, R. &amp; Bengio, Y.. (2015). Show, Attend and Tell: Neural Image Caption Generation with Visual Attention. *Proceedings of the 32nd International Conference on Machine Learning*, in *Proceedings of Machine Learning Research* 37:2048-2057. [[arXiv](https://arxiv.org/abs/1502.03044)]

Papers, repositories, or any site that taught me something useful:
- https://pytorch-lightning.readthedocs.io/en/latest/
- Microsoft COCO Captions : https://arxiv.org/abs/1504.00325
- https://www.nltk.org/api/nltk.translate.html - Bleu Score
- sgrvinod, a-PyTorch-Tutorial-to-Image-Captioning (https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
    - similar preprocessing
    - freeze the encoder by setting requires_grad=False in encoder init 
    - permute output of encoder
    - attention block (relu is wrong, I use tanh)
    - decoder forward with reducing batch size (my implement is using different indexing to avoid sorting)
    - gradient clipping (I used pl clipping, which calls pt clipping)
    - train.py L186 - I took the code for doubly stochastic attention regularization
    - early stopping on bleu4 metric
- AaronCCWong, Show-Attend-And-Tell (https://github.com/AaronCCWong/Show-Attend-and-Tell)
    - clean additive attention module
    - decoder.py L113 - beam search, add the scores to the output and topk over a 1d list, the use // and % to get the vocab and sequence indexes

- Ronald J. Williams, David Zipser; A Learning Algorithm for Continually Running Fully Recurrent Neural Networks. *Neural Comput* 1989; 1 (2): 270–280. doi: https://doi.org/10.1162/neco.1989.1.2.270
    - 3.2 Teacher-Forced Real-Time Recurrent Learning
- Papineni, K., Roukos, S., Ward, T. & Zhu, W.-J. (2002). BLEU: a method for automatic evaluation of machine translation. *Proceedings of the 40th annual meeting on association for computational linguistics* (p./pp. 311--318).
    - I wrote my own bleu score function using this original paper
- Pascanu, R., Gülçehre, Ç., Cho, K., & Bengio, Y. (2014). How to Construct Deep Recurrent Neural Networks. *CoRR, abs/1312.6026*. [[arXiv](https://arxiv.org/abs/1312.6026)]
    - 1 Introduction - Stacks of RNNs potentially operate at different timescales
    - 3.3.2 Deep Output RNN - intermediate layers can be used to compute the output
    - 3.3.3 Stacked RNN - all hidden states can possibly be used to compute the output, also the input can be feed to all layers, making shortcut connections
- Vinyals, O., Toshev, A., Bengio, S., & Erhan, D. (2015). Show and tell: A neural image caption generator. *2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 3156-3164. [[arXiv](https://arxiv.org/abs/1411.4555)]
    - 3.1. LSTM-based Sentence Generator - sampled with a beam width of 20
    - 4.3.1 Training Details - avoid overfitting by starting with a pretrained Encoder. pretrained word embeddings had no significant impact. use dropout. embedding and hidden size of 512
    - 4.3.3 Transfer Learning, Data Size and Label Quality - "we see gains by adding more training data since the whole process is data-driven and overfitting prone"
    - 4.3.4 Generation Diversity Discussion - "If we take the best candidate, the sentence is present in the training set 80% of the times."
- Bengio, S., Vinyals, O., Jaitly, N. & Shazeer, N. (2015). Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks.. In C. Cortes, N. D. Lawrence, D. D. Lee, M. Sugiyama & R. Garnett (eds.), *NIPS* (p./pp. 1171-1179). [[arXiv](https://arxiv.org/abs/1506.03099)]
    - 1 Introduction - discrepancy between training with teacher forcing and inference. inference mistakes made early can be amplified by pushing the model to a state space not seen during training. use a curriculum to force the model to deal with its own mistakes
    - 2.4 Bridging the Gap with Scheduled Sampling - sample the true input at each step with epsilon probability. decrease epsilon over training
    - 4.1 Image Captioning - dropout had a negative impact on metrics besides log likelihood. random sampling for an entire sequence rather than at each step had bad results 
- Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks (cite arxiv:1706.04599Comment: ICML 2017) [[arXiv](https://arxiv.org/abs/1706.04599)]
    - 4.2 Extension to Multiclass Models - "The method to get an optimal temperature T for a trained model is through minimizing the negative log likelihood for a held-out validation dataset."
- Inan, H., Khosravi, K., & Socher, R. (2017). Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling [[arXiv](https://arxiv.org/abs/1611.01462)]
    - Reusing the embedidng matrx as the output projection matrix is approximately the same as using a KL-divergence loss between y-hat prediction and y* empirical target distribution

