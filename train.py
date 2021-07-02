import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning import callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms as T

from model import SAT
from util import CocoCaptionDataset, AddGaussianNoise


def get_args():
    parser = argparse.ArgumentParser()
    # Init and setup
    parser.add_argument('--seed', default=42, type=int,
        help="int. default=42. deterministic seed. cudnn.deterministic is always set True by deafult.")
    parser.add_argument('--name', default='default', type=str,
        help="str. default=default. Tensorboard name and log folder name.")
    parser.add_argument('--workers', default=0, type=int,
        help="int. default=0. Dataloader num_workers. good practice is to use number of cpu cores.")
    parser.add_argument('--ngpu', default=1, type=int,
        help="int. default=1. number of gpus to train on. see pl docs for details.")
    parser.add_argument('--benchmark', default=False, action='store_true',
        help="store_true. set cudnn.benchmark.")
    parser.add_argument('--precision', default=32, type=int, choices=[16, 32],
        help="int. default=32. 32 for full precision and 16 uses pytorch amp")
    # Vision Dataset
    parser.add_argument('--json', type=str, required=True,
        help="str. REQUIRED. Path to json made in preproces.ipynb.")
    parser.add_argument('--mean', nargs=3, default=[0.485, 0.456, 0.406], type=float,
        help="3 floats. default is imagenet [0.485, 0.456, 0.406].")
    parser.add_argument('--std', nargs=3, default=[0.229, 0.224, 0.225], type=float,
        help="3 floats, default is imagenet [0.229, 0.224, 0.225].")
    # Vision Encoder Parameters
    parser.add_argument('--encoder_arch', default='shufflenet_v2_x0_5', type=str,
        help="str. default=shufflenet_v2_x0_5. torchvision model name.")
    parser.add_argument('--input_size', default=224, type=int,
        help="int.deafult-224. input size in pixels.")
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--encoder_finetune', default=False, action='store_true')
    parser.add_argument('--encoder_size', default=14, type=int,
        help="int. default=14. Square size of the encoder feature maps. Square this to get L in the paper.")
    parser.add_argument('--encoder_dim', default=None, type=int,
        help="int. default=None. Adds a 1x1 conv to the encoder with out_channels=encoder_dim. D in the paper.")
    # Text Decoder Parameters
    parser.add_argument('--embed_dim', default=512, type=int,
        help="int. default=512. Dimension of vocab embeddings.")
    parser.add_argument('--embed_norm', default=None, type=float,
        help="float. default=None. Maximum L2 norm for the embeddings.")
    parser.add_argument('--decoder_dim', default=512, type=int,
        help="int. default=512. Dimension of LSTM hidden states.")
    parser.add_argument('--decoder_layers', default=1, type=int,
        help="int. default=1. Number of LSTM layers.")
    parser.add_argument('--decoder_teacher_forcing', default=False, action='store_true',
        help="store_true. use teacher forcing during training.")
    # General Training Hyperparameters
    parser.add_argument('--batch', default=1, type=int,
        help="int. default=1. batch size.")
    parser.add_argument('--accumulate', default=1, type=int,
        help="int. default=1. number of gradient accumulation steps. simulate larger batches when >1.")
    parser.add_argument('--epochs', default=10, type=int,
        help="int. deafult=10. number of epochs")
    # Optimizer
    parser.add_argument('--opt', default='adam', type=str, choices=['sgd', 'adam', 'adamw'],
        help="str. default=adam. use sgd, adam, or adamw.")
    parser.add_argument('--encoder_lr', default=1e-5, type=float,
        help="float. default=1e-5. encoder learning rate")
    parser.add_argument('--decoder_lr', default=4e-3, type=float,
        help="float. default=4e-3. decoder learning rate")
    parser.add_argument('--momentum', default=0.9, type=float,
        help="float. default=0.9. sgd momentum value.")
    parser.add_argument('--nesterov', default=False, action='store_true',
        help="store_true, sgd with nestrov acceleration.")
    parser.add_argument('--weight_decay', default=0.0, type=float,
        help="float. default=0.0. weight decay for sgd and adamw. 0=no weight decay.")
    parser.add_argument('--adam_b1', default=0.9, type=float)
    parser.add_argument('--adam_b2', default=0.999, type=float)
    parser.add_argument('--grad_clip', default='value', type=str, choices=['value', 'norm'],
        help="str. default=value. pl uses clip_grad_value_ and clip_grad_norm_ from nn.utils.")
    parser.add_argument('--clip_value', default=0, type=float,
        help="float. default=0 is no clipping.")
    # Scheduler
    parser.add_argument('--scheduler', default=None, type=str, choices=['step', 'plateau', 'exp'],
        help="str. default=None. use step, plateau, or exp schedulers.")
    parser.add_argument('--lr_gamma', default=0.1, type=float,
        help="float. default=0.1. gamma for schedulers that scale the learning rate.")
    parser.add_argument('--milestones', nargs='+', default=[10, 15], type=int,
        help="ints. step scheduler milestones.")
    parser.add_argument('--plateau_patience', default=20, type=int,
        help="int. plateau scheduler patience. monitoring the train loss.")
    # Validation
    parser.add_argument('--val_interval', default=5, type=int,
        help="int. default=5. check validation every val_interval epochs. assigned to pl's check_val_every_n_epoch.")
    parser.add_argument('--val_beamk', default=3, type=int,
        help="int. default=3. beam width used during validation step.")
    parser.add_argument('--val_max_len', default=32, type=int,
        help="int. default=32. maximum caption length during validation step.")
    # Callbacks
    parser.add_argument('--save_top_k', default=1, type=int,
        help="int. default=1. save topk model checkpoints.")
    metric_choices = ["chrf", "bleu1", "bleu2", "bleu3", "bleu4", "gleu", "precision", "recall"]
    parser.add_argument('--save_monitor', default='bleu4', type=str, choices=metric_choices,
        help="str. default=bleu4. which metric to find topk models.")
    parser.add_argument('--early_stop_monitor', default=None, type=str, choices=metric_choices,
        help="str. default=None. which metric to use for early stop callback.")
    parser.add_argument('--early_stop_patience', default=6, type=int,
        help="int. default=6. patience epochs for the early stop callback.")
    # Augmentations
    parser.add_argument('--dropout', default=0.1, type=float,
        help="float. default=0.1. Dropout is used before intializing the lstm and before projecting to the vocab.")
    parser.add_argument('--aug_scale', default=0.8, type=float,
        help="float. default=0.8. lower bound for RandomResizedCrop.")
    # SAT Specific
    parser.add_argument('--att_gamma', default=1.0, type=float,
        help="float. default=1.0. Weight multiplied to the doubly stochastic loss")
    args = parser.parse_args()
    return args


def main(args):
    pl.seed_everything(args.seed)

    print(" * Preparing Tensorboard...")

    # Increment to find the next availble name
    logger = TensorBoardLogger(save_dir="logs", name=args.name)    
    dirpath = f"logs/{args.name}/version_{logger.version}"
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    callbacks = [
        ModelCheckpoint(
            monitor=args.save_monitor,
            dirpath=dirpath,
            filename='{epoch:d}-{step}-{bleu4:.4f}',
            save_top_k=args.save_top_k,
            mode='max',
            period=1,  # Check every validation epoch
            save_last=True,  # Always save the latest weights
        )
    ]

    if args.early_stop_monitor is not None:
        callbacks.append(EarlyStopping(monitor=args.early_stop_monitor, patience=args.early_stop_patience, mode='max'))

    print(" * Creating Datasets and Dataloaders...")

    # Setup transforms
    valid_transforms = T.Compose([
        T.Resize(args.input_size),
        T.CenterCrop(args.input_size),
        T.ToTensor(),
    ])
    train_transforms = T.Compose([
        T.RandomResizedCrop(args.input_size, scale=(args.aug_scale, 1.0)),
        T.RandomChoice([
            T.RandomPerspective(distortion_scale=0.2, p=1),
            T.RandomAffine(degrees=10, shear=10),
            T.RandomRotation(degrees=10)
        ]),
        T.ColorJitter(brightness=0.16, contrast=0.15, saturation=0.5, hue=0.04),
        T.RandomHorizontalFlip(),
        # T.RandomVerticalFlip(p=0.1),  # Does this work still for captioning?
        T.RandomGrayscale(),
        T.ToTensor(),
        AddGaussianNoise(std=0.01)
    ])

    train_ds = CocoCaptionDataset(jsonpath=args.json, split="train", transforms=train_transforms)
    valid_ds = CocoCaptionDataset(jsonpath=args.json, split="val", transforms=valid_transforms)

    # Add dataset parameters to the args/hparams
    args.vocab_stoi = train_ds.json["vocab_stoi"]
    args.vocab_itos = {v:k for k,v in train_ds.json["vocab_stoi"].items()}
    args.vocab_size = train_ds.json["vocab_size"]

    train_loader = DataLoader(dataset=train_ds,
            batch_size=args.batch, num_workers=args.workers,
            persistent_workers=(True if args.workers > 0 else False),
            pin_memory=True)
    val_loader = DataLoader(dataset=valid_ds,
            batch_size=args.batch, num_workers=args.workers,
            persistent_workers=(True if args.workers > 0 else False),
            pin_memory=True)

    # imgs, caps, lens = next(iter(train_loader))
    # print("imgs:", type(imgs), imgs.device, imgs.dtype, imgs.shape)
    # print("caps:", type(caps), caps.device, caps.dtype, caps.shape)
    # print("lens:", type(lens), lens.device, lens.dtype, lens.shape)
    print(" * Effective Batch Size = {}".format(args.batch*args.accumulate))

    model = SAT(**vars(args))

    trainer = pl.Trainer(
        accumulate_grad_batches=args.accumulate,
        benchmark=args.benchmark,  # cudnn.benchmark
        callbacks=callbacks,
        check_val_every_n_epoch=args.val_interval,
        deterministic=True,  # cudnn.deterministic
        gpus=args.ngpu,
        gradient_clip_algorithm=args.grad_clip,
        gradient_clip_val=args.clip_value,
        logger=logger,
        precision=args.precision,
        progress_bar_refresh_rate=1,
        max_epochs=args.epochs,
    )

    trainer.fit(
        model=model,
        train_dataloader=train_loader,
        # val_dataloaders=val_loader
    )


if __name__ == "__main__":
    args = get_args()
    main(args)