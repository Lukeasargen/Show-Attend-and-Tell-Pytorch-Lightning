import argparse
import os

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from model import SAT
from util import CocoCaptionDataset, BucketSampler, AddGaussianNoise, RestartCheckpoint


def get_args():
    parser = argparse.ArgumentParser()
    # Early Stop, Mdodel Checkpointing, Plateau Scheduler can use these metrics
    metric_choices = ["bleu1", "bleu2", "bleu3", "bleu4", "gleu"]
    # Init and setup
    parser.add_argument('--seed', default=42, type=int,
        help="int. default=42. deterministic seed. cudnn.deterministic is always set True by deafult.")
    parser.add_argument('--name', default='default', type=str,
        help="str. default=default. Tensorboard name and log folder name.")
    parser.add_argument('--workers', default=0, type=int,
        help="int. default=0. Dataloader num_workers. good practice is to use number of cpu cores.")
    parser.add_argument('--gpus', nargs="+", default=None, type=int,
        help="str. default=None (cpu). gpus to train on. see pl multi_gpu docs for details.")
    parser.add_argument('--benchmark', default=False, action='store_true',
        help="store_true. set cudnn.benchmark.")
    parser.add_argument('--precision', default=32, type=int, choices=[16, 32],
        help="int. default=32. 32 for full precision and 16 uses pytorch amp")
    # Dataset
    parser.add_argument('--json', type=str, required=True,
        help="str. REQUIRED. Path to json made in preproces.ipynb.")
    parser.add_argument('--mean', nargs=3, default=[0.485, 0.456, 0.406], type=float,
        help="3 floats. default is imagenet [0.485, 0.456, 0.406].")
    parser.add_argument('--std', nargs=3, default=[0.229, 0.224, 0.225], type=float,
        help="3 floats, default is imagenet [0.229, 0.224, 0.225].")
    parser.add_argument('--bucket_sampler', default=False, action='store_true',
        help="store_true. replicate the function of tensorflow bucket_by_sequence_length.")
    # Vision Encoder Parameters
    parser.add_argument('--encoder_arch', default='shufflenet_v2_x0_5', type=str,
        help="str. default=shufflenet_v2_x0_5. torchvision model name.")
    parser.add_argument('--input_size', default=224, type=int,
        help="int. default=224. input size in pixels.")
    parser.add_argument('--pretrained', default=False, action='store_true')
    parser.add_argument('--encoder_finetune_after', default=-1, type=int,
        help="int. default=-1 is no finetuning. Start finetuning after this number of steps.")
    parser.add_argument('--encoder_dim', default=None, type=int,
        help="int. default=None. Adds a 1x1 conv to the encoder if out_channels!=encoder_dim. D in the paper.")
    # Text Decoder Parameters
    parser.add_argument('--embed_dim', default=256, type=int,
        help="int. default=256. Dimension of vocab embeddings.")
    parser.add_argument('--embed_norm', default=None, type=float,
        help="float. default=None. Maximum L2 norm for the embeddings.")
    parser.add_argument('--attention_dim', default=128, type=int,
        help="int. default=512. Dimension of soft attention projection.")
    parser.add_argument('--decoder_dim', default=512, type=int,
        help="int. default=512. Dimension of LSTM hidden states.")
    parser.add_argument('--decoder_layers', default=1, type=int,
        help="int. default=1. Number of LSTM layers.")
    parser.add_argument('--decoder_tf', default=None, type=str, choices=['always', 'linear', 'inv_sigmoid', 'exp'],
        help="str. default=None. use always, linear, inv_sigmoid, exp.")
    parser.add_argument('--decoder_tf_min', default=0.5, type=float,
        help="float. default=0.5. Minimum percent of teacher forcing epsilon.")
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
    parser.add_argument('--decoder_lr', default=1e-3, type=float,
        help="float. default=1e-3. decoder learning rate")
    parser.add_argument('--embedding_lr', default=1e-2, type=float,
        help="float. default=1e-2. embedding learning rate")
    parser.add_argument('--lr_warmup_steps', default=0, type=int,
        help="int. deafult=0. linearly increase learning rate for this number of steps.")
    parser.add_argument('--momentum', default=0.9, type=float,
        help="float. default=0.9. sgd momentum value.")
    parser.add_argument('--nesterov', default=False, action='store_true',
        help="store_true. sgd with nestrov acceleration.")
    parser.add_argument('--weight_decay', default=0.0, type=float,
        help="float. default=0.0. weight decay for sgd and adamw. 0=no weight decay.")
    parser.add_argument('--adam_b1', default=0.9, type=float)
    parser.add_argument('--adam_b2', default=0.999, type=float)
    parser.add_argument('--grad_clip', default='value', type=str, choices=['value', 'norm'],
        help="str. default=value. pl uses clip_grad_value_ and clip_grad_norm_ from nn.utils.")
    parser.add_argument('--clip_value', default=0, type=float,
        help="float. default=0 is no clipping.")
    parser.add_argument('--min_lr', default=0.0, type=float,
        help="float. default=0.0. minimum learning rate.")
    # Scheduler
    parser.add_argument('--scheduler', default=None, type=str,
        choices=['step', 'plateau', 'exp', 'cosine', 'one_cycle'],
        help="str. default=None. use step, plateau, exp, or cosine schedulers.")
    parser.add_argument('--lr_gamma', default=0.1, type=float,
        help="float. default=0.1. gamma for schedulers that scale the learning rate.")
    parser.add_argument('--milestones', nargs='+', default=[10, 15], type=int,
        help="ints. step scheduler milestones.")
    parser.add_argument('--plateau_patience', default=20, type=int,
        help="int. plateau scheduler patience. monitoring the train loss.")
    parser.add_argument('--plateau_monitor', default='bleu4', type=str, choices=metric_choices,
        help="str. default=bleu4. which metric to drop the lr.")
    parser.add_argument('--cosine_iterations', default=1e3, type=float,
        help="float. default=1e3. number of iterations for the first restart.")
    parser.add_argument('--cosine_multi', default=1, type=int,
        help="int. default=1. multiply factor increases iterations after a restart.")
    parser.add_argument('--one_cycle_pct', default=0.3, type=float,
        help="float. default=0.3. percentage of steps increasing the lr.")
    parser.add_argument('--one_cycle_div', default=25, type=float,
        help="float. default=25. determines the initial lr.")
    parser.add_argument('--one_cycle_fdiv', default=1e4, type=float,
        help="float. default=1e4. determines the minimum lr.")
    # Validation
    parser.add_argument('--val_interval', default=5, type=int,
        help="int. default=5. check validation every val_interval epochs. assigned to pl's check_val_every_n_epoch.")
    parser.add_argument('--val_percent', default=1.0, type=float,
        help="float. default=1.0. percentage of validation set to test during a validation step.")
    parser.add_argument('--val_beamk', default=3, type=int,
        help="int. default=3. beam width used during validation step.")
    parser.add_argument('--val_max_len', default=32, type=int,
        help="int. default=32. maximum caption length during validation step.")
    # Callbacks
    parser.add_argument('--save_top_k', default=1, type=int,
        help="int. default=1. save topk model checkpoints.")
    parser.add_argument('--save_monitor', default='bleu4', type=str, choices=metric_choices,
        help="str. default=bleu4. which metric to find topk models.")
    parser.add_argument('--early_stop_monitor', default=None, type=str, choices=metric_choices,
        help="str. default=None. which metric to use for early stop callback.")
    parser.add_argument('--early_stop_patience', default=6, type=int,
        help="int. default=6. patience epochs for the early stop callback.")
    # Misc
    parser.add_argument('--dropout', default=0.0, type=float,
        help="float. default=0.0. Dropout is used before intializing the lstm and before projecting to the vocab.")
    parser.add_argument('--embedding_dropout', default=0.0, type=float,
        help="float. default=0.0. Dropout is used on the word embeddings.")
    parser.add_argument('--label_smoothing', default=0.0, type=float,
        help="float. default=0. label smoothing epsilon value.")
    parser.add_argument('--weight_tying', default=False, action='store_true',
        help="store_true. set to use weight tying (Inan et al., 2016.")
    # Augmentations
    parser.add_argument('--aug_scale', default=0.9, type=float,
        help="float. default=0.9. lower bound for RandomResizedCrop. 1.0 uses CenterCrop")
    parser.add_argument('--aug_hflip', default=0.5, type=float,
        help="float. default=0.5. probability for RandomHorizontalFlip.")
    parser.add_argument('--aug_color_jitter', default=0.0, type=float,
        help="float. default=0.0. ColorJitter brightness, contrast, and saturation value.")
    parser.add_argument('--aug_optical_strength', default=0.0, type=float,
        help="float. default=0.0. linearly scale the strength of rotation, shearing, and distortion up to 45 degrees.")
    parser.add_argument('--aug_noise_std', default=0.01, type=float,
        help="float. default=0.01. add guassian noise to the inputs. <=0.02 is best.")
    # SAT Specific
    parser.add_argument('--deep_output', default=False, action='store_true',
        help="store_true. set to use deep output (equation 7), the deafult is to use the last hidden layer.")
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
            every_n_epochs=1,
            save_last=True,  # Always save the latest weights
        ),
        RestartCheckpoint(
                dirpath=dirpath,
                every_n_train_steps=1,
        )
    ]

    if args.early_stop_monitor is not None:
        callbacks.append(
            EarlyStopping(
                monitor=args.early_stop_monitor,
                patience=args.early_stop_patience,
                mode='max',
                check_on_train_epoch_end=False
            )
        )

    print(" * Creating Datasets and Dataloaders...")

    # Setup transforms
    valid_transforms = T.Compose([
        T.Resize(args.input_size),
        T.CenterCrop(args.input_size),
        T.ToTensor(),
    ])

    train_transforms = []
    if args.aug_scale==1.0:
        train_transforms += [T.Resize(args.input_size), T.CenterCrop(args.input_size)]
    elif args.aug_scale>=0 and  args.aug_scale<1.0:
        train_transforms += [T.RandomResizedCrop(args.input_size, scale=(args.aug_scale, 1.0))]
    else:
        raise ValueError("Invalid value for aug_scale. Choose in the range {0,1}.")
    if args.aug_hflip>0 and  args.aug_hflip<1.0:
        train_transforms += [T.RandomHorizontalFlip(p=args.aug_hflip)]
    if args.aug_color_jitter!=0 and args.aug_color_jitter<=1.0:
        train_transforms += [T.ColorJitter(brightness=args.aug_color_jitter, contrast=args.aug_color_jitter, saturation=args.aug_color_jitter, hue=0.03)]
    if args.aug_optical_strength!=0.0 and args.aug_optical_strength<=1.0:
        train_transforms += [
            T.RandomChoice([
                T.RandomPerspective(distortion_scale=0.5*args.aug_optical_strength, p=1),
                T.RandomAffine(degrees=45*args.aug_optical_strength, shear=45*args.aug_optical_strength),
                T.RandomRotation(degrees=45*args.aug_optical_strength)
            ])]
    train_transforms += [T.ToTensor(), AddGaussianNoise(std=args.aug_noise_std)]
    train_transforms = T.Compose(train_transforms)

    train_ds = CocoCaptionDataset(jsonpath=args.json, split="train", transforms=train_transforms)

    # Add dataset parameters to the args/hparams
    args.vocab_stoi = train_ds.json["vocab_stoi"]
    args.vocab_itos = {v:k for k,v in train_ds.json["vocab_stoi"].items()}
    args.vocab_size = train_ds.json["vocab_size"]
    args.embed_dim = train_ds.json["embed_dim"] if (train_ds.json["embed_dim"] is not None) else args.embed_dim
    args.pretrained_embedding = train_ds.json["pretrained_embedding"]

    train_loader = DataLoader(dataset=train_ds,
            sampler=(BucketSampler(train_ds.lengths, args.batch) if args.bucket_sampler else None),
            shuffle=(not args.bucket_sampler),
            batch_size=args.batch, num_workers=args.workers,
            persistent_workers=(True if args.workers > 0 else False),
            pin_memory=True)
    args.train_loader_len = len(train_loader)


    valid_ds = CocoCaptionDataset(jsonpath=args.json, split="val", transforms=valid_transforms)
    val_loader = DataLoader(dataset=valid_ds,
            sampler=(BucketSampler(valid_ds.lengths, args.batch) if args.bucket_sampler else None),
            shuffle=(not args.bucket_sampler),
            batch_size=int(max(1, args.batch//1)), num_workers=args.workers,
            persistent_workers=(True if args.workers > 0 else False),
            pin_memory=True)


    print(f" * Effective Batch Size = {args.batch*args.accumulate}")

    model = SAT(**vars(args))

    trainer = pl.Trainer(
        accumulate_grad_batches=args.accumulate,
        benchmark=args.benchmark,  # cudnn.benchmark
        callbacks=callbacks,
        check_val_every_n_epoch=args.val_interval,
        deterministic=True,  # cudnn.deterministic
        gpus=args.gpus,
        gradient_clip_algorithm=args.grad_clip,
        gradient_clip_val=args.clip_value,
        limit_val_batches=args.val_percent,
        logger=logger,
        precision=args.precision,
        progress_bar_refresh_rate=1,
        max_epochs=args.epochs,
        num_sanity_val_steps=0,
    )

    trainer.fit(
        model=model,
        train_dataloader=train_loader,
        val_dataloaders=val_loader
    )


if __name__ == "__main__":
    args = get_args()
    main(args)