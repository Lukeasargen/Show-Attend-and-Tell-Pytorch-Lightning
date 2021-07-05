import time
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.core.memory import ModelSummary

from model import get_encoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
args = parser.parse_args()


pretrained = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
    'squeezenet1_0', 'squeezenet1_1',
    'densenet121', 'densenet169', 'densenet201', 'densenet161',
    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0',
    'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
    'mnasnet0_5', 'mnasnet1_0',
]
# missing = mnasnet0_75, mnasnet1_3, shufflenet_v2_x1_5, shufflenet_v2_x2_0

ptmodels = [
    'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
    'resnext50_32x4d','resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2',
    'squeezenet1_0', 'squeezenet1_1',
    'densenet121', 'densenet169', 'densenet201', 'densenet161',
    'mobilenet_v2', 'mobilenet_v3_large', 'mobilenet_v3_small',
    'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0', 'mnasnet1_3',
    'shufflenet_v2_x0_5', 'shufflenet_v2_x1_0', 'shufflenet_v2_x1_5', 'shufflenet_v2_x2_0'
]


class LitModel(pl.LightningModule):
    def __init__(self, net, x):
        super().__init__()
        self.net = net
        self.example_input_array = x

    def forward(self, x):
        return self.net(x)

steps = 200
x = torch.zeros(64, 3, 224, 224).to(device)

args.pretrained = False
args.encoder_finetune = False
args.mean = 0.5
args.std = 0.5

for arch in ptmodels:
    args.encoder_dim = None
    args.encoder_size = 14
    args.encoder_arch = arch

    torch.cuda.empty_cache()
    model = get_encoder(args).to(device)

    
    # with torch.set_grad_enabled(False):
    #     y = model(x)
    # print(f"{arch: <19} {y.shape} {args.encoder_dim = } {args.encoder_size = }")


    with torch.set_grad_enabled(False):
        for i in range(20): # warmup
            y = model(x)
        t0 = time.time()
        for i in range(steps):
            y = model(x)
    m = LitModel(model, x).to(device)
    ret = ModelSummary(m)
    gpu_mem = torch.cuda.memory_allocated(0) * 1e-9
    print(f"{arch: <18} features={y.shape[2]}. latency={(time.time()-t0)*1e3/steps:.2f} ms. params={ret.total_parameters}. {gpu_mem=:.3f} GB.")

    # print(f"{gpu_mem = :.3f} GB")

    del model



