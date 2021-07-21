import time
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.core.memory import ModelSummary

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from model import get_encoder


class LitModel(pl.LightningModule):
    def __init__(self, net, x):
        super().__init__()
        self.net = net
        self.example_input_array = x

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.pretrained = False
    args.encoder_finetune = False
    args.mean = 0.5
    args.std = 0.5
    
    archs = [
        # "resnet18",
        # "resnet34",
        # "resnet50",
        # "resnet101",
        # "resnet152",
        # "resnext50_32x4d",
        # "resnext101_32x8d",
        # "wide_resnet50_2",
        # "wide_resnet101_2",
        # "squeezenet1_0",
        "squeezenet1_1",
        # "densenet121",
        # "densenet169",
        # "densenet201",
        # "densenet161",
        "shufflenet_v2_x0_5",
        "shufflenet_v2_x1_0",
        # "shufflenet_v2_x1_5",
        # "shufflenet_v2_x2_0",
        # "mobilenet_v2",
        # "mobilenet_v3_large",
        "mobilenet_v3_small",
        "mnasnet0_5",
        # "mnasnet0_75",
        # "mnasnet1_0",
        # "mnasnet1_3",
    ]

    batch = 128
    in_channels = 3
    input_size = 224
    warmup = 5
    n_trials = 100

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data = torch.rand((batch, in_channels, input_size, input_size)).to(device)
    amp_grad = [
        # (False, True),
        # (False, False),
        (True, True),
        # (True, False)
    ]

    for arch in archs:
        torch.cuda.empty_cache()

        args.encoder_dim = None
        args.encoder_size = None
        args.encoder_arch = arch

        model = get_encoder(args).to(device)
        m = LitModel(model, data).to(device)
        ret = ModelSummary(m)
        params = ret.total_parameters * 1e-6

        for amp, grad in amp_grad:
            with torch.cuda.amp.autocast(amp):
                with torch.set_grad_enabled(grad):
                    for i in range(warmup): yhat = model(data)
                    t0 = time.time()
                    for i in range(n_trials): yhat = model(data)
                    duration = time.time() - t0

            latency = 1e3*duration/n_trials
            batches_per_sec = n_trials/duration
            images_per_sec = data.size(0)*batches_per_sec
            features = yhat.shape[2]

            print(f"{arch=:18} {features=:4d} params={params:6.2f}M. amp={str(amp):5} grad={str(grad):5} Latency={latency:7.3f} ms. {batch=:4d}. Batches/s={batches_per_sec:5.1f}. Imgs/s={images_per_sec:7.1f}.")

        del model
