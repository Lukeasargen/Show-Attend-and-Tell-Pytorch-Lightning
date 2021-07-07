
import torch
from torch import nn
from torch.nn import functional as F


encoder_size = 2

# x = torch.tensor([[[[ 10.0, 20.0],[ 30.0, 40.0]]]])

in_size = 5
x = torch.randn(1, 1, in_size, in_size)

print(f"{x = }")
print(f"{x.shape = }")


pool = nn.AdaptiveAvgPool2d((encoder_size, encoder_size))
y1 = pool(x)
print(f"{y1 = }")
print(f"{y1.shape = }")

upsample = nn.Upsample((encoder_size, encoder_size), mode="bilinear", align_corners=False)
y2 = upsample(x)
print(f"{y2 = }")
print(f"{y2.shape = }")

print(f"{y2-y1 = }")


