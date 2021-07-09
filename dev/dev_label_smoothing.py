
import torch
from torch.nn import functional as F

import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from util import LabelSmoothing


classes = 10
samples = 20

logits = torch.randn(samples, classes)
targets = torch.randint(classes, size=(samples,))

loss1 = F.cross_entropy(logits, targets)
print(f"{loss1 = :.4f}")

criterion = LabelSmoothing(0)
loss2 = criterion(logits, targets)
print(f"{loss2 = :.4f}")

criterion = LabelSmoothing(0.3)
loss3 = criterion(logits, targets)
print(f"{loss3 = :.4f}")

