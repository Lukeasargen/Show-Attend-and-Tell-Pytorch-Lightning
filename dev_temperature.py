import torch
from torch.nn import functional as F


logits = torch.randn(2)
print(f"{logits = }")

temperature = 0.5
temperature = [1.5, 1.0, 0.5, 0.1]

if not isinstance(temperature, list):
    temperature = [temperature]

for i in range(5):
    current_temperature = temperature[i % len(temperature)]
    p = F.softmax(logits/current_temperature, dim=0)
    print(f"{i=} {current_temperature=}. {p = }")
