
import matplotlib.pyplot as plt
import torch


steps = int(30e3)
lr = 1e-1
t0 = 1e3
tm = 2
lr_min = 1e-8

model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer,
    T_0=int(t0),
    T_mult=tm,
    eta_min=lr_min,
)

lrs = []

for i in range(steps):
    optimizer.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    scheduler.step()

plt.plot(range(steps), lrs)
plt.show()
