
import matplotlib.pyplot as plt
import torch
from torch.optim.lr_scheduler import OneCycleLR


epochs = 140 #torch.randint(50, 300, (1,))
train_loader_len = 128 #torch.randint(100, 800, (1,))

steps = int(epochs*train_loader_len)

lr = [1e-3, 1e-1]

one_cycle_pct = 0.3
one_cycle_div = 25
one_cycle_fdiv = 1e4


model1 = torch.nn.Linear(2, 1)
model2 = torch.nn.Linear(2, 1)
params = [
    {'params': model1.parameters(), 'lr': lr[0]},
    {'params': model2.parameters(), 'lr': lr[1]}
]
optimizer = torch.optim.AdamW(params, lr=1e-2)
scheduler = OneCycleLR(
    optimizer,
    lr,
    epochs=epochs,
    steps_per_epoch=train_loader_len,
    pct_start=one_cycle_pct,
    cycle_momentum=True,
    base_momentum=0.85,
    max_momentum=0.95,
    div_factor=one_cycle_div,
    final_div_factor=one_cycle_fdiv,
)

lrs = []
momentums = []

for i in range(steps):
    optimizer.step()
    scheduler.step()
    lrs.append(optimizer.param_groups[0]["lr"])
    momentums.append(optimizer.param_groups[0]["betas"][0])

print(f"Last lr={lrs[-1]:e}")

print(optimizer.param_groups)

fig, axs = plt.subplots(2)
axs[0].plot(range(steps), lrs, label="lr")
axs[1].plot(range(steps), momentums, label="momentum")
plt.legend()
plt.show()
