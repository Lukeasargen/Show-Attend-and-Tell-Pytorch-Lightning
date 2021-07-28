
import matplotlib.pyplot as plt
import torch


epochs = 140 #torch.randint(50, 300, (1,))
len_dataloader = 128 #torch.randint(100, 800, (1,))
accumulate = 4 # torch.randint(1, 8, (1,))
warmup = 200 #torch.randint(0, 2000, (1,))

steps = int(epochs*len_dataloader)

lr = 1e-3
t0 = 5e4
tm = 2
lr_min = 1e-9

print(f"{epochs=} ")
print(f"{len_dataloader=} ")
print(f"{accumulate=} ")
print(f"{warmup=} ")
print(f"{steps=} ")


""" Adjust the t0 to end with a low learning rate
Get number of restarts for specified t0 and tm, then update t0
Under estimate the number of restarts and over estimate the t0 size
Add the accumulate steps just in case of unfortunate rounding
"""
adj_steps = steps-warmup
if tm!=1:
    # Use the sum of geometric sequence solved for n to get the number of restarts
    # geometric sum = t0*(1-tm**n)/(1-tm)
    restarts = (torch.log(torch.tensor(1-(adj_steps*(1-tm)/t0)))/torch.log(torch.tensor(tm))).floor()
    # Divide by the geometric sum to get t0
    if restarts==0.0:
        t0 = adj_steps
    else:
        t0 = ((adj_steps+accumulate)/((1-tm**restarts)/(1-tm))).ceil()
else:
    restarts = torch.tensor(adj_steps/t0).floor()
    if restarts==0.0:
        t0 = adj_steps
    else:
        t0 = ((adj_steps+accumulate)/restarts).ceil()

print(f"{restarts=}")
print(f"{t0=}")

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
    if i > warmup:
        lrs.append(optimizer.param_groups[0]["lr"])
        scheduler.step()
    else:
        lrs.append(lr*i/warmup)

print(f"Last lr={lrs[-1]:e}")
plt.plot(range(steps), lrs)
plt.show()
