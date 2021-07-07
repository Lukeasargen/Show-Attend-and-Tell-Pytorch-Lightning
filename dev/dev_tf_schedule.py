import torch

epochs = 20
decoder_tf = 'inv_sigmoid'

vals = []

for current_epoch in range(epochs):

    if current_epoch==0:
        epsilon = 1
    else:
        if decoder_tf=="always":
            epsilon = 1
        elif decoder_tf=="linear":
            epsilon = 1 - current_epoch/epochs
        elif decoder_tf=="inv_sigmoid":
            # Shift the 50% point with b, change the slope with g
            b, g = 0.5*epochs, 5.0
            epsilon = 1/(1+torch.exp(torch.tensor((g/b)*(current_epoch-b))))
        elif decoder_tf=="exp":
            epsilon = torch.exp(torch.log(torch.tensor(0.01))/epochs)**current_epoch
    vals.append(epsilon)

print(vals)
