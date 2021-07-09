
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T

from model import SAT
from util import CocoCaptionDataset


def main():
    checkpoint_path = "logs/default/version_52/last.ckpt"

    workers = 0
    batch = 16

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = SAT.load_from_checkpoint(checkpoint_path, map_location=device).to(device)
    model.freeze()

    valid_transforms = T.Compose([
        T.Resize(model.hparams.input_size),
        T.CenterCrop(model.hparams.input_size),
        T.ToTensor()
    ])
    valid_ds = CocoCaptionDataset(jsonpath=model.hparams.json, split="val", transforms=valid_transforms)
    val_loader = DataLoader(dataset=valid_ds, batch_size=batch, num_workers=workers, shuffle=False,
                        persistent_workers=(True if workers > 0 else False), pin_memory=True)

    # Get all the logits and targets
    logits, targets = [], []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            img, encoded_captions, lengths = batch
            # Reshape the validation batch to unroll the multiple captions into the batch dimension
            batch_len, repeats, pad_seq_len = encoded_captions.shape
            img = torch.repeat_interleave(img, repeats=repeats, dim=0).to(device)
            lengths = lengths.view(batch_len*repeats, 1).to(device)
            encoded_captions = encoded_captions.view(batch_len*repeats, 1, pad_seq_len).to(device)
            # Forward pass
            logits_packed, targets_packed, _ = model.train_batch([img, encoded_captions, lengths], epsilon=1)
            # Keep all predictions and targets in a list
            logits.append(logits_packed.data)
            targets.append(targets_packed.data)

            print(i)
            if i>40: break

    # Combine all logits and targets into 1 tensor
    logits = torch.cat(logits)
    targets = torch.cat(targets)

    # to() creates a copy, so I detach it to make this a leaf tensor
    temperature = (torch.ones(1)*1.5).to(device).detach().requires_grad_(True)
    
    optimizer = torch.optim.SGD([temperature], lr=1e-2, momentum=0.8, nesterov=True)
    for i in range(70):
        loss = F.cross_entropy(logits/temperature, targets)
        print(f"{temperature = }")
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    print(f"{temperature = }")


if __name__ == "__main__":
    main()

