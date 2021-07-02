import json

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class CocoCaptionDataset(Dataset):
    def __init__(self, jsonpath, split="train", transforms=None):
        self.transforms = T.Compose([T.ToTensor()])
        if transforms:
            self.transforms = transforms
        self.json = json_loader(jsonpath)
        self.split = split
        self.vocab_stoi = self.json["vocab_stoi"]
        self.vocab_itos = {v:k for k,v in self.vocab_stoi.items()}

        self.img_paths = self.json[split]["img_paths"]
        self.encoded_captions = self.json[split]["encoded_captions"]
        self.lengths = self.json[split]["lengths"]
        assert len(self.img_paths)==len(self.encoded_captions)==len(self.lengths)

    def stoi(self, s):
        return int(self.vocab_stoi.get(s, self.vocab_stoi['<UNK>']))

    def itos(self, i):
        return str(self.vocab_itos.get(int(i), "<UNK>"))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = self.transforms(pil_loader(self.img_paths[idx]))
        # Expects self.encoded_captions and self.lengths are lists of lists
        encoded_captions = torch.LongTensor(self.encoded_captions[idx])
        lengths = torch.LongTensor(self.lengths[idx])
        return img, encoded_captions, lengths


def time_to_string(t):
    if t > 3600: return "{:.2f} hours".format(t/3600)
    if t > 60: return "{:.2f} minutes".format(t/60)
    else: return "{:.2f} seconds".format(t)


class AddGaussianNoise(torch.nn.Module):
    """ Apply Guassian noise to a float tensor """
    def __init__(self, std=0.01):
        self.std = std
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std
    
    def __repr__(self):
        return self.__class__.__name__ + '(std={1})'.format(self.std)


def json_loader(path):
    return json.load(open(path))


def pil_loader(path):
    return Image.open(open(path, 'rb')).convert('RGB')


def load_square(path, size=None):
    img = pil_loader(path)
    return crop_max_square(img, size)


def prepare_image(img, size=None):
    if size:
        img = crop_max_square(img, int(size))
    return T.ToTensor()(img).unsqueeze_(0)


def crop_center(pil_img, crop_width, crop_height):
    img_width, img_height = pil_img.size
    return pil_img.crop(((img_width - crop_width) // 2,
                         (img_height - crop_height) // 2,
                         (img_width + crop_width) // 2,
                         (img_height + crop_height) // 2))


def crop_max_square(pil_img, size):
    pil_img = crop_center(pil_img, min(pil_img.size), min(pil_img.size))
    if size:
        pil_img = pil_img.resize((size, size))
    return pil_img





