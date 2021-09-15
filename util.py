from collections import OrderedDict
import json

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as T
from torch.utils.data.sampler import Sampler


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


class BucketSampler(Sampler):
    """ I used several of the examples from this link below
        https://discuss.pytorch.org/t/tensorflow-esque-bucket-by-sequence-length/41284/10
    """
    def __init__(self, lengths, batch_size, indices=None):
        self.lengths = lengths
        self.batch_size = batch_size
        if indices:
            self.indices = indices
        else:
            self.indices = list(range(len(self.lengths)))
        self.idx_len_zip = list(zip(self.indices, self.lengths))
        # Use an OrderedDict with the lengths as keys
        len_map = OrderedDict()
        for i, length_list in self.idx_len_zip:
            # Use the sum of the length list
            # This is essentially the number of targets of the sample
            # Placing the largest number of targets in the first batch
            # will hopefully avoid OOM by avoiding memory growth.
            l = sum(length_list)
            if l not in len_map:
                len_map[l] = [i]  # Create a new lsit for this length
            else:
                len_map[l].append(i)  # Add to an existing list
        # Sort and reverse so the longest sequences are first
        self.grouped_indices = []
        for l, idxs in reversed(sorted(len_map.items())):
            self.grouped_indices.append(idxs)

    def __iter__(self):
        # Appened the shuffled groups to a new list
        suffled_grouped_indices = []
        for indices in self.grouped_indices:
            # Calling this shuffle on the groups is faster than on the whole zipped list
            np.random.shuffle(indices)
            suffled_grouped_indices.extend(indices)
        return iter(suffled_grouped_indices)

    def __len__(self):
        return len(self.lengths)


""" https://github.com/NVIDIA/DeepLearningExamples/blob/8d8b21a933fff3defb692e0527fca15532da5dc6/PyTorch/Classification/ConvNets/image_classification/smoothing.py """
class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


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





