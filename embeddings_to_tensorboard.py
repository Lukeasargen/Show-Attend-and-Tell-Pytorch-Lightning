"""
tensorboard --logdir runs/ --port=6007
http://localhost:6007/#projector
"""


import torch
from torch.utils.tensorboard import SummaryWriter

from model import SAT


def main():
    checkpoint_path = "logs/default/version_54/last.ckpt"

    model = SAT.load_from_checkpoint(checkpoint_path)
    word_embedding = model.embedding.weight
    words = list(model.hparams.vocab_stoi.keys())

    writer = SummaryWriter()
    writer.add_embedding(word_embedding, metadata=words, tag="word embedding")
    writer.close()

if __name__ == "__main__":
    main()
