from torch import LongTensor, norm
from torch.nn import Embedding

sentences = LongTensor([
        [1,2,4,5],
        [4,3,2,9]
    ])

embedding = Embedding(num_embeddings=10, embedding_dim=32, max_norm=1)

for sentence in embedding(sentences):
    for word in sentence:
        print(norm(word))
