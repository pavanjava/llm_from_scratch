import torch

sentence = "learning NLP and Large Language Models is very interesting"

tokens = sorted(sentence.split(" "))
vocab = [{token: idx} for idx, token in enumerate(tokens)]

vocab_size = len(vocab)
dimension_size = 4

torch.manual_seed(42)
embedding_layer = torch.nn.Embedding(num_embeddings=vocab_size, embedding_dim=dimension_size)

print(embedding_layer.weight)

print(embedding_layer(torch.tensor([3])))

print(embedding_layer(torch.tensor([3, 6, 4, 1])))
