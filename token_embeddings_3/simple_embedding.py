import torch
import tiktoken

sentence = "learning NLP and Large Language Models is very interesting"
encoder = tiktoken.get_encoding("gpt2")
# print(tiktoken.list_encoding_names())
ids = encoder.encode(text=sentence, allowed_special="all")

# tokens = sorted(sentence.split(" ")) # just for simplicity
# vocab = [{token: idx} for idx, token in enumerate(tokens)]

# vocab_size = len(vocab)
dimension_size = 4

torch.manual_seed(42)
embedding_layer = torch.nn.Embedding(num_embeddings=len(ids), embedding_dim=dimension_size)

print(embedding_layer.weight)

print(embedding_layer(torch.tensor([3])))

print(embedding_layer(torch.tensor([3, 6, 4, 1])))
