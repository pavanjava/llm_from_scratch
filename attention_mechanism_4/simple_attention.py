import torch
import numpy as np
import matplotlib.pyplot as plt


inputs = torch.tensor(
    [
        np.random.rand(1,3)[0], # your
        np.random.rand(1,3)[0], # journey
        np.random.rand(1,3)[0], # of
        np.random.rand(1,3)[0], # NLP
        np.random.rand(1,3)[0], # looks
        np.random.rand(1,3)[0] # Awesome
    ]
)

query = inputs[1]

print(inputs)
print(query)

attn_scores = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
   attn_scores[i] = torch.dot(x_i, query)

print(attn_scores)
