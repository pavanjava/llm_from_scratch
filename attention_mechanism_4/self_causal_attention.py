import asyncio

import torch
from torch import nn


class SelfCausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, kqv_bias: bool = False):
        super().__init__()
        self.d_out = d_out
        self.w_k = nn.Linear(d_in, d_out, bias=kqv_bias)
        self.w_q = nn.Linear(d_in, d_out, bias=kqv_bias)
        self.w_v = nn.Linear(d_in, d_out, bias=kqv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries: torch.Tensor = self.w_q(x)
        keys: torch.Tensor = self.w_k(x)
        values: torch.Tensor = self.w_v(x)

        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores.masked_fill(
            self.mask.bool()[:num_tokens, :num_tokens],
            -torch.inf
        )
        attention_weights: torch.Tensor = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        attention_weights: torch.Tensor = self.dropout(attention_weights)

        context_vector: torch.Tensor = attention_weights @ values
        return context_vector


async def main():
    torch.manual_seed(42)
    input_1 = torch.tensor([
        [0.8, 0.6, 0.1],  # I
        [0.7, 0.7, 0.1],  # Love
        [0.2, 0.8, 0.6],  # Coding
        [0.1, 0.7, 0.7],  # GenAI
    ])
    input_2 = torch.tensor([
        [0.5, 0.2, 0.7],
        [0.8, 0.9, 0.3],
        [0.1, 0.6, 0.4],
        [0.7, 0.3, 0.8]
    ])
    input_3 = torch.tensor([
        [0.4, 0.8, 0.1],
        [0.9, 0.2, 0.6],
        [0.3, 0.5, 0.9],
        [0.6, 0.1, 0.4]
    ])
    batch = torch.stack((input_1, input_2, input_3), dim=0)
    print(batch.shape)
    contex_length = batch.shape[1]
    ca = SelfCausalAttention(3,3, contex_length, 0.3)
    # attention_scores = await compute_attention_scores_for_single_query(inputs=inputs)
    context_vectors = ca(batch)
    print(context_vectors)

if __name__ == "__main__":
    asyncio.run(main())
