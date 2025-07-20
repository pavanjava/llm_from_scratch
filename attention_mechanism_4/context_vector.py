import asyncio

import torch

from attention_mechanism_4.simple_attention import (
    compute_attention_scores_for_single_query,
    compute_attention_weights,
    compute_attention_scores_for_whole_inputs
)

# compute the context for one query word
async def compute_context_vector_for_query(inputs: torch.tensor, attention_weights: torch.tensor, context_vector: torch.tensor):
    for i, _input in enumerate(inputs):
        context_vector += attention_weights[i] * _input
    return context_vector

# compute the context for all words
async def compute_context_vector(inputs: torch.tensor, attention_weights: torch.tensor):
    context_vector = attention_weights @ inputs
    return context_vector


async def main():
    inputs = torch.tensor(
        [
            [0.8, 0.6, 0.1],  # I
            [0.7, 0.7, 0.1],  # Love
            [0.2, 0.8, 0.6],  # Coding
            [0.1, 0.7, 0.7],  # GenAI
        ]
    )
    context_vector = torch.zeros(inputs.shape)

    # attention_scores = await compute_attention_scores_for_single_query(inputs=inputs)
    attention_scores = await compute_attention_scores_for_whole_inputs(inputs=inputs)
    attention_weights = await compute_attention_weights(attention_scores=attention_scores)

    # ctx_vector = await compute_context_vector_for_query(attention_weights=attention_weights, inputs=inputs,
    #                                           context_vector=context_vector)
    ctx_vector = await compute_context_vector(attention_weights=attention_weights, inputs=inputs)

    print(f"attention scores: {attention_scores}")
    print(f"attention weights: {attention_weights}")
    print(f"context vector: {ctx_vector}")


if __name__ == "__main__":
    asyncio.run(main())
