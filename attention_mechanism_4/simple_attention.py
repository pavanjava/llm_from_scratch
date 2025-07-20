import torch
import asyncio


async def compute_attention_scores_for_single_query(inputs: torch.tensor, debug: bool = False):
    attn_scores = torch.empty(inputs.shape)
    query = inputs[1]
    for i, x_i in enumerate(inputs):
       attn_scores[i] = torch.dot(x_i, query) # dot product to compute the similarity of the words
    if debug:
        print(f"Raw Attention scores: {attn_scores}")
    return attn_scores

async def compute_attention_scores_for_whole_inputs(inputs: torch.tensor):
    return inputs @ inputs.T

async def compute_attention_weights(attention_scores: torch.tensor, debug: bool = False):
    # standard torch implementation of softmax
    attn_weights = torch.softmax(attention_scores, dim=-1)
    if debug:
        print(f"std.softmax normalised attention (attention weights): {attn_weights}")
    return attn_weights

# attn_scores_normalized = attention_scores / attention_scores.sum()
# print(f"Normalised attention scores: {attn_scores_normalized}")

# naive implementation of softmax
# def softmax_naive(x):
#     return torch.exp(x) / torch.exp(x).sum(dim=0)
#
# attn_scores_normalized_2 = softmax_naive(attention_scores)
# print(f"naive.softmax normalised attention (attention weights): {attn_scores_normalized_2}")


async def main():
    inputs = torch.tensor(
        [
            [0.8, 0.6, 0.1],        # I
            [0.7, 0.7, 0.1],        # Love
            [0.2, 0.8, 0.6],        # Coding
            [0.1, 0.7, 0.7],        # GenAI
        ]
    )

    # attention_scores = await compute_attention_scores_for_single_query(inputs=inputs)
    attention_scores = await compute_attention_scores_for_whole_inputs(inputs=inputs)
    attention_weights = await compute_attention_weights(attention_scores=attention_scores)

    print(f"attention scores: {attention_scores}")
    print(f"attention weights: {attention_weights}")


if __name__ == "__main__":
    asyncio.run(main())