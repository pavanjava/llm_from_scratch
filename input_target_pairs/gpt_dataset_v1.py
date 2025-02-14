import torch
from tiktoken.core import Encoding
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1:
    def __init__(self, text: str, tokenizer: Encoding, max_context_length: int = 200, stride: int = 1):
        self.input_tensors = []
        self.output_tensors = []

        token_ids = tokenizer.encode(text=text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_context_length, stride):
            input_chunk = token_ids[i:i + max_context_length]
            output_chunk = token_ids[i + 1: i + max_context_length + 1]

            self.input_tensors.append(torch.tensor(input_chunk))
            self.output_tensors.append(torch.tensor(output_chunk))
