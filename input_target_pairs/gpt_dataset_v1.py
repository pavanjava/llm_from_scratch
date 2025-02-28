import tiktoken
import torch
from tiktoken.core import Encoding
from torch.utils.data import Dataset, DataLoader


class GPTDatasetV1(Dataset):
    def __init__(self, text: str, tokenizer: Encoding, max_context_length: int = 256, stride: int = 128):
        self.input_tensors = []
        self.output_tensors = []

        token_ids = tokenizer.encode(text=text, allowed_special={"<|endoftext|>"})

        for i in range(0, len(token_ids) - max_context_length, stride):
            input_chunk = token_ids[i:i + max_context_length]
            output_chunk = token_ids[i + 1: i + max_context_length + 1]

            self.input_tensors.append(torch.tensor(input_chunk))
            self.output_tensors.append(torch.tensor(output_chunk))

    def __len__(self):
        return len(self.input_tensors)

    def __getitem__(self, idx):
        return self.input_tensors[idx], self.output_tensors[idx]


def create_dataloader(text, batch_size=4, max_context_length=256, stride=128, shuffle=True, drop_last=True,
                      num_workers=0):
    tokenizer = tiktoken.get_encoding('gpt2')

    gpt_dataset = GPTDatasetV1(text=text, tokenizer=tokenizer, max_context_length=max_context_length, stride=stride)

    data_loader = DataLoader(
        dataset=gpt_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last
    )

    return data_loader


if __name__ == '__main__':
    with open('../the-verdict.txt', 'r') as f:
        text = f.read()

    dataloader = create_dataloader(text=text, batch_size=10, max_context_length=4, stride=4,
                                   shuffle=False, num_workers=2)

    # data_iter = iter(dataloader)
    # first_batch = next(data_iter)
    # print(first_batch)
