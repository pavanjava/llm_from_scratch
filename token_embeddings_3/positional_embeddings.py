# demonstrate the positional embeddings
from input_target_pairs_2.gpt_dataset_v1 import create_dataloader
import torch


class InputEmbeddings:
    def __init__(self, max_length: int = 128, output_dim: int = 256, verbose: bool = False):
        vocab_size = 50257
        self.verbose = verbose
        self.max_length = max_length
        self.output_dim = output_dim
        self.token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
        self.positional_embedding_layer = torch.nn.Embedding(max_length, output_dim)

    def _fetch_raw_input(self):
        with open('../the-verdict.txt', 'r') as f:
            raw_text = f.read()
        return raw_text

    def compute_token_embeddings(self):
        raw_text = self._fetch_raw_input()
        data_loader = create_dataloader(text=raw_text, batch_size=8, max_context_length=self.max_length,
                                        stride=self.max_length, shuffle=True)
        data_iter = iter(data_loader)
        inputs, targets = next(data_iter)
        token_embeddings = self.token_embedding_layer(inputs)
        if self.verbose:
            print(inputs.shape)
            print(targets.shape)
            print(token_embeddings.shape)
        return token_embeddings

    def compute_positional_embeddings(self):
        positional_embeddings = self.positional_embedding_layer(torch.arange(self.max_length))
        if self.verbose:
            print(positional_embeddings.shape)
        return positional_embeddings

    def text_embeddings(self):
        input_embeddings = self.compute_token_embeddings() + self.compute_positional_embeddings()
        if self.verbose:
            print(input_embeddings.shape)
        return input_embeddings





ie = InputEmbeddings()
input_embeddings = ie.text_embeddings()

print(input_embeddings.shape)
