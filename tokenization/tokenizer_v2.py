from typing import List, Dict
import re


class SimpleTokenizerV2:
    def __init__(self, vocab: Dict):
        self.vocab = vocab
        self.reverse_vocab = {i: s for s, i in vocab.items()}

    def encode(self, text: str):
        text_split = re.split(r'([,.:;!@#$%^&*?_"()\']|--|\s)', text)
        tokens = [text.strip() for text in text_split if text.strip()]
        source_tokens = [token if token in self.vocab else "<unk>" for token in tokens]
        encoded = [self.vocab.get(token) for token in source_tokens]
        return encoded

    def decode(self, ids: List[int]):
        tokens = [self.reverse_vocab[idx] for idx in ids]
        constructed_text = " ".join(tokens)
        return re.sub(r'\s([?.!,"](?:\s|$))', r'\1', constructed_text)


if __name__ == "__main__":
    with open("../the-verdict.txt", "r") as f:
        raw_text = f.read()

    processed_text = re.split(r'([,.:;!@#$%^&*?_"()\']|--|\s)', raw_text)
    processed_tokens = [text.strip() for text in processed_text if text.strip()]
    processed_tokens.extend(["<unk>", "<endoftext>"])
    processed_tokens = sorted(set(processed_tokens))
    vocab = {token: idx+1 for idx, token in enumerate(processed_tokens)}

    st2 = SimpleTokenizerV2(vocab=vocab)
    query = "The morning i got to know"
    resp1 = st2.encode(text=query)
    print(query)
    print(resp1)
    resp2 = st2.decode(ids=resp1)
    print(resp2)
