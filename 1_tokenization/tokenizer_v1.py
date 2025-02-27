import re


class SimpleTokenizerV1:
    def __init__(self, vocab: dict):
        self.vocab = vocab
        self.reverse_vocab = {i: s for s, i in vocab.items()}

    def encoder(self, text):
        text = re.split(r'([,.:;!@#$%^&*?_"()\']|--|\s)', text)
        pre_processed = [
            item.strip() for item in text if item.strip()
        ]
        ids = [self.vocab.get(word, -1) for word in pre_processed]
        return ids

    def decode(self, ids):
        words = [self.reverse_vocab.get(id) for id in ids]
        pre_processed = [
            item.strip() for item in words if item.strip()
        ]

        return pre_processed


if __name__ == "__main__":
    with open("../the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    processed_text = re.split(r'([,.:;!@#$%^&*?_"()\']|--|\s)', raw_text)
    processed_tokens = [text.strip() for text in processed_text if text.strip()]
    all_words = sorted(set(processed_tokens))
    vocab = {word: idx for idx, word in enumerate(all_words)}
    tokenizer = SimpleTokenizerV1(vocab=vocab)

    ids = tokenizer.encoder(text="I found the couple at tea beneath their palm-trees; and Mrs. Gisburn's welcome was so genial that, in the ensuing weeks, I claimed it frequently")
    print(ids)
    words = tokenizer.decode(ids=ids)
    print(" ".join(words))
