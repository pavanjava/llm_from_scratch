import tiktoken


def get_data():
    with open('../the-verdict.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        return text


def encode(text: str):
    tokenizer = tiktoken.get_encoding("gpt2")
    return tokenizer.encode(text=text)


text_ = get_data()
encoding = encode(text=text_)
print(len(encoding))
