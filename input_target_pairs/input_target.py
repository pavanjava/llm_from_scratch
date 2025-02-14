import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")


def get_data():
    with open('../the-verdict.txt', 'r', encoding='utf-8') as f:
        text = f.read()
        return text


def encode(text: str):
    return tokenizer.encode(text=text)

 
text_ = get_data()
encoding = encode(text=text_)

context_length = 50

x = encoding[:context_length]
y = encoding[1:context_length + 1]

for i in range(1, context_length + 1):
    input_context = encoding[:i]
    output = encoding[i]

    print(f"{input_context} --> {output}")
    print(f"{tokenizer.decode(input_context)} --> {tokenizer.decode([output])}")
