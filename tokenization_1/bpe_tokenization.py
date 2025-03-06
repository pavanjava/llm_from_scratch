import tiktoken

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "pavankumarmantha is a good programmer.<|endoftext|>"
    "deepakmantha is as good as pavanmantha."
)

ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(ids)
text_constructed = tokenizer.decode(tokens=ids)
print(text_constructed)