from src.tarp.services.tokenizers.pretrained.character import CharacterTokenizer

tokenizer = CharacterTokenizer()

text = "Hello, World!"
tokenized = tokenizer.tokenize(text)

print("Original text:", text)
print("Tokenized:", tokenized)
