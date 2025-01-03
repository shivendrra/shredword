from shredword import Shred

tokenizer = Shred()
input_file = "test data/captions.txt"
train_file = "test data/new.txt"

with open(input_file, "r", encoding="utf-8") as f:
  text = f.read()
with open(train_file, "r", encoding="utf-8") as f:
  train = f.read()

VOCAB_SIZE = 260
tokenizer.train(text, VOCAB_SIZE)
tokenizer.save("vocab/trained_vocab")
# tokenizer.load_model("vocab/vocab.model")

encoded = tokenizer.encode(train)
print("Encoded:", encoded)

decoded = tokenizer.decode(encoded)
print("Decoded:", decoded)