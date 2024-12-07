from shredword import shred

with open("test data/captions.txt", "r", encoding="utf-8") as f:
  data = f.read()
  print("data imported")
  f.close()

with open("test data/new.txt", "r", encoding="utf-8") as f:
  test = f.read()
  f.close()

## training + save logic
# tokenizer = shred()
# tokenizer.train(data, 356)
# tokenizer.save("vocab/main_vocab")
# encoded = tokenizer.encode(test)
# print(encoded)
# decoded = tokenizer.decode(encoded)
# print(decoded)
# print(test == decoded)

## loading & encoding/decoding logic
tokenizer = shred()
tokenizer.load('vocab/main_vocab.model')
encoded = tokenizer.encode(data)
print(encoded[:200])
decoded = tokenizer.decode(encoded)
print(decoded[:200])
print(test == decoded)