from shredword import shredword

with open("captions.txt", "r", encoding="utf-8") as f:
  data = f.read()
  print("data imported")
  f.close()

with open("new.txt", "r", encoding="utf-8") as f:
  test = f.read()
  f.close()

tokenizer = shredword()
tokenizer.train(data, 356, True)
encoded = tokenizer.encode(test)
tokenizer.save("vocab/main_vocab")
print(encoded)
decoded = tokenizer.decode(encoded)
print(decoded)
print(test == decoded)