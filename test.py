from shredword import shred

with open("test data/captions.txt", "r", encoding="utf-8") as f:
  data = f.read()
  print("data imported")
  f.close()

with open("test data/new.txt", "r", encoding="utf-8") as f:
  test = f.read()
  f.close()

# training + save logic
tokenizer = shred()
tokenizer.train(data, 280, True)
tokenizer.save("vocab/main_vocab")
encoded = tokenizer.encode(test)
print(encoded)
decoded = tokenizer.decode(encoded)
print(decoded)
print(test == decoded)

# ## loading & encoding/decoding logic
# import timeit

# tokenizer = shred()
# tokenizer.load('vocab/main_vocab.model')

# start_time = timeit.default_timer()
# encoded = tokenizer.encode(data)
# end_time = timeit.default_timer()
# print(f"total characters: {len(data)}")
# print(f"encoded length: {len(encoded)}")
# print("compression ratio: ", (len(data) / len(encoded)))
# print("encoding time taken: ", (end_time - start_time) / 60, "mins")
# print(encoded[:200])
# decoded = tokenizer.decode(encoded)
# print(decoded[:200])
# print(data == decoded)