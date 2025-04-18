from src import Shred
import timeit

tokenizer = Shred()
input_file = "test data/final.txt"
train_file = "test data/new.txt"

with open(input_file, "r", encoding="utf-8") as f:
  text = f.read()

VOCAB_SIZE = 1256
# start_time = timeit.default_timer()
# tokenizer.train(text, VOCAB_SIZE)
# tokenizer.save("vocab/trained_vocab")
# end_time = timeit.default_timer()
# print("\n\ntime taken: ", (end_time - start_time) )
tokenizer.load("vocab/trained_vocab.model")

with open(train_file, "r", encoding="utf-8") as f:
  train = f.read()

encoded = tokenizer.encode(text)
print("Encoded:", encoded[:500])

decoded = tokenizer.decode(encoded)
print("Decoded:", decoded[:1000])