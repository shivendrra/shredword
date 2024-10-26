# import ctypes
# import os

# class Tokenizer:
#   def __init__(self, vocab_size=256, lib_path="libtokenizer.so"):
#     self.vocab_size = vocab_size
#     # Load the shared library (.so)
#     self.tokenizer = ctypes.CDLL(os.path.join(os.getcwd(), lib_path))

#     # Set return types and argument types for C functions
#     self.tokenizer.init_vocab.restype = None
#     self.tokenizer.free_vocab.restype = None

#     self.tokenizer.encode.argtypes = [ctypes.c_char_p, 
#                                       ctypes.POINTER(ctypes.POINTER(ctypes.c_int)), 
#                                       ctypes.POINTER(ctypes.c_size_t)]
#     self.tokenizer.encode.restype = None

#     self.tokenizer.decode.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_size_t]
#     self.tokenizer.decode.restype = ctypes.c_char_p

#     self.tokenizer.train.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_bool]
#     self.tokenizer.train.restype = None

#   def init_vocab(self):
#     """Initialize the tokenizer's vocabulary."""
#     self.tokenizer.init_vocab()

#   def free_vocab(self):
#     """Free the memory allocated for the vocabulary."""
#     self.tokenizer.free_vocab()

#   def encode(self, text):
#     """Encode the given text into token IDs."""
#     length = ctypes.c_size_t()
#     ids = ctypes.POINTER(ctypes.c_int)()

#     # Call the C encode function
#     self.tokenizer.encode(text.encode('utf-8'), ctypes.byref(ids), ctypes.byref(length))

#     # Convert the C pointer to a Python list
#     result = [ids[i] for i in range(length.value)]
#     # Free the allocated memory from the C side if necessary

#     return result

#   def decode(self, ids):
#     """Decode a list of token IDs into text."""
#     array_type = ctypes.c_int * len(ids)
#     result = self.tokenizer.decode(array_type(*ids), len(ids))
#     return result.decode('utf-8')

#   def train(self, text, verbose=False):
#     """Train the tokenizer on the provided text."""
#     self.tokenizer.train(text.encode('utf-8'), self.vocab_size, verbose)

#   def save_model(self, filename):
#     """Save the trained model to a file."""
#     self.tokenizer.save_model(filename.encode('utf-8'))

#   def load_model(self, filename):
#     """Load a saved model from a file."""
#     self.tokenizer.load_model(filename.encode('utf-8'))

#   def tokenize_file(self, filename, verbose=False):
#     """Tokenize a text file by training on its content."""
#     with open(filename, 'r', encoding='utf-8') as f:
#       text = f.read()
#     self.train(text, verbose=verbose)


# if __name__ == "__main__":
#   tokenizer = Tokenizer(vocab_size=356)

#   try:
#     tokenizer.init_vocab()
#     tokenizer.tokenize_file("training_data.txt", verbose=True)
#     encoded = tokenizer.encode("Hello, World!")
#     print("Encoded:", encoded)
#     decoded = tokenizer.decode(encoded)
#     print("Decoded:", decoded)
#     tokenizer.save_model("tokenizer_model.txt")
#     tokenizer.load_model("tokenizer_model.txt")
#     encoded = tokenizer.encode("Hello, World!")
#     print("Encoded:", encoded)
#     decoded = tokenizer.decode(encoded)
#     print("Decoded:", decoded)

#   finally:
#     tokenizer.free_vocab()

import ctypes
import json
import os

class Tokenizer:
  def __init__(self, vocab_size):
    self.tokenizer = ctypes.CDLL('./libtokenizer.so')
    self.tokenizer.build_vocab.argtypes = [ctypes.c_char_p]
    self.tokenizer.build_vocab.restype = None
    self.tokenizer.merge.argtypes = [ctypes.c_char_p]
    self.tokenizer.merge.restype = None
    self.tokenizer.train.argtypes = [ctypes.c_char_p]
    self.tokenizer.train.restype = None

    self.tokenizer.encode.argtypes = [ctypes.c_char_p]
    self.tokenizer.encode.restype = ctypes.POINTER(ctypes.c_int)

    self.tokenizer.decode.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_size_t]
    self.tokenizer.decode.restype = ctypes.c_char_p

    self.vocab = {}
    self.merges = []
    self.vocab_size = vocab_size

  def build_vocab(self, text):
    """Call the C function to build vocab."""
    text_bytes = text.encode('utf-8')
    self.tokenizer.build_vocab(text_bytes)

  def merge(self, merge_data):
    """Call the C function to perform merges."""
    merge_data_bytes = merge_data.encode('utf-8')
    self.tokenizer.merge(merge_data_bytes)

  def train(self, data):
    """Train the tokenizer using C function."""
    data_bytes = data.encode('utf-8')
    self.tokenizer.train(data_bytes)

  def encode(self, text):
    """Encode text using the C function and return token IDs."""
    text_bytes = text.encode('utf-8')
    encoded_ptr = self.tokenizer.encode(text_bytes)

    encoded_tokens = []
    i = 0
    while encoded_ptr[i] != -1:
      encoded_tokens.append(encoded_ptr[i])
      i += 1

    return encoded_tokens

  def decode(self, token_ids):
    """Decode token IDs back to text using the C function."""
    array_type = ctypes.c_int * len(token_ids)
    token_array = array_type(*token_ids)
    decoded_text = self.tokenizer.decode(token_array, len(token_ids))
    return decoded_text.decode('utf-8')

  def save_model(self, vocab_path='vocab.json', merges_path='merges.txt'):
    """Save vocab and merges to disk using Python."""
    with open(vocab_path, 'w') as vocab_file:
      json.dump(self.vocab, vocab_file, indent=2)

    with open(merges_path, 'w') as merges_file:
      merges_file.write('\n'.join(self.merges))

  def load_model(self, vocab_path='vocab.json', merges_path='merges.txt'):
    """Load vocab and merges from disk."""
    if os.path.exists(vocab_path):
      with open(vocab_path, 'r') as vocab_file:
        self.vocab = json.load(vocab_file)

    if os.path.exists(merges_path):
      with open(merges_path, 'r') as merges_file:
        self.merges = merges_file.read().splitlines()

if __name__ == "__main__":
  tokenizer = Tokenizer(vocab_size=356)

  with open("training_data.txt", "r", encoding="utf-8") as f:
    data = f.read()
    print("file opened")
    f.close()
  tokenizer.train(data)

  tokens = tokenizer.encode("Hello world!")
  print("Encoded tokens:", tokens)
  decoded_text = tokenizer.decode(tokens)
  print("Decoded text:", decoded_text)

  tokenizer.save_model()
  tokenizer.load_model()
  print("Loaded vocab:", tokenizer.vocab)
  print("Loaded merges:", tokenizer.merges)