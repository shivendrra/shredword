import ctypes
import os

class Tokenizer:
  def __init__(self, vocab_size=256, lib_path="libtokenizer.so"):
    self.vocab_size = vocab_size
    self.tokenizer = ctypes.CDLL(os.path.join(os.getcwd(), lib_path))

    self.tokenizer.init_vocab.restype = None
    self.tokenizer.free_vocab.restype = None

    self.tokenizer.encode.argtypes = [ctypes.c_char_p, 
                                       ctypes.POINTER(ctypes.POINTER(ctypes.c_int)), 
                                       ctypes.POINTER(ctypes.c_size_t)]
    self.tokenizer.encode.restype = None

    self.tokenizer.decode.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_size_t]
    self.tokenizer.decode.restype = ctypes.c_char_p

    self.tokenizer.train.argtypes = [ctypes.c_char_p, ctypes.c_size_t, ctypes.c_bool]
    self.tokenizer.train.restype = None

    self.tokenizer.save_model.argtypes = [ctypes.c_char_p]
    self.tokenizer.save_model.restype = None

    self.tokenizer.load_model.argtypes = [ctypes.c_char_p]
    self.tokenizer.load_model.restype = None

    self.tokenizer.free_vocab.argtypes = []

  def init_vocab(self):
    self.tokenizer.init_vocab()

  def free_vocab(self):
    self.tokenizer.free_vocab()

  def encode(self, text):
    length = ctypes.c_size_t()
    ids = ctypes.POINTER(ctypes.c_int)()
    self.tokenizer.encode(text.encode('utf-8'), ctypes.byref(ids), ctypes.byref(length))
    
    # Convert C pointer to Python list
    return [ids[i] for i in range(length.value)]

  def decode(self, ids):
    array_type = ctypes.c_int * len(ids)
    result = self.tokenizer.decode(array_type(*ids), len(ids))
    return result.decode('utf-8')

  def train(self, text, verbose=False):
    self.tokenizer.train(text.encode('utf-8'), self.vocab_size, verbose)

  def save_model(self, filename):
    self.tokenizer.save_model(filename.encode('utf-8'))

  def load_model(self, filename):
    self.tokenizer.load_model(filename.encode('utf-8'))

  def tokenize_file(self, filename, verbose=False):
    with open(filename, 'r', encoding='utf-8') as f:
      text = f.read()
    self.train(text, verbose=verbose)

if __name__ == "__main__":
  tokenizer = Tokenizer(vocab_size=300)

  try:
    tokenizer.init_vocab()
    tokenizer.tokenize_file("training_data.txt")
    tokenizer.save_model("tokenizer_model.txt")
    tokenizer.load_model("tokenizer_model.txt")
  finally:
    tokenizer.free_vocab()