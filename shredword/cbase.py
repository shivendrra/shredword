import ctypes
import os

lib_path = os.path.join(os.path.dirname(__file__), "./../build/libtoken.so")
lib = ctypes.CDLL(lib_path)

class BaseTokenizer(ctypes.Structure):
  _fields_ = [
    ("vocab", ctypes.c_char_p * (256 + 10000)),
    ("merges", ctypes.c_int * 10000),
    ("merge_count", ctypes.c_int),
    ("vocab_size", ctypes.c_int),
    ("special_token_indices", ctypes.c_int * 100),
    ("special_token_count", ctypes.c_int),
    ("special_tokens", (ctypes.c_char * 2048) * 100),
    ("pattern", ctypes.c_char * 2048),
  ]

class Tokenizer(ctypes.Structure):
  _fields_ = [
    ("base", BaseTokenizer),
  ]

lib.init_tokenizer.argtypes = [ctypes.POINTER(BaseTokenizer)]
lib.init_tokenizer.restype = None

lib.init_shred.argtypes = [ctypes.POINTER(Tokenizer)]
lib.init_shred.restype = None

lib.train.argtypes = [ctypes.POINTER(Tokenizer), ctypes.c_char_p, ctypes.c_int]
lib.train.restype = None

lib.encode.argtypes = [ctypes.POINTER(Tokenizer), ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
lib.encode.restype = ctypes.POINTER(ctypes.c_int)

lib.decode.argtypes = [ctypes.POINTER(Tokenizer), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.decode.restype = ctypes.c_char_p

lib.save_model.argtypes = [ctypes.POINTER(Tokenizer), ctypes.c_char_p]
lib.save_model.restype = None

lib.load_model.argtypes = [ctypes.POINTER(Tokenizer), ctypes.c_char_p]
lib.load_model.restype = None

lib.export_merges.argtypes = [ctypes.POINTER(Tokenizer)]
lib.export_merges.restype = ctypes.c_char_p

lib.export_vocab.argtypes = [ctypes.POINTER(Tokenizer)]
lib.export_vocab.restype = ctypes.c_char_p

class Shred:
  def __init__(self):
    self.tokenizer = Tokenizer()
    lib.init_shred(ctypes.byref(self.tokenizer))
  
  def __repr__(self):
    return f"\nShredWord Tokenizer, initialized!\n"

  def train(self, text, vocab_size):
    lib.train(ctypes.byref(self.tokenizer), text.encode("utf-8"), vocab_size)

  def encode(self, text):
    output_size = ctypes.c_int()
    encoded = lib.encode(ctypes.byref(self.tokenizer), text.encode("utf-8"), ctypes.byref(output_size))
    return [encoded[i] for i in range(output_size.value)]

  def decode(self, ids):
    array_type = ctypes.c_int * len(ids)
    id_array = array_type(*ids)
    return lib.decode(ctypes.byref(self.tokenizer), id_array, len(ids)).decode("utf-8")

  def save_model(self, file_path):
    file_path_c = ctypes.create_string_buffer(file_path.encode("utf-8"))
    lib.save_model(ctypes.byref(self.tokenizer), file_path_c)

  def load_model(self, file_path):
    lib.load_model(ctypes.byref(self.tokenizer), file_path.encode("utf-8"))

  def export_merges(self):
    return lib.export_merges(ctypes.byref(self.tokenizer)).decode("utf-8")

  def export_vocab(self):
    return lib.export_vocab(ctypes.byref(self.tokenizer)).decode("utf-8")