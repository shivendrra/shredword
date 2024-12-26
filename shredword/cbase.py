import ctypes
import os

lib_path = os.path.join(os.path.dirname(__file__), 'libtoken.so')
lib = ctypes.CDLL(lib_path)

class BaseTokenizer(ctypes.Structure):
  pass

class Shred(ctypes.Structure):
  _fields_ = [
    ('base', ctypes.POINTER(BaseTokenizer)),
  ]

# Function prototypes
lib.init_shred.argtypes = [ctypes.POINTER(Shred)]
lib.init_shred.restype = None

lib.train.argtypes = [ctypes.POINTER(Shred), ctypes.c_char_p, ctypes.c_int, ctypes.c_bool]
lib.train.restype = None

lib.decode.argtypes = [ctypes.POINTER(Shred), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.decode.restype = ctypes.c_char_p

lib.encode.argtypes = [ctypes.POINTER(Shred), ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
lib.encode.restype = ctypes.POINTER(ctypes.c_int)

lib.print_merges.argtypes = [ctypes.POINTER(Shred)]
lib.print_merges.restype = None

lib.print_vocab.argtypes = [ctypes.POINTER(Shred)]
lib.print_vocab.restype = None

lib.save_model.argtypes = [ctypes.POINTER(Shred), ctypes.c_char_p]
lib.save_model.restype = None

lib.load_model.argtypes = [ctypes.POINTER(Shred), ctypes.c_char_p]
lib.load_model.restype = None

lib.export_merges.argtypes = [ctypes.POINTER(Shred)]
lib.export_merges.restype = ctypes.c_char_p

lib.export_vocab.argtypes = [ctypes.POINTER(Shred)]
lib.export_vocab.restype = ctypes.c_char_p


class Tokenizer:
  def __init__(self):
    self.tokenizer = Shred()
    lib.init_shred(ctypes.byref(self.tokenizer))

  def train(self, text: str, vocab_size: int, verbose: bool = True): lib.train(ctypes.byref(self.tokenizer), text.encode('utf-8'), vocab_size, verbose)
  def print_merges(self): lib.print_merges(ctypes.byref(self.tokenizer))
  def print_vocab(self): lib.print_vocab(ctypes.byref(self.tokenizer))
  def save_model(self, file_path: str): lib.save_model(ctypes.byref(self.tokenizer), file_path.encode('utf-8'))
  def load_model(self, file_path: str): lib.load_model(ctypes.byref(self.tokenizer), file_path.encode('utf-8'))
  def export_merges(self) -> str: return (lib.export_merges(ctypes.byref(self.tokenizer))).decode('utf-8')
  def export_vocab(self) -> str: return (lib.export_vocab(ctypes.byref(self.tokenizer))).decode('utf-8')

  def encode(self, text: str):
    output_size = ctypes.c_int()
    ids_ptr = lib.encode(ctypes.byref(self.tokenizer), text.encode('utf-8'), ctypes.byref(output_size))
    ids = [ids_ptr[i] for i in range(output_size.value)]
    lib.free(ids_ptr)
    return ids

  def decode(self, ids: list[int]):
    ids_array = (ctypes.c_int * len(ids))(*ids)
    decoded = lib.decode(ctypes.byref(self.tokenizer), ids_array, len(ids))
    return decoded.decode('utf-8')