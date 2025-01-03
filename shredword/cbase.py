import ctypes, os
import regex as re

lib_path = os.path.join(os.path.dirname(__file__), "./../build/libtoken.so")
lib = ctypes.CDLL(lib_path)

class CBase(ctypes.Structure):
  _fields_ = [
    ("vocab", ctypes.c_void_p * (256 + 10000)),
    ("merges", ctypes.c_void_p * 10000),
    ("merge_count", ctypes.c_int),
    ("vocab_size", ctypes.c_int),
    ("special_token_indices", ctypes.c_int * 100),
    ("special_token_count", ctypes.c_int),
    ("special_tokens", ctypes.c_char * 2048 * 100),
    ("pattern", ctypes.c_char * 2048),
  ]

class Tokenizer(ctypes.Structure):
  _fields_ = [("base", CBase)]

lib.init_tokenizer.argtypes = [ctypes.POINTER(CBase)]
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
lib.export_pattern.argtypes = [ctypes.POINTER(Tokenizer)]
lib.export_pattern.restype = ctypes.c_char_p
lib.export_special_tokens.argtypes = [ctypes.POINTER(Tokenizer)]
lib.export_special_tokens.restype = ctypes.c_char_p
lib.free_string.argtypes = [ctypes.c_char_p]
lib.free_string.restype = None

class Shred:
  def __init__(self):
    self.tokenizer = Tokenizer()
    lib.init_shred(ctypes.byref(self.tokenizer))
  
  def __repr__(self):
    return f"\nShredWord Tokenizer, initialized!\n"

  @property
  def merges(self):
    merges = lib.export_merges(ctypes.byref(self.tokenizer))
    result = ctypes.string_at(merges).decode("utf-8")
    # regex to match (p0, p1) idx pattern
    pattern = re.compile(r"\((\d+), (\d+)\) (\d+)")
    return [tuple(map(int, match.groups())) for match in pattern.finditer(result)]

  @property
  def vocab(self):
    return self._build_vocab()

  @property
  def pattern(self):
    pattern = lib.export_pattern(ctypes.byref(self.tokenizer))
    result = ctypes.string_at(pattern).decode("utf-8")
    return result

  @property
  def special_tokens(self):
    tokens = lib.export_special_tokens(ctypes.byref(self.tokenizer))
    if not tokens:
      raise RuntimeError("Failed to export special tokens.")
    result = ctypes.string_at(tokens).decode("utf-8").strip().splitlines()
    # parse each line into (special_token, index) tuples
    return [(line.rsplit(" ", 1)[0], int(line.rsplit(" ", 1)[1])) for line in result]

  def _build_vocab(self):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for p0, p1, idx in self.merges:
      vocab[idx] = vocab[p0] + vocab[p1]
    # for special, idx in self.special_tokens:
    #   vocab[idx] = special.encode("utf-8")
    return vocab

  def train(self, text, vocab_size):
    lib.train(ctypes.byref(self.tokenizer), text.encode("utf-8"), vocab_size)

  def encode(self, text):
    output_size = ctypes.c_int()
    encoded = lib.encode(ctypes.byref(self.tokenizer), text.encode("utf-8"), ctypes.byref(output_size))
    return [encoded[i] for i in range(output_size.value)]

  def decode(self, ids):
    array_type = ctypes.c_int * len(ids)
    id_array = array_type(*ids)
    return lib.decode(ctypes.byref(self.tokenizer), id_array, int(len(ids))).decode("utf-8", errors="replace")

  def save(self, file_prefix: str):
    # saves two files: ``.model`` & ``.vocab``
    # `.vocab` human readable version thats just for debugging & pretty presentation
    # `.model` for furthur training & implementing merges, can be loaded in model
    model_file = file_prefix + ".model"
    with open(model_file, 'w') as f:
      f.write("shredword v1\n")
      # f.write(f"{self.pattern}\n")
      # f.write(f"{len(self.special_tokens)}\n")
      # for special, idx in self.special_tokens:
      #   f.write(f"{special} {idx}\n")
      for p01, p02, idx in self.merges:
        f.write(f"{p01}, {p02} {idx}\n")

  def load_model(self, file_path: str):
    file_path = str(file_path)
    lib.load_model(ctypes.byref(self.tokenizer), file_path.encode("utf-8"))