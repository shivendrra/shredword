import ctypes, os
from typing import *
import regex as re

# define paths to the DLL
lib_path = os.path.join(os.path.dirname(__file__), "../build/libtoken.so")
lib = ctypes.CDLL(lib_path)

# define constants
VOCAB_SIZE = 256
MAX_SPECIAL_TOKENS = 100
MAX_MERGES = 10000
MAX_LINE_LENGTH = 2048

# define ctypes structures and functions
class Pair(ctypes.Structure):
  _fields_ = [
    ("idx1", ctypes.c_int),
    ("idx2", ctypes.c_int)
  ]

class VocabEntry(ctypes.Structure):
  _fields_ = [
    ("idx", ctypes.c_int),
    ("value", ctypes.c_char_p)
  ]

class MergeEntry(ctypes.Structure):
  _fields_ = [
    ("pair", Pair),
    ("idx", ctypes.c_int)
  ]

class BaseTokenizer(ctypes.Structure):
  _fields_ = [
    ("vocab", VocabEntry * (VOCAB_SIZE + MAX_MERGES)),
    ("merges", MergeEntry * MAX_MERGES),
    ("merge_count", ctypes.c_int),
    ("vocab_size", ctypes.c_int),
    ("special_token_indices", ctypes.c_int * MAX_SPECIAL_TOKENS),
    ("special_token_count", ctypes.c_int),
    ("special_tokens", ctypes.c_char * MAX_LINE_LENGTH * MAX_SPECIAL_TOKENS),
    ("pattern", ctypes.c_char * MAX_LINE_LENGTH)
  ]

class CShred(ctypes.Structure):
  _fields_ = [("base", BaseTokenizer)]

# Map the BaseTokenizer functions
lib.init_tokenizer.argtypes = [ctypes.POINTER(BaseTokenizer)]
lib.init_tokenizer.restype = None
lib.build_vocab.argtypes = [ctypes.POINTER(BaseTokenizer)]
lib.build_vocab.restype = None
lib.replace_control_characters.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.replace_control_characters.restype = None
lib.save_tokenizer.argtypes = [ctypes.POINTER(BaseTokenizer), ctypes.c_char_p]
lib.save_tokenizer.restype = None
lib.load_tokenizer.argtypes = [ctypes.POINTER(BaseTokenizer), ctypes.c_char_p]
lib.load_tokenizer.restype = None
lib.get_stats.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_int * 3 * MAX_MERGES)]
lib.get_stats.restype = None
lib.merge.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, Pair, ctypes.c_int, ctypes.POINTER(ctypes.c_size_t)]
lib.merge.restype = ctypes.POINTER(ctypes.c_int)
lib.render_token.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
lib.render_token.restype = None
lib.free_tokenizer.argtypes = [ctypes.POINTER(BaseTokenizer)]
lib.free_tokenizer.restype = None

# Map Shred functions
lib.init_shred.argtypes = [ctypes.POINTER(CShred)]
lib.init_shred.restype = None
lib.train.argtypes = [ctypes.POINTER(CShred), ctypes.c_char_p, ctypes.c_int]
lib.train.restype = None
lib.encode.argtypes = [ctypes.POINTER(CShred), ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
lib.encode.restype = ctypes.POINTER(ctypes.c_int)
lib.decode.argtypes = [ctypes.POINTER(CShred), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.decode.restype = ctypes.c_char_p
lib.save_model.argtypes = [ctypes.POINTER(CShred), ctypes.c_char_p]
lib.save_model.restype = None
lib.load_model.argtypes = [ctypes.POINTER(CShred), ctypes.c_char_p]
lib.load_model.restype = None
lib.export_merges.argtypes = [ctypes.POINTER(CShred)]
lib.export_merges.restype = ctypes.c_char_p
lib.export_special_tokens.argtypes = [ctypes.POINTER(CShred)]
lib.export_special_tokens.restype = ctypes.c_char_p
lib.export_pattern.argtypes = [ctypes.POINTER(CShred)]
lib.export_pattern.restype = ctypes.c_char_p
lib.free_string.argtypes = [ctypes.c_char_p]
lib.free_string.restype = None

class TokenizerBase:
  def __init__(self):
    self.tokenizer = BaseTokenizer()
    lib.init_tokenizer(ctypes.byref(self.tokenizer))

  def build_vocab(self):
    lib.build_vocab(ctypes.byref(self.tokenizer))

  def replace_control_characters(self, input_str):
    output = ctypes.create_string_buffer(MAX_LINE_LENGTH)
    lib.replace_control_characters(input_str.encode("utf-8"), output)
    return output.value.decode("utf-8")

  def save_tokenizer(self, file_path):
    lib.save_tokenizer(ctypes.byref(self.tokenizer), file_path.encode("utf-8"))

  def load_tokenizer(self, file_path):
    lib.load_tokenizer(ctypes.byref(self.tokenizer), file_path.encode("utf-8"))

  def free(self):
    lib.free_tokenizer(ctypes.byref(self.tokenizer))

class Shred:
  def __init__(self):
    self._tokenizer = CShred()
    lib.init_shred(ctypes.byref(self._tokenizer))

  def train(self, text, vocab_size):
    text_c = ctypes.create_string_buffer(text.encode("utf-8"))
    lib.train(ctypes.byref(self._tokenizer), text_c, vocab_size)

  def encode(self, text):
    text_c = ctypes.create_string_buffer(text.encode("utf-8"))
    output_size = ctypes.c_int()
    encoded_ptr = lib.encode(ctypes.byref(self._tokenizer), text_c, ctypes.byref(output_size))
    encoded = [encoded_ptr[i] for i in range(output_size.value)]
    return encoded

  def decode(self, ids):
    array_type = ctypes.c_int * len(ids)
    id_array = array_type(*ids)
    decoded_ptr = lib.decode(ctypes.byref(self._tokenizer), id_array, len(ids))
    decoded = ctypes.string_at(decoded_ptr).decode("utf-8")
    return decoded

  def save(self, file_path):
    file_path_c = ctypes.create_string_buffer(file_path.encode("utf-8"))
    lib.save_model(ctypes.byref(self._tokenizer), file_path_c)
    print("Saved the model sucessfully!!")

  def load(self, file_path):
    file_path_c = ctypes.create_string_buffer(file_path.encode("utf-8"))
    lib.load_model(ctypes.byref(self._tokenizer), file_path_c)
    print("Loaded the model sucessfully!!")

  def _build_vocab(self):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for p0, p1, idx in self.merges:
      vocab[idx] = vocab[p0] + vocab[p1]
    return vocab

  @property
  def merges(self):
    merges = lib.export_merges(ctypes.byref(self._tokenizer))
    result = ctypes.string_at(merges).decode("utf-8")
    # regex to match (p0, p1) idx pattern
    pattern = re.compile(r"\((\d+), (\d+)\) (\d+)")
    return [tuple(map(int, match.groups())) for match in pattern.finditer(result)]

  @property
  def vocab(self):
    return self._build_vocab()

  @property
  def pattern(self):
    pattern = lib.export_pattern(ctypes.byref(self._tokenizer))
    if not pattern: return ["None"]
    result = ctypes.string_at(pattern).decode("utf-8").strip()
    return result

  @pattern.setter
  def pattern(self, new_pattern):
    lib.set_pattern(ctypes.byref(self.tokenizer), new_pattern.encode("utf-8"))

  @property
  def special_tokens(self):
    tokens = lib.export_special_tokens(ctypes.byref(self._tokenizer))
    if not tokens: return ["None"]
    result = ctypes.string_at(tokens).decode("utf-8").strip().splitlines()
    return result
  
  @special_tokens.setter
  def special_tokens(self, token_list):
    serialized = "\n".join(f"{token} {index}" for token, index in token_list)
    lib.load_special_tokens(ctypes.byref(self.tokenizer), serialized.encode("utf-8"))

  # def __del__(self):
  #   lib.free_tokenizer(ctypes.byref(self._tokenizer))