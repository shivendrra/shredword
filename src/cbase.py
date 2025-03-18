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
lib.train_with_cache.argtypes = [ctypes.POINTER(CShred), ctypes.c_char_p, ctypes.c_int]
lib.train_with_cache.restype = None
lib.train_optimized.argtypes = [ctypes.POINTER(CShred), ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
lib.train_optimized.restype = None
lib.encode.argtypes = [ctypes.POINTER(CShred), ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
lib.encode.restype = ctypes.POINTER(ctypes.c_int)
lib.decode.argtypes = [ctypes.POINTER(CShred), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.decode.restype = ctypes.c_char_p
lib.encode_with_cache.argtypes = [ctypes.POINTER(CShred), ctypes.c_char_p, ctypes.POINTER(ctypes.c_int)]
lib.encode_with_cache.restype = ctypes.POINTER(ctypes.c_int)
lib.decode_with_cache.argtypes = [ctypes.POINTER(CShred), ctypes.POINTER(ctypes.c_int), ctypes.c_int]
lib.decode_with_cache.restype = ctypes.c_char_p
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
lib.set_pattern.argtypes = [ctypes.POINTER(CShred), ctypes.c_char_p]
lib.set_pattern.restype = None
# lib.set_special_tokens.argtypes = [ctypes.POINTER(CShred), ctypes.c_char_p]
# lib.set_sepcial_tokens.restype = None
lib.free_string.argtypes = [ctypes.c_char_p]
lib.free_string.restype = None