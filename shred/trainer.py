import ctypes, os
from ctypes import *
from typing import *
import regex as re

lib_path = os.path.join(os.path.dirname(__file__, "../build/libtrainer.so"))
lib = ctypes.CDLL(lib_path)

# all constants go here ->
INITIAL_VOCAB_SIZE = 256
MIN_HEAP_SIZE = 4096

# forward declaration for to self-refrence
class Symbol(ctypes.Structure): pass
class wordPos(ctypes.Structure): pass
class Corpus(ctypes.Structure): pass
class BPEConfig(ctypes.Structure): pass
class BPETrainer(ctypes.Structure): pass
class HeapEntry(ctypes.Structure): pass
class MaxHeap(ctypes.Structure): pass
class PairKey(ctypes.Structure): pass
class StrEntry(ctypes.Structure): pass
class StrMap(ctypes.Structure): pass
class Info(ctypes.Structure): pass
class BIEntry(ctypes.Structure): pass
class BIMap(ctypes.Structure): pass

# populating the fields
Symbol._fields_ = [("id", c_int32), ("prev", POINTER(Symbol)), ("next", POINTER(Symbol))]
wordPos._fields_ = [("word_index", c_size_t), ("pos", POINTER(Symbol))]
Corpus._fields_ = [("words", POINTER(POINTER(Symbol))), ("word_counts", POINTER(c_uint64)), ("vocab_size", c_size_t)]
PairKey._fields_ = [("first", c_int32), ("second", c_int32)]
HeapEntry._fields_ = [("key", PairKey), ("freq", c_uint64), ("version", c_uint32)]
MaxHeap._fields_ = [("data", POINTER(HeapEntry)), ("size", c_size_t), ("cap", c_size_t)]
Info._fields_ = [("freq", c_uint64), ("positions", POINTER(wordPos))]
BIEntry._fields_ = [("key", PairKey), ("info", Info), ("next", POINTER(BIEntry))]
BIMap._fields_ = [("buckets", POINTER(POINTER(BIEntry))), ("nbuckets", c_size_t)]
StrEntry._fields_ = [("key", c_char_p), ("value", c_uint64), ("next", POINTER(StrEntry))]
StrMap._fields_ = [("buckets", POINTER(POINTER(StrEntry))), ("nbuckets", c_size_t)]
BPEConfig._fields_ = [("target_vocab", c_size_t), ("unk_id", c_int32), ("character_coverage", c_double), ("min_pair_freq", c_uint64)]
BPETrainer._fields_ = [
  ("config", BPEConfig), 
  ("heap", MaxHeap),
  ("corpus", Corpus),
  ("bigram_map", BIMap),
  ("next_token_id", c_size_t),
  ("initial_vocab_size", c_size_t),
  ("merge_ops", POINTER(PairKey)),
  ("token_strs", c_char_p),
  ("token_freqs", c_uint64)
]