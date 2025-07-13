import ctypes, os, sys, platform, sysconfig
from ctypes import Structure, c_float, c_int, c_int32, c_uint64, c_size_t, c_char_p, POINTER, c_bool
from typing import *

def _get_lib_path():  
  pkg_dir = os.path.dirname(__file__)
  possible_names = ['trainer', 'libtrainer']
  possible_exts = ['.pyd', '.dll', '.so', '.dylib', sysconfig.get_config_var('EXT_SUFFIX') or '']
  search_dirs = [pkg_dir, os.path.join(pkg_dir, 'lib'), os.path.join(pkg_dir, '..', 'build')]

  for search_dir in search_dirs:
    if not os.path.exists(search_dir): continue
    for root, dirs, files in os.walk(search_dir):
      for file in files:
        for name in possible_names:
          if file.startswith(name) and any(file.endswith(ext) for ext in possible_exts if ext):
            return os.path.join(root, file)
  
  raise FileNotFoundError(f"Could not find trainer library in {search_dirs}. Available files: {[f for d in search_dirs if os.path.exists(d) for f in os.listdir(d)]}")
lib = ctypes.CDLL(_get_lib_path())

MIN_HEAP_SIZE = 4096
MAX_OCCS_PER_MERGE = 50000
INITIAL_VOCAB_SIZE = 256
INITIAL_STR_SIZE = 4096

class Symbol(Structure): pass
class WordPos(Structure): pass
class Corpus(Structure): pass
class BPEConfig(Structure): pass
class Trainer(Structure): pass
class MaxHeap(Structure): pass
class BIMap(Structure): pass
class PairKey(Structure): pass

# populating fields------------
Symbol._fields_ = [("id", c_int32), ("prev", POINTER(Symbol)), ("next", POINTER(Symbol)), ("deleted", c_bool)]
WordPos._fields_ = [("word_index", c_size_t), ("pos", POINTER(Symbol))]
Corpus._fields_ = [("words", POINTER(POINTER(Symbol))), ("word_counts", POINTER(c_uint64)), ("vocab_size", c_size_t)]
BPEConfig._fields_ = [("target_vocab_size", c_size_t), ("unk_id", c_int32), ("character_coverage", c_float), ("min_pair_freq", c_uint64)]
Trainer._fields_ = [("config", BPEConfig), ("heap", MaxHeap), ("corpus", Corpus), ("bigram_map", BIMap), ("next_token", c_size_t), ("num_merges", c_size_t),
                    ("merge_ops", POINTER(PairKey)), ("token_strs", POINTER(c_char_p)), ("token_freq", POINTER(c_uint64))]

lib.create_trainer.argtypes = [POINTER(BPEConfig)]
lib.create_trainer.restype = POINTER(Trainer)
lib.bpe_trainer_destroy.argtypes = [POINTER(Trainer)]
lib.bpe_trainer_destroy.restype = None
lib.bpe_init.argtypes = [POINTER(Trainer)]
lib.bpe_init.restype = None
lib.bpe_count_bigrams.argtypes = [POINTER(Trainer)]
lib.bpe_count_bigrams.restype = None
lib.bpe_load_corpus.argtypes = [POINTER(Trainer), c_char_p]
lib.bpe_load_corpus.restype = c_int
lib.bpe_merge_batch.argtypes = [POINTER(Trainer), c_int]
lib.bpe_merge_batch.restype = c_int
lib.bpe_train.argtypes = [POINTER(Trainer)]
lib.bpe_train.restype = c_int
lib.bpe_save.argtypes = [POINTER(Trainer), c_char_p, c_char_p]
lib.bpe_save.restype = None