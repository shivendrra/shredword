import os, ctypes
from ctypes import *

libtrainer_path = os.path.join(os.path.dirname(__file__), "../build/libtrainer.so")
libtrainer = ctypes.CDLL(libtrainer_path)

MIN_HEAP_SIZE = 4096
MAX_OCCS_PER_MERGE = 50000
INITIAL_VOCAB_SIZE = 256
INITIAL_STR_SIZE = 4096

# forward declarations------
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
Corpus._fields_ = [("words"), POINTER(POINTER(Symbol)), ("word_counts", POINTER(c_uint64)), ("vocab_size", c_size_t)]
BPEConfig._fields_ = [("target_vocab_size", c_size_t), ("unk_id", c_int32), ("character_coverage", c_float), ("min_pair_freq", c_uint64)]
Trainer._fields_ = [("config", BPEConfig), ("heap", MaxHeap), ("corpus", Corpus), ("bigram_map", BIMap), ("next_token", c_size_t), ("num_merges", c_size_t),
                    ("merge_ops", POINTER(PairKey)), ("token_strs", POINTER(c_char_p)), ("token_freq", POINTER(c_uint64))]


libtrainer.create_trainer.argtypes = [POINTER(BPEConfig)]
libtrainer.create_trainer.restype = POINTER(Trainer)
libtrainer.bpe_trainer_destroy.argtypes = [POINTER(Trainer)]
libtrainer.bpe_trainer_destroy.restype = None
libtrainer.bpe_init.argtypes = [POINTER(Trainer)]
libtrainer.bpe_init.restype = None
libtrainer.bpe_count_bigrams.argtypes = [POINTER(Trainer)]
libtrainer.bpe_count_bigrams.restype = None
libtrainer.bpe_load_corpus.argtypes = [POINTER(Trainer), c_char_p]
libtrainer.bpe_load_corpus.restype = c_int
libtrainer.bpe_merge_batch.argtypes = [POINTER(Trainer), c_int]
libtrainer.bpe_merge_batch.restype = c_int
libtrainer.bpe_train.argtypes = [POINTER(Trainer)]
libtrainer.bpe_train.restype = c_int
libtrainer.bpe_save.argtypes = [POINTER(Trainer), c_char_p, c_char_p]
libtrainer.bpe_save.restype = None