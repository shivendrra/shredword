import os, ctypes
from ctypes import *

# trying multiple possible library names and paths
possible_paths = [
  os.path.join(os.path.dirname(__file__), "../build/lib.dll"),
  os.path.join(os.path.dirname(__file__), "../build/libtrainer.dll"), 
  os.path.join(os.path.dirname(__file__), "../libtrainer.dll"),
  os.path.join(os.path.dirname(__file__), "libtrainer.dll"),
  "libtrainer.dll"
]

libtrainer = None
for path in possible_paths:
  if os.path.exists(path):
    try:
      libtrainer = ctypes.CDLL(path)
      # print(f"Successfully loaded library from: {path}")
      break
    except Exception as e:
      print(f"Failed to load {path}: {e}")
      continue

if libtrainer is None:
  error_msg = f"""
Could not find or load the BPE trainer shared library.

Searched in the following locations:
{chr(10).join(f"  - {path}" for path in possible_paths)}

To fix this issue:

1. First, make sure you have a C++ compiler installed:
   - Windows: Install Visual Studio with C++ support or MinGW-w64
   - Linux: Install g++ (sudo apt install g++ or equivalent)
   - macOS: Install Xcode command line tools (xcode-select --install)

2. Create the missing header files. You need these files in an 'inc' directory:
   - inc/heap.h and inc/heap.cpp
   - inc/hash.h and inc/hash.cpp  
   - bpe/histogram.h and bpe/histogram.cpp

3. Compile the shared library using one of these commands:

   For Windows (MinGW):
   g++ -shared -o libtrainer.dll bpe/bpe.cpp bpe/histogram.cpp inc/hash.cpp inc/heap.cpp

   For Windows (Visual Studio):
   cl /LD bpe/bpe.cpp bpe/histogram.cpp inc/hash.cpp inc/heap.cpp /Fe:libtrainer.dll

   For Linux:
   g++ -shared -fPIC -o libtrainer.so bpe/bpe.cpp bpe/histogram.cpp inc/hash.cpp inc/heap.cpp

   For macOS:
   g++ -shared -fPIC -o libtrainer.dylib bpe/bpe.cpp bpe/histogram.cpp inc/hash.cpp inc/heap.cpp

4. Place the compiled library in one of the searched directories above.
"""
  raise FileNotFoundError(error_msg)

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

class BPETrainer:
  def __init__(self, target_vocab_size=8192, unk_id=0, character_coverage=0.995, min_pair_freq=2000):
    self.config = BPEConfig(
      target_vocab_size=target_vocab_size,
      unk_id=unk_id,
      character_coverage=character_coverage,
      min_pair_freq=min_pair_freq
    )
    self.trainer = libtrainer.create_trainer(ctypes.byref(self.config))
    if not self.trainer:
      raise RuntimeError("Failed to create BPE trainer")

  def load_corpus(self, path: str):
    result = libtrainer.bpe_load_corpus(self.trainer, path.encode('utf-8'))
    if result != 0:
      raise IOError(f"Failed to load corpus from {path}")

  def train(self):
    merges = libtrainer.bpe_train(self.trainer)
    if merges < 0:
      raise RuntimeError("Training failed")
    print(f"Training completed: {merges} merges performed.")

  def save(self, model_path: str, vocab_path: str):
    libtrainer.bpe_save(self.trainer, model_path.encode('utf-8'), vocab_path.encode('utf-8'))
    print(f"Model saved to: {model_path}")
    print(f"Vocabulary saved to: {vocab_path}")

  def destroy(self):
    if self.trainer:
      libtrainer.bpe_trainer_destroy(self.trainer)
      self.trainer = None

  def __del__(self):
    self.destroy()