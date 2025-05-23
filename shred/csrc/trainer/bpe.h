#ifndef __BPE__H__
#define __BPE__H__

#include <stdint.h>
#include "heap.h"
#include "hash.h"

#define  MIN_HEAP_SIZE  4096
#define  INITIAL_VOCAB_SIZE  256  // UTF-8 base chars from 0 -> 255

typedef struct Symbol {
  int32_t id;   // cuttent token's id
  struct Symbol* prev;  // previous symbol in word
  struct Symbol* next;  // next symbol in word
} Symbol;

typedef struct wordPos {
  size_t word_index;  // index to word list
  Symbol* pos;    // pointer to first bigram symbol
} wordPos;

typedef struct Corpus {
  Symbol** words;   // array of head pointers of each word
  uint64_t* word_counts;  // corresponding freq
  size_t vocab_size;    // no of unique words
} Corpus;

typedef struct {
  size_t target_vocab;    // desired vocab_size
  int32_t unk_id;   // ID to use for unknown tokens
  double  character_coverage;   // e.g. 0.995
  uint64_t min_pair_freq;       // e.g. 5
} BPEConfig;

typedef struct BpeTrainer {
  BPEConfig config;   // user-settings
  MaxHeap heap;   // bigram merge items
  Corpus corpus;    // word-> countmap & symbol chains
  BIMap bigram_map;  // global bigram -> Info map
  size_t next_token_id; // next unused subword ID (starts at initial vocab size)
  size_t initial_vocab_size;
  size_t num_merges;    // how many merges have been applied
  PairKey* merge_ops;     // array of length num_merges
  char** token_strs;    // maps token ID -> UTF‑8 string
  uint64_t* token_freqs;   // maps token ID -> frequency
} BpeTrainer;   // for handling the training part

static BIMap bigram_map;  // global bigram→Info map for lazy invalidation

extern "C" {
  BpeTrainer* bpe_trainer_create(const BPEConfig* config);
  void bpe_trainer_destroy(BpeTrainer* trainer);

  int bpe_loadCorpus(BpeTrainer* trainer, const char* input_path);
  uint32_t bpe_get_current_version(const PairKey key);
  uint64_t recompute_freq(PairKey key, Info* info, BpeTrainer* trainer);
  void bpe_initialize(BpeTrainer* trainer);
  void bpe_count_bigrams(BpeTrainer* trainer);
  int bpe_merge(BpeTrainer* trainer);
  int bpe_train(BpeTrainer* trainer);
  void bpe_save(const BpeTrainer* trainer, const char* model_path, const char* vocab_path);
}


#endif  //!__BPE__H__