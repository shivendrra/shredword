#ifndef __BPE__H__
#define __BPE__H__

#include <stdint.h>
#include "heap.h"
#include "hash.h"

#define  MIN_HEAP_SIZE  2048

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
  int num_threads;    // threads for initial counting
} BPEConfig;

typedef struct BpeTrainer BpeTrainer;   // for handling the training part
static BIMap bigram_map;  // global bigram→Info map for lazy invalidation

extern "C" {
  BpeTrainer* bpe_trainer_create(const BPEConfig* config);
  void bpe_trainer_destroy(BpeTrainer* trainer);

  int bpe_loadCorpus(BpeTrainer* trainer, const char* input_path);
  uint32_t bpe_get_current_version(const PairKey key);
  void bpe_initialize(BpeTrainer* trainer);
  void bpe_count_bigrams(BpeTrainer* trainer);
  int bpe_merge(BpeTrainer* trainer);
  int bpe_train(BpeTrainer* trainer);
  int bpe_save(const BpeTrainer* trainer, const char* model_path, const char* vocab_path);
}


#endif  //!__BPE__H__