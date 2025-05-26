/** 
  @brief bpe.h header file for the bpr trainer code logic.
  * each new vocab is determined & merged based on traditional bpe merging
      with help of hashing & heaps for faster merges.
  * main entry point file code for BPE-trainer related codebase.
  * compile it as:
    *- '.so': g++ -shared -fPIC -o libtrainer.so bpe/bpe.cpp bpe/histogram.cpp inc/hash.cpp inc/heap.cpp
    *- '.dll': g++ -shared -o libtrainer.dll bpe/bpe.cpp bpe/histogram.cpp inc/hash.cpp inc/heap.cpp
*/

#ifndef __BPE__H__
#define __BPE__H__

#include <stdint.h>
#include "../inc/heap.h"
#include "../inc/hash.h"

#define  MIN_HEAP_SIZE  4096
#define  INITIAL_VOCAB_SIZE  256  // UTF-8 base chars from 0 -> 255
#define  INITIAL_STR_BUFFER  4096  // no of characters to be loaded
#define  MAX_OCCS_PER_MERGE  50000
#define  MIN_PAIR_FREQ  2000

typedef struct Symbol {
  int32_t id;  // current token's int value
  struct Symbol* prev;  // previous token
  struct Symbol* next;  // next token
  bool deleted; // check deleted or not?
} Symbol;

typedef struct WordPos {
  size_t word_index;  // index to word list
  Symbol* pos;  // pointer to first bigram symbol
} WordPos;

typedef struct Corpus {
  Symbol** words; // index to a word list
  uint64_t* word_counts;  // corresponding freq
  size_t vocab_size;  // no of unique word in train corpus
} Corpus;

typedef struct BPEConfig {
  size_t target_vocab_size;
  int32_t unk_id;   // for unknown tokens
  float character_coverage;   // 0.995 -> 99.5%
  uint64_t min_pair_freq;   // eg: 400
} BPEConfig;

typedef struct Trainer {
  BPEConfig config;
  MaxHeap heap;
  Corpus corpus;
  BIMap bigram_map;
  size_t next_token;    // id for next token
  size_t num_merges;
  PairKey* merge_ops;
  char** token_strs;
  uint64_t* token_freq;
} Trainer;

extern "C" {
  Trainer* create_trainer(const BPEConfig* config);
  void bpe_trainer_destroy(Trainer* trainer);
  int bpe_load_corpus(Trainer* trainer, const char* input_path);

  void bpe_init(Trainer* trainer);
  void bpe_count_bigrams(Trainer* trainer);
  int bpe_merge_batch(Trainer* trainer, int batch_size);
  int bpe_train(Trainer* trainer);
  void bpe_save(const Trainer* trainer, const char* model_path, const char* vocab_path);
}

#endif