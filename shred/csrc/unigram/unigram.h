/** 
  @brief unigram.h header file for the unigram trainer code logic.
  * each new vocab is determined & merged based EM using unigram approach
      with help of hashing & heaps for faster merges.
  * main entry point file code for Unigram-trainer related codebase.
  * compile it as:
    *- '.so': g++ -shared -fPIC -o libtrainer.so unigram/unigram.cpp unigram/normalize.cpp unigrma/hash.cpp trie.cpp
    *- '.dll': g++ -shared -o libtrainer.dll unigram/unigram.cpp unigram/normalize.cpp unigram/hash.cpp trie.cpp
*/

#ifndef __UNIGRAM__H__
#define __UNIGRAM__H__

#include <stdint.h>
#include <stddef.h>
#include "normalize.h"
#include "hash.h"

#define MAX_VOCAB_SIZE 100000
#define MAX_WORD_LEN 128
#define MAX_LINE 8192
#define HASH_TABLE_SIZE 131072  // 2^17 for better hash distribution

typedef struct TokenEntry {
  char* token;
  int index;
  struct TokenEntry* next;
} TokenEntry;

typedef struct TokenMap {
  TokenEntry** buckets;
  size_t nbuckets;
} TokenMap;

typedef struct UnigramEntry {
  char* subword;
  double score;
  int freq;
  uint32_t hash;
  uint16_t len;
} UnigramEntry;

typedef struct UnigramModel {
  UnigramEntry* entries;
  size_t size;
  size_t capacity;
  TokenMap token_map;
  int* hash_table;
  int* next_in_bucket;
} UnigramModel;

typedef struct ViterbiCell {
  double score;
  int prev;
  int token_index;
} ViterbiCell;

typedef struct {
  UnigramModel* model;
  size_t* count;
  uint64_t* total_freq;
} TransferContext;

// Fast hash function for strings
static inline uint32_t fast_hash(const char* str, size_t len) {
  uint32_t hash = 5381;
  for (size_t i = 0; i < len; i++) {
    hash = ((hash << 5) + hash) + (unsigned char)str[i];
  }
  return hash;
}

extern "C" {
  // core model functions
  UnigramModel* create_unigram_model(size_t capacity);
  void free_unigram_model(UnigramModel* model);
  void initialize_from_vocab(UnigramModel* model, VocabBuilder* builder);
  // void initialize_from_vocab(UnigramModel* model, VocabTable* vocab);
  
  // training and tokenization
  void run_em_training(UnigramModel* model, const char* const* corpus_lines, size_t num_lines, size_t max_steps);
  char** viterbi_tokenize(const UnigramModel* model, const char* line, size_t* out_token_count);
  int* encode_to_ids(const UnigramModel* model, const char* line, size_t* out_count);
  
  // model management
  void prune_unigram_model(UnigramModel* model, size_t target_vocab_size);
  void save_unigram_model(const UnigramModel* model, const char* filepath);
  void dump_unigram_model(UnigramModel* model);
  
  // lookup functions
  int get_token_id(const UnigramModel* model, const char* token);
  int fast_token_lookup(const UnigramModel* model, const char* token, size_t len);
  
  // internal functions
  void rebuild_hash_table(UnigramModel* model);
  void rebuild_token_map(UnigramModel* model);
}

#endif  //!__UNIGRAM__H__