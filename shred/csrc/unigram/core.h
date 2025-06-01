/** 
  @brief core.h
  Core data structures and model management functions
  * Contains the main data structures and basic model operations
  * Integrates with existing normalize.h and hash.h modules
*/

#ifndef __CORE__H__
#define __CORE__H__

#include <stdint.h>
#include <stddef.h>
#include "normalize.h"
#include "hash.h"
#include "tokenmap.h"

#define MAX_VOCAB_SIZE 100000
#define MAX_WORD_LEN 128
#define MAX_LINE 8192
#define HASH_TABLE_SIZE 131072  // 2^17 for better hash distribution

typedef struct UnigramEntry {
  char* subword;
  double score;
  uint64_t freq; // using uint64_t to match HashVocabEntry
  uint32_t hash;
  uint16_t len;
} UnigramEntry;

typedef struct UnigramModel {
  UnigramEntry* entries;
  size_t size;
  size_t capacity;
  TokenMap token_map;  // Now properly defined
  int* hash_table;
  int* next_in_bucket;
} UnigramModel;

// using the hash function from normalize.cpp (FNV-1a)
static inline uint32_t model_hash(const char* str, size_t len) {
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < len; i++) {
    hash ^= (unsigned char)str[i];
    hash *= 16777619u;
  }
  return hash;
}

extern "C" {
  // Core model creation and destruction
  UnigramModel* create_unigram_model(size_t capacity);
  void free_unigram_model(UnigramModel* model);

  // Model initialization from hash-based vocabulary builder
  void initialize_from_hash_vocab(UnigramModel* model, HashVocabEntry* entries, size_t count, StringInterner* interner);
  
  // Model initialization from traditional VocabTable
  void initialize_from_vocab_table(UnigramModel* model, VocabTable* vocab_table);

  // Model management
  void prune_unigram_model(UnigramModel* model, size_t target_vocab_size);
  void save_unigram_model(const UnigramModel* model, const char* filepath);
  void dump_unigram_model(UnigramModel* model);

  // Hash table and token map management
  void rebuild_hash_table(UnigramModel* model);
  void rebuild_token_map(UnigramModel* model);

  // Token lookup functions
  int get_token_id(const UnigramModel* model, const char* token);
  int fast_token_lookup(const UnigramModel* model, const char* token, size_t len);

  // Convenience function to build vocabulary and create model
  UnigramModel* create_model_from_text(const char** lines, size_t num_lines, size_t max_subword_len, size_t min_frequency, size_t target_vocab_size);
}

#endif  //!__CORE__H__