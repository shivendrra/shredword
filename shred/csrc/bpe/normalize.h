#ifndef __NORMALIZE_H__
#define __NORMALIZE_H__

#include <stddef.h>
#include <stdint.h>

#define MAX_LINE 8192
#define U_2581 "\xE2\x96\x81"
#define MAX_VOCAB_SIZE 100000
#define MAX_SUBWORD_LEN 16
#define LOAD_FACTOR_THRESHOLD 0.75
#define INITIAL_CAPACITY 1024

typedef struct VocabEntry {
  char* str;
  size_t count;
  size_t str_len;  // Cache string length
} VocabEntry;

typedef struct VocabTable {
  VocabEntry* entries;
  size_t capacity;
  size_t size;
  size_t threshold;  // Resize threshold
} VocabTable;

extern "C" {
  // replaces all spaces with U+2581 (▁) and lowercases the string
  // returns the length of the output string, or -1 on error
  int normalize_line(const char* input, char* output, size_t output_size);
  VocabTable* create_vocab(size_t initial_capacity);    // Fixed parameter type
  void free_vocab(VocabTable* table); // frees vocabulary table and all its entries
  int add_subwords(VocabTable* table, const char* line, size_t max_subword_len);  // Fixed return type and parameter
  void dump_vocab(VocabTable* table); // dumps vocabulary to stdout
}

// Inline utility functions
static inline size_t vocab_size(const VocabTable* table) {
  return table ? table->size : 0;
}

static inline int vocab_is_empty(const VocabTable* table) {
  return !table || table->size == 0;
}

static inline int vocab_is_full(const VocabTable* table) {
  return table && table->size >= MAX_VOCAB_SIZE;
}

#endif