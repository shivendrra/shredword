/**
  @file histogram.h
  @brief Histogram and symbol chain utilities for BPE training.
  * This module handles preprocessing utilities required during corpus loading,
   including:
  * - Building a character-level histogram from the corpus to estimate which characters
     should be retained based on a configured character coverage threshold.
  * - Generating a symbol chain (linked list of Symbol structs) for each word, using either
     the original character ID or a fallback UNK token for rare characters.
  * - Sorting characters by frequency to determine inclusion into the vocabulary.
  * - Providing helper callbacks for StrMap iteration (word frequency, character histogram, etc.).

  * This file decouples symbol chain construction and histogram logic from the main trainer module,
  * making it easier to maintain and reuse for different subword algorithms.
*/

#ifndef __HISTOGRAM__H__
#define __HISTOGRAM__H__

#include "hash.h"

typedef struct Trainer Trainer;   // forward declaration
typedef struct Symbol Symbol;   // forward declaration

struct load_ctx {
  Trainer* trainer;
  size_t idx;
};

typedef struct {
  Trainer* trainer;
  size_t* idx;
  bool* keep_char;
} BuildCtx;

typedef struct {
  char c;
  uint64_t count;
} CharCount;

typedef struct {
  CharCount *arr;
  size_t idx;
} CharCountCtx;

extern "C" {
  void build_symbol_cb(const char* w, uint64_t count, void* u);
  void char_hist(const char* word, uint64_t wcount, void* u);
  void collect_char(const char* kc, uint64_t vc, void* u);
  int charcount_cmp(const void *a, const void *b);
  void load_entry(const char* key, uint64_t val, void* user);
}

#endif  //!__HISTOGRAM__H__