/*
  base.h
  - base class for basic BPE functions & logics
  - all classes to be built on top of this code, since it has necessary blocks
  - to be compiled with ``tokenizer.cpp`` containing the main logic (no regex, no caching)
  - compile it as:
    -- '.so': g++ -shared -fPIC -o libtensor.so tokenizer.cpp base.cpp / for linux
    -- '.dll': g++ -shared -o libtensor.dll tokenizer.cpp base.cpp / for windows
*/

#ifndef __BASE__H__
#define __BASE__H__

#include <stdbool.h>  // for bool type
#include <stdint.h>   // for uint32_t & others
#include <stddef.h>   // for size_t

// storeing the token pairs
typedef struct pairs {
  int *first, *second;
} pairs;

// storing the token pair & frequency count
typedef struct paircount {
  int first, second;
  size_t count; // occurence count of pair
} paircount;

typedef uint32_t Token;

extern Token **vocab;         // 2-d array representation of vocab
extern size_t vocab_size;     // current vocab size
extern pairs* merges;         // array of token pairs for merge
extern size_t n_merges;     // no of merges performed

extern "C" {
  void init_vocab();
  void free_vocab();
  void build_vocab(pairs* merges, size_t n_merges);
  void replace_control_characters(char* str);
  char* render_token(const Token* token);
  void get_stats(const Token* ids, size_t len, paircount** stats, size_t* stats_size, size_t* capacity);
  void merge(const Token* ids, size_t len, Token* merged_ids, size_t* merged_len, pairs pair, Token new_token);
  void encode(const char *text, Token** ids, size_t len);
  char* decode(const Token* ids, size_t len);
  void train(const char* text, size_t target_vocab_size, bool verbose);
  const char* get_token_from_vocab(size_t index);
}

#endif