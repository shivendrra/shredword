/** 
  @brief unigram.h header file for the unigram trainer code logic.
  * each new vocab is determined & merged based EM using unigram approach
      with help of hashing & heaps for faster merges.
  * main entry point file code for Unigram-trainer related codebase.
  * compile it as:
    *- '.so': g++ -shared -fPIC -o libtrainer.so unigram/unigram.cpp unigram/normalize.cpp inc/hash.cpp inc/heap.cpp
    *- '.dll': g++ -shared -o libtrainer.dll unigram/unigram.cpp unigram/normalize.cpp inc/hash.cpp inc/heap.cpp
*/

#ifndef __UNIGRAM__H__
#define __UNIGRAM__H__

#include <stdint.h>
#include <stddef.h>
#include "normalize.h"

#define MAX_VOCAB_SIZE 100000
#define MAX_WORD_LEN 128

typedef struct {
  char* subword;
  double score;
  int freq;
} UnigramEntry;

typedef struct {
  UnigramEntry *entries;
  size_t size;
  size_t capacity;
} UnigramModel;

extern "C" {
  UnigramModel* create_unigram_model(size_t capacity);
  void free_unigram_model(UnigramModel* model);
  void initialize_from_vocab(UnigramModel* model, VocabTable* table);
  void dump_unigram_model(UnigramModel* model);

  void run_em_training(UnigramModel* model, const char* const* corpus_lines, size_t num_lines, size_t max_steps);
}

#endif  //!__UNIGRAM__H__