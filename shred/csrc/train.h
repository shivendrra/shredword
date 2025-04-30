// train.h
#ifndef __TRAIN__H__
#define __TRAIN__H__

#include "base.h"

typedef struct TokenPairs {
  int first, second;
} TokenPairs;

typedef struct VocabEntry {
  int idx;
  char* value;
} VocabEntry;

typedef struct MergeEntry {
  TokenPairs pair;
  int idx;
} MergeEntry;

extern "C" {
  void train_vocab_bpe(const char* train_file, const char* vocab_file, int merge_steps);
  void train_vocab(const char* train_file, const char* vocab_file, int vocab_limit);
}

#endif  //!__TRAIN__H__