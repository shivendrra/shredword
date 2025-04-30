// train.h
#ifndef __TRAIN__H__
#define __TRAIN__H__

#include "base.h"

typedef struct TokenPairs {
  int first, second;
} TokenPairs;

extern "C" {
  void bpe_learn(const char* train_file, int merge_steps, TrieNode* root);
  void train_vocab(const char* train_file, const char* vocab_file, int merge_steps);
}

#endif  //!__TRAIN__H__