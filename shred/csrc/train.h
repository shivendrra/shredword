// train.h
#ifndef __TRAIN__H__
#define __TRAIN__H__

#include "base.h"
#include "threads.h"
#include "heap.h"

extern "C" {
  void train_vocab_bpe(const char* train_file, const char* vocab_file, int merge_steps);
  void train_vocab_naive(const char* train_file, const char* vocab_file, int vocab_limit);
  void train_bpe_fast(const char* train_file, const char* vocab_file, int merge_steps, int num_threads);

  int get_symbol_id(const char* sym);
}

#endif  //!__TRAIN__H__