// train.h
#ifndef __TRAIN__H__
#define __TRAIN__H__

#include "base.h"

extern "C" {

  // void train_vocab(const char* train_file, const char* vocab_file, int vocab_limit);
  void train_vocab(const char* train_file, const char* vocab_file, int merge_steps);
}

#endif  //!__TRAIN__H__