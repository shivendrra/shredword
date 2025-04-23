#ifndef __TRAIN__H__
#define __TRAIN__H__

#include "base.h"
#include "main.h"

extern "C" {
  void train_vocab(const char* train_file, const char* vocab_file);
}

#endif  //!__TRAIN__H__