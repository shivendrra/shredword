#ifndef __TRAIN__H__
#define __TRAIN__H__

#include "base.h"
#include "main.h"

extern "C" {
  void train(Shred* tokenizer, const char* text, int vocab_size);
  void dynamic_train_bpe(Shred* tokenizer, const char* text, int vocab_size, int min_freq);
}

#endif  //!__TRAIN__H__