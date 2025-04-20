#ifndef __TRAIN__H__
#define __TRAIN__H__

#include "base.h"
#include "main.h"

extern "C" {
  void train(Shred* tokenizer, const char* text, int vocab_size);
  void dynamic_train_bpe(Shred* tokenizer, const char* text, int vocab_size, int min_freq);

  // New optimized training function that implements techniques inspired by SentencePiece:
  // - Multi-threaded frequency counting with filtering of low-frequency pairs
  // - Reduced memory reallocation overhead in the merge step
  // - Early stopping when no candidate pair meets the minimum frequency threshold
  void optimized_train(Shred* tokenizer, const char* text, int vocab_size, int min_freq);
}

#endif  //!__TRAIN__H__