/** 
  @brief training.h
  EM training and tokenization functions
  * Contains training algorithms and tokenization functionality
  * Split from the original unigram.h for better organization
*/

#ifndef __UNIGRAM_TRAINING__H__
#define __UNIGRAM_TRAINING__H__

#include <stdint.h>
#include <stddef.h>

typedef struct ViterbiCell {
  double score;
  int prev;
  int token_index;
} ViterbiCell;

typedef struct UnigramModel UnigramModel;

extern "C" {
  // Training functions
  void run_em_training(UnigramModel* model, const char* const* corpus_lines, size_t num_lines, size_t max_steps);

  // Tokenization functions
  char** viterbi_tokenize(const UnigramModel* model, const char* line, size_t* out_token_count);
  int* encode_to_ids(const UnigramModel* model, const char* line, size_t* out_count);
}

#endif