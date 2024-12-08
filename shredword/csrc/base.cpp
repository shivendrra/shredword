#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <assert.h>
#include "base.h"

Token** vocab = NULL;
size_t vocab_size = 0;
pairs* merges = NULL;
size_t n_merges = 0;

void init_vocab() {
  vocab_size = BASE_VOCAB_SIZE;
  vocab = (Token**)(vocab_size * sizeof(Token*));
  if (!vocab) {
    fprintf(stderr, "Memory allocation for base vocab failed");
    exit(1);
  }
  for (size_t i = 0; i < vocab_size; i++) {
    vocab[i] = (Token*)malloc(sizeof(Token));
    if (!vocab) {
      fprintf(stderr, "Memory allocation for vocab[%zu] failed\n", i);
      exit(1);
    }
    vocab[i][0] = (Token)i;
  }
}

void free_vocab() {
  if (vocab) {
    for (size_t i = 0; i < vocab_size; i++) {
      free(vocab[i]);
    }
    free(vocab);
    vocab = NULL;
  }
}