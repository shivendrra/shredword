/*
  cache.h
  * uses Static Precomputation caching technique (not LRU yet)
  * caching implementation in a file for the encoding/decoding functions in ``main.cpp``
  * also uses ThreadPools to execute parallel computation
  * it still has showed a significant speed boost in both:
    - encoding(x18-19 times)
    - decoding(x40 times)
*/

#ifndef __CACHE__H__
#define __CACHE__H__

#include <pthread.h>
#include "base.h"
#include "main.h"

#define MAX_THREADS 6
static size_t token_length_cache[VOCAB_SIZE + MAX_MERGES];

typedef struct {
  const Shred* tokenizer;
  const int* ids;
  int start;
  int end;
  char* output_str;
  int* output_int;
  size_t* output_size;
} ThreadArgs;

extern "C" {
  void initialize_token_cache(const Shred* tokenizer);
  void* decode_worker(void* args);
  void* encode_worker(void* args);
}

#endif