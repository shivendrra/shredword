/*
  cache.h
  * uses Static Precomputation caching technique
  * caching implementation in a file for the encoding/decoding functions in ``main.cpp``
  * also uses ThreadPools to execute parallel computation
  
  * implemented LRU caching for training the merges & vocabs
    - has some memory related issues & bugs
    - crashes after some iters
*/

#ifndef __CACHE__H__
#define __CACHE__H__

#include <pthread.h>
#include "base.h"
#include "main.h"

extern int MAX_THREADS;  // declared globally
void initialize_threads();  // function to set MAX_THREADS dynamically
#define INITIAL_CACHE_SIZE 2048  // Max size of LRU cache
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

// for training worker thread arguments
typedef struct {
  int idx1;
  int idx2;
  int freq;
} PairStat;   // this is used in the training worker to count occurrences of adjacent token pairs

typedef struct {
  const int* ids;
  int start;
  int end;
  PairStat* local_stats;  // array to store partial pair stats (size at least INITIAL_CACHE_SIZE)
  int* local_count;       // pointer to an int storing the number of distinct pairs found
} TrainThreadArgs;

extern "C" {
  void initialize_token_cache(const Shred* tokenizer);
  void* decode_worker(void* args);
  void* encode_worker(void* args);
  void* train_worker(void* args);
  void merge_train_stats(PairStat* global_stats, int* global_count, PairStat** partial_stats, int* counts, int num_threads);
  unsigned int hash(const char* key);
}

#endif