#ifndef __THREADS__H__
#define __THREADS__H__

#include <stdint.h>
#include "inc/khash.h"

KHASH_MAP_INIT_STR(str_int, int)  // Map from char* to int
KHASH_MAP_INIT_STR(pair_int, int) // Map from pair string ("A\0B") to int

static khash_t(str_int)* sym2id;
static char** id2sym;
static int sym_capacity;
static int sym_count;

// pairID = (A_id << 32) | B_id
static inline uint64_t pack_pair(int A, int B) {
  return ((uint64_t)A << 32) | (uint32_t)B;
}
static inline void unpack_pair(uint64_t p, int* A, int* B) {
  *A = p >> 32; *B = (int)p;
}

typedef struct {
  int thread_id;
  char* train_file;
  char*** seq_syms;
  int* seq_lens;
  int corpus_size;
  khash_t(pair_int)* local_map;
} ThreadArg;

extern "C" {
  void initialize_threads();  // initializes threads for training
  int get_max_threads();  // returns the max no of threads

  void* thread_count_pairs(void* _arg);  // training stats counting worker
}

#endif