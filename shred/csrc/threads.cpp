#include <stdlib.h>
#include <string.h>
#include "threads.h"
#include "train.h"
#include "base.h"
#include "heap.h"

void* thread_count_pairs(void* _arg) {
  ThreadArg* arg = (ThreadArg*)_arg;
  int T = arg->num_threads, tid = arg->thread_id;
  arg->local_map = kh_init(pair_int);

  for (int i = tid; i < arg->corpus_size; i += T) {
    for (int j = 0; j + 1 < arg->seq_lens[i]; j++) {
      int A = get_symbol_id(arg->seq_syms[i][j]);
      int B = get_symbol_id(arg->seq_syms[i][j+1]);
      uint64_t p = pack_pair(A,B);
      char key[32];
      int ret;
      memcpy(key, &p, sizeof(p));

      // manually duplicating 8-byte pair key
      char* key_dup = (char*)malloc(sizeof(uint64_t));
      memcpy(key_dup, &p, sizeof(uint64_t));
      khiter_t k = kh_put(pair_int, arg->local_map, key_dup, &ret);
      kh_val(arg->local_map, k)++;
    }
  }
  return NULL;
}