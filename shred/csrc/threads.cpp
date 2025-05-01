#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "threads.h"
#include "train.h"
#include "base.h"
#include "heap.h"

#ifdef _WIN32
  #include <windows.h>
#else
  #include <unistd.h>
#endif

#define DEFAULT_MAX_THREADS 8

int get_max_threads() {
  int num_threads = 1;
#ifdef _WIN32
  SYSTEM_INFO sysinfo;
  GetSystemInfo(&sysinfo);
  num_threads = sysinfo.dwNumberOfProcessors;
#else
  num_threads = sysconf(_SC_NPROCESSORS_ONLN);
#endif
  int max_threads = (num_threads > 2) ? (num_threads - 2) : 1;
  return max_threads;
}
int MAX_THREADS = DEFAULT_MAX_THREADS;
void initialize_threads() {
  MAX_THREADS = get_max_threads();
  printf("Detected CPU threads: %d, using max threads: %d\n", MAX_THREADS + 2, MAX_THREADS);
}

void* thread_count_pairs(void* _arg) {
  ThreadArg* arg = (ThreadArg*)_arg;
  int T = get_max_threads(), tid = arg->thread_id;
  arg->local_map = kh_init(pair_int);

  for (int i = tid; i < arg->corpus_size; i += T) {
    for (int j = 0; j + 1 < arg->seq_lens[i]; j++) {
      int A = get_symbol_id(arg->seq_syms[i][j]);
      int B = get_symbol_id(arg->seq_syms[i][j+1]);
      uint64_t p = pack_pair(A,B);
      char key[32];
      int ret;
      memcpy(key, &p, sizeof(p));

      // manually duplicating 8-byte pair key + null-terminator for safe C-string handling
      char* key_dup = (char*)malloc(sizeof(uint64_t) + 1);
      if (!key_dup) {
        fprintf(stderr, "[ERROR] malloc failed in thread %d\n", tid);
        exit(EXIT_FAILURE);
      }
      memcpy(key_dup, &p, sizeof(uint64_t));
      key_dup[sizeof(uint64_t)] = '\0';

      khiter_t k = kh_put(pair_int, arg->local_map, key_dup, &ret);
      if (ret < 0) {
        fprintf(stderr, "[ERROR] kh_put failed in thread %d\n", tid);
        exit(EXIT_FAILURE);
      }
      kh_val(arg->local_map, k)++;
    }
  }
  return NULL;
}