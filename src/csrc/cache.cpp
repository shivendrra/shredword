#include "cache.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef _WIN32
  #include <windows.h>
#else
  #include <unistd.h>
#endif

#define DEFAULT_MAX_THREADS 8  // default fallback in case detection fails

int get_max_threads() {
  int num_threads = 1;  // defaulted to 1 to prevent issues

  // detect the number of CPU threads
  #ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    num_threads = sysinfo.dwNumberOfProcessors;
  #else
    num_threads = sysconf(_SC_NPROCESSORS_ONLN);  // get the number of online CPU cores
  #endif

  int max_threads = (num_threads > 2) ? (num_threads - 2) : 1;  // ensure a minimum of 1 thread is always used
  printf("Detected CPU threads: %d, using max threads: %d\n", num_threads, max_threads);
  return max_threads;
}
int MAX_THREADS = DEFAULT_MAX_THREADS;  // global variable for thread count
void initialize_threads() {
  MAX_THREADS = get_max_threads();  // setting MAX_THREADS dynamically
}

// initialize token length cache from the tokenizer’s vocabulary
void initialize_token_cache(const Shred* tokenizer) {
  for (int i = 0; i < VOCAB_SIZE + MAX_MERGES; i++) {
    if (tokenizer->base.vocab[i].value) {
      token_length_cache[i] = strlen(tokenizer->base.vocab[i].value);
    } else {
      token_length_cache[i] = 0;
    }
  }
}

/* ====================================================
   parallel Training Worker: train_worker
   ====================================================
   This worker computes partial frequency counts for
   adjacent token pairs over a segment of the ids array.
   Each thread writes its results into a local array
*/

// train_worker: computes frequency counts for token pairs in the range [start, end).
// Note: It processes indices up to (end - 1) since pairs are formed with i and i+1
void* train_worker(void* args) {
  TrainThreadArgs* targs = (TrainThreadArgs*) args;
  const int* ids = targs->ids;
  int start = targs->start;
  int end = targs->end;
  
  // looping through the segment (stoping at end-1 to avoid out-of-bound access)
  for (int i = start; i < end - 1; i++) {
    int a = ids[i];
    int b = ids[i + 1];
    int found = 0;
    // looking for the pair (a, b) in the local_stats array.
    for (int j = 0; j < *(targs->local_count); j++) {
      if (targs->local_stats[j].idx1 == a && targs->local_stats[j].idx2 == b) {
        targs->local_stats[j].freq++;
        found = 1;
        break;
      }
    }
    // if not found and we have capacity, add a new entry
    if (!found) {
      int count = *(targs->local_count);
      if (count < INITIAL_CACHE_SIZE) {
        targs->local_stats[count].idx1 = a;
        targs->local_stats[count].idx2 = b;
        targs->local_stats[count].freq = 1;
        (*(targs->local_count))++;
      }
      // else, if capacity exceeded, you may decide to ignore further pairs
    }
  }
  return NULL;
}

// merge partial stats from multiple training workers into a single global stats array
// global_stats: array to hold the merged stats (size at least INITIAL_CACHE_SIZE)
// global_count: pointer to an int where the total number of distinct pairs will be stored
// partial_stats: array of pointers to each thread’s local_stats arrays
// counts: array holding each thread’s local_count value
// num_threads: number of training threads used
void merge_train_stats(PairStat* global_stats, int* global_count, PairStat** partial_stats, int* counts, int num_threads) {
  *global_count = 0;
  for (int t = 0; t < num_threads; t++) {
    for (int i = 0; i < counts[t]; i++) {
      int a = partial_stats[t][i].idx1;
      int b = partial_stats[t][i].idx2;
      int freq = partial_stats[t][i].freq;
      int found = 0;
      // Try to find an existing entry in global_stats.
      for (int j = 0; j < *global_count; j++) {
        if (global_stats[j].idx1 == a && global_stats[j].idx2 == b) {
          global_stats[j].freq += freq;
          found = 1;
          break;
        }
      }
      // If not found, add a new entry.
      if (!found) {
        if (*global_count < INITIAL_CACHE_SIZE) {
          global_stats[*global_count].idx1 = a;
          global_stats[*global_count].idx2 = b;
          global_stats[*global_count].freq = freq;
          (*global_count)++;
        }
      }
    }
  }
}

// worker function for decoding in parallel
void* decode_worker(void* args) {
  ThreadArgs* thread_args = (ThreadArgs*) args;
  const Shred* tokenizer = thread_args->tokenizer;
  const int* ids = thread_args->ids;
  int start = thread_args->start;
  int end = thread_args->end;
  
  size_t local_size = 0;
  // calculate local output size based on token lengths
  for (int i = start; i < end; i++) {
    local_size += token_length_cache[ids[i]];
  }
  
  char* local_output = (char*) malloc(local_size + 1);
  if (!local_output) {
    fprintf(stderr, "Error: Memory allocation failed for local output.\n");
    exit(EXIT_FAILURE);
  }
  
  char* current_pos = local_output;
  for (int i = start; i < end; i++) {
    const char* token = tokenizer->base.vocab[ids[i]].value;
    size_t token_len = token_length_cache[ids[i]];
    memcpy(current_pos, token, token_len);
    current_pos += token_len;
  }
  *current_pos = '\0';
  
  thread_args->output_str = local_output;
  thread_args->output_int = NULL;
  *thread_args->output_size = local_size;
  
  return NULL;
}

// worker function for encoding in parallel
void* encode_worker(void* args) {
  ThreadArgs* thread_args = (ThreadArgs*) args;
  const Shred* tokenizer = thread_args->tokenizer;
  const int* ids = thread_args->ids;
  int start = thread_args->start;
  int end = thread_args->end;
  
  size_t local_size = end - start;
  int* local_output = (int*) malloc(local_size * sizeof(int));
  if (!local_output) {
    fprintf(stderr, "Error: Memory allocation for local_output failed.\n");
    exit(EXIT_FAILURE);
  }
  
  size_t new_output_size = 0;
  for (size_t i = start; i < end; i++) {
    int current_id = ids[i];
    if (i < end - 1) {
      for (int j = 0; j < tokenizer->base.merge_count; j++) {
        Pair max_pair = tokenizer->base.merges[j].pair;
        if (current_id == max_pair.idx1 && ids[i + 1] == max_pair.idx2) {
          local_output[new_output_size++] = VOCAB_SIZE + j;
          i++; // skip the next ID as it is part of the merge
          current_id = -1; // mark as merged
          break;
        }
      }
    }
    if (current_id != -1) {
      local_output[new_output_size++] = current_id;
    }
  }
  
  thread_args->output_int = local_output;
  thread_args->output_str = NULL;
  *thread_args->output_size = new_output_size;
  
  return NULL;
}