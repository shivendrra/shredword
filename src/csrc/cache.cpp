#include "cache.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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
  printf("Detected CPU threads: %d, using max threads: %d\n", num_threads, max_threads);
  return max_threads;
}
int MAX_THREADS = DEFAULT_MAX_THREADS;
void initialize_threads() {
  MAX_THREADS = get_max_threads();
}

size_t token_length_cache[VOCAB_SIZE + MAX_MERGES] = {0};

void initialize_token_cache(const Shred* tokenizer) {
  int total = VOCAB_SIZE + MAX_MERGES;
  for (int i = 0; i < total; i++) {
    if (tokenizer->base.vocab[i].value)
      token_length_cache[i] = strlen(tokenizer->base.vocab[i].value);
    else
      token_length_cache[i] = 0;
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
  for (int i = start; i < end - 1; i++) {
    int a = ids[i];
    int b = ids[i+1];
    int found = 0;
    for (int j = 0; j < *(targs->local_count); j++) {
      if (targs->local_stats[j].idx1 == a && targs->local_stats[j].idx2 == b) {
        targs->local_stats[j].freq++;
        found = 1;
        break;
      }
    }
    if (!found) {
      int count = *(targs->local_count);
      if (count < INITIAL_CACHE_SIZE) {
        targs->local_stats[count].idx1 = a;
        targs->local_stats[count].idx2 = b;
        targs->local_stats[count].freq = 1;
        (*(targs->local_count))++;
      }
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
      for (int j = 0; j < *global_count; j++) {
        if (global_stats[j].idx1 == a && global_stats[j].idx2 == b) {
          global_stats[j].freq += freq;
          found = 1;
          break;
        }
      }
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

// worker function for decoding in parallel (same as trainign, with a luttle tweaks)
void* decode_worker(void* args) {
  ThreadArgs* thread_args = (ThreadArgs*) args;
  const Shred* tokenizer = thread_args->tokenizer;
  const int* ids = thread_args->ids;
  int start = thread_args->start;
  int end = thread_args->end;
  size_t local_size = 0;
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

// worker function for encoding in parallel (same as training, with a little tweaks)
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
          i++;
          current_id = -1;
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

/* ====================================================
   Hashmap-based Cache Implementation
   ==================================================== */

// Simple helper function to duplicate strings.
static char* strdup_safe(const char* s) {
  if (!s) return NULL;
  size_t len = strlen(s) + 1;
  char* copy = (char*) malloc(len);
  if (copy) memcpy(copy, s, len);
  return copy;
}

// A basic hash function (djb2 variant)
unsigned int hash(const char* key) {
  unsigned int h = 5381;
  while (*key) {
    h = ((h << 5) + h) + (unsigned char)(*key);
    key++;
  }
  return h;
}

LRUCache* create_lru_cache(int capacity) {
  LRUCache* cache = (LRUCache*) malloc(sizeof(LRUCache));
  cache->capacity = capacity;
  cache->size = 0;
  cache->entries = (LRUCacheEntry*) calloc(capacity, sizeof(LRUCacheEntry));
  return cache;
}

void free_lru_cache(LRUCache* cache) {
  if (!cache) return;
  for (int i = 0; i < cache->capacity; i++) {
    if (cache->entries[i].key) {
      free(cache->entries[i].key);
      free(cache->entries[i].value);
    }
  }
  free(cache->entries);
  free(cache);
}

// Hashmap-based get: use open addressing with linear probing
void* lru_cache_get(LRUCache* cache, const char* key, size_t* value_size) {
  unsigned int idx = hash(key) % cache->capacity;
  for (int i = 0; i < cache->capacity; i++) {
    unsigned int pos = (idx + i) % cache->capacity;
    if (cache->entries[pos].key == NULL) {
      // empty slot means key not present
      return NULL;
    }
    if (strcmp(cache->entries[pos].key, key) == 0) {
      if (value_size) *value_size = cache->entries[pos].value_size;
      void* copy = malloc(cache->entries[pos].value_size);
      memcpy(copy, cache->entries[pos].value, cache->entries[pos].value_size);
      return copy;
    }
  }
  return NULL;
}

// Hashmap-based put: use open addressing with linear probing
// If the key exists, update it; otherwise, insert into an empty slot
// If cache is full, evict the entry at the computed position
void lru_cache_put(LRUCache* cache, const char* key, const void* value, size_t value_size) {
  unsigned int idx = hash(key) % cache->capacity;
  for (int i = 0; i < cache->capacity; i++) {
    unsigned int pos = (idx + i) % cache->capacity;
    if (cache->entries[pos].key == NULL) {
      // Insert new entry.
      cache->entries[pos].key = strdup_safe(key);
      cache->entries[pos].value = malloc(value_size);
      memcpy(cache->entries[pos].value, value, value_size);
      cache->entries[pos].value_size = value_size;
      cache->size++;
      return;
    }
    if (strcmp(cache->entries[pos].key, key) == 0) {
      // Update existing entry.
      free(cache->entries[pos].value);
      cache->entries[pos].value = malloc(value_size);
      memcpy(cache->entries[pos].value, value, value_size);
      cache->entries[pos].value_size = value_size;
      return;
    }
  }
  // If no free slot found, evict the slot at idx.
  unsigned int pos = idx;
  free(cache->entries[pos].key);
  free(cache->entries[pos].value);
  cache->entries[pos].key = strdup_safe(key);
  cache->entries[pos].value = malloc(value_size);
  memcpy(cache->entries[pos].value, value, value_size);
  cache->entries[pos].value_size = value_size;
}

LRUCache* g_encode_cache = NULL;
LRUCache* g_decode_cache = NULL;
LRUCache* g_train_cache = NULL;

void initialize_caches() {
  if (!g_encode_cache) g_encode_cache = create_lru_cache(10240);
  if (!g_decode_cache) g_decode_cache = create_lru_cache(10240);
  if (!g_train_cache) g_train_cache = create_lru_cache(10240);
}

// ---------------------- Incremental Merge Caching Helpers ----------------------

// merge_with_positions(): Given the current ids array, it scans for every occurrence 
// of 'pair' and replaces it with 'new_token'. It returns the new ids array,
// its new size (in *new_size) and an array (merge_positions) containing the positions
// (in the new ids array) where a merge occurred
int* merge_with_positions(const int* ids, int ids_size, Pair pair, int new_token, size_t* new_size, int** merge_positions, int* num_positions) {
  int* new_ids = (int*) malloc(ids_size * sizeof(int));
  if (!new_ids) {
    fprintf(stderr, "Error: Memory allocation failed in merge_with_positions.\n");
    exit(EXIT_FAILURE);
  }
  int* positions = (int*) malloc(ids_size * sizeof(int));  // worst-case allocation
  if (!positions) {
    fprintf(stderr, "Error: Memory allocation failed for positions in merge_with_positions.\n");
    exit(EXIT_FAILURE);
  }
  int pos_count = 0;
  int new_idx = 0;
  for (int i = 0; i < ids_size; i++) {
    if (i < ids_size - 1 && ids[i] == pair.idx1 && ids[i + 1] == pair.idx2) {
      new_ids[new_idx] = new_token;
      positions[pos_count++] = new_idx;
      new_idx++;
      i++; // skip next token because it’s merged
    } else {
      new_ids[new_idx++] = ids[i];
    }
  }
  *new_size = new_idx;
  *merge_positions = (int*) realloc(positions, pos_count * sizeof(int));
  *num_positions = pos_count;
  return new_ids;
}

// update_frequency_cache_for_merge(): For one merge occurrence at position merge_pos (in the updated ids array),
// update the frequency cache for the left and right neighbors. It decrements the frequency
// for the old adjacent pair and increments the frequency for the new pair with new_token
void update_frequency_cache_for_merge(const int* ids, int ids_size, int merge_pos, int new_token) {
  size_t size;
  if (merge_pos > 0) {
    char key_old[64];
    snprintf(key_old, sizeof(key_old), "P:%d,%d", ids[merge_pos-1], ids[merge_pos]);
    int* freq_old = (int*) lru_cache_get(g_train_cache, key_old, &size);
    int f_old = (freq_old ? *freq_old : 0);
    if (freq_old) free(freq_old);
    if (f_old > 0) f_old--;
    lru_cache_put(g_train_cache, key_old, &f_old, sizeof(int));
    
    char key_new[64];
    snprintf(key_new, sizeof(key_new), "P:%d,%d", ids[merge_pos-1], new_token);
    int* freq_new = (int*) lru_cache_get(g_train_cache, key_new, &size);
    int f_new = (freq_new ? *freq_new : 0);
    if (freq_new) free(freq_new);
    f_new++;
    lru_cache_put(g_train_cache, key_new, &f_new, sizeof(int));
  }
  if (merge_pos < ids_size - 1) {
    char key_old[64];
    snprintf(key_old, sizeof(key_old), "P:%d,%d", ids[merge_pos], ids[merge_pos+1]);
    int* freq_old = (int*) lru_cache_get(g_train_cache, key_old, &size);
    int f_old = (freq_old ? *freq_old : 0);
    if (freq_old) free(freq_old);
    if (f_old > 0) f_old--;
    lru_cache_put(g_train_cache, key_old, &f_old, sizeof(int));
    
    char key_new[64];
    snprintf(key_new, sizeof(key_new), "P:%d,%d", new_token, ids[merge_pos+1]);
    int* freq_new = (int*) lru_cache_get(g_train_cache, key_new, &size);
    int f_new = (freq_new ? *freq_new : 0);
    if (freq_new) free(freq_new);
    f_new++;
    lru_cache_put(g_train_cache, key_new, &f_new, sizeof(int));
  }
}

// Optionally, clear the frequency for the merged pair from the training cache.
void clear_merged_pair_in_cache(Pair pair) {
  char key[64];
  snprintf(key, sizeof(key), "P:%d,%d", pair.idx1, pair.idx2);
  int zero = 0;
  lru_cache_put(g_train_cache, key, &zero, sizeof(int));
}

/* ---------- Priority Queue (Binary Heap) for Training Merge Selection ---------- */
static void swap_tokenpair(TokenPair* a, TokenPair* b) {
  TokenPair temp = *a;
  *a = *b;
  *b = temp;
}

PriorityQueue* pq_create(int capacity) {
  PriorityQueue* pq = (PriorityQueue*) malloc(sizeof(PriorityQueue));
  pq->data = (TokenPair*) malloc(capacity * sizeof(TokenPair));
  pq->size = 0;
  pq->capacity = capacity;
  return pq;
}

void pq_free(PriorityQueue* pq) {
  if (pq) {
    free(pq->data);
    free(pq);
  }
}

void pq_heapify_up(PriorityQueue* pq, int idx) {
  while (idx > 0) {
    int parent = (idx - 1) / 2;
    if (pq->data[idx].frequency > pq->data[parent].frequency) {
      swap_tokenpair(&pq->data[idx], &pq->data[parent]);
      idx = parent;
    } else {
      break;
    }
  }
}

void pq_heapify_down(PriorityQueue* pq, int idx) {
  while (1) {
    int left = 2 * idx + 1;
    int right = 2 * idx + 2;
    int largest = idx;
    if (left < pq->size && pq->data[left].frequency > pq->data[largest].frequency)
      largest = left;
    if (right < pq->size && pq->data[right].frequency > pq->data[largest].frequency)
      largest = right;
    if (largest != idx) {
      swap_tokenpair(&pq->data[idx], &pq->data[largest]);
      idx = largest;
    } else {
      break;
    }
  }
}

void pq_push(PriorityQueue* pq, TokenPair pair) {
  if (pq->size == pq->capacity) {
    pq->capacity *= 2;
    pq->data = (TokenPair*) realloc(pq->data, pq->capacity * sizeof(TokenPair));
  }
  pq->data[pq->size] = pair;
  pq_heapify_up(pq, pq->size);
  pq->size++;
}

TokenPair pq_pop(PriorityQueue* pq) {
  TokenPair top = pq->data[0];
  pq->size--;
  pq->data[0] = pq->data[pq->size];
  pq_heapify_down(pq, 0);
  return top;
}

int pq_empty(PriorityQueue* pq) {
  return (pq->size == 0);
}