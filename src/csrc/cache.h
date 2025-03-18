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
void initialize_threads();  // set MAX_THREADS dynamically
#define INITIAL_CACHE_SIZE 2048  // Maximum number of LRU cache entries
extern size_t token_length_cache[VOCAB_SIZE + MAX_MERGES]; // defined in cache.cpp

typedef struct {
  const Shred* tokenizer;
  const int* ids;
  int start;
  int end;
  char* output_str;
  int* output_int;
  size_t* output_size;
} ThreadArgs;

// For training worker thread arguments
typedef struct {
  int idx1;
  int idx2;
  int freq;
} PairStat;

typedef struct {
  const int* ids;
  int start;
  int end;
  PairStat* local_stats;  // Array to store partial pair stats (size at least INITIAL_CACHE_SIZE)
  int* local_count;       // Pointer to an int storing number of distinct pairs found
} TrainThreadArgs;

#ifdef __cplusplus
extern "C" {
#endif

  void initialize_token_cache(const Shred* tokenizer);
  void* decode_worker(void* args);
  void* encode_worker(void* args);
  void* train_worker(void* args);
  void merge_train_stats(PairStat* global_stats, int* global_count, PairStat** partial_stats, int* counts, int num_threads);
  unsigned int hash(const char* key);

  /* ---------- LRU Cache (hashmap-based) ---------- */
  typedef struct {
    char* key;         // dynamically allocated key
    void* value;       // pointer to cached data
    size_t value_size; // size of the cached data in bytes
  } LRUCacheEntry;

  typedef struct {
    LRUCacheEntry* entries; // array of entries (open addressing)
    int capacity;           // maximum number of entries
    int size;               // current number of entries
  } LRUCache;

  LRUCache* create_lru_cache(int capacity);
  void free_lru_cache(LRUCache* cache);
  // lru_cache_get returns a malloced copy of the value if found; caller must free it.
  void* lru_cache_get(LRUCache* cache, const char* key, size_t* value_size);
  void lru_cache_put(LRUCache* cache, const char* key, const void* value, size_t value_size);
  void initialize_caches(); // initialize global caches

  // Global caches for encoding, decoding, and training merges.
  extern LRUCache* g_encode_cache;
  extern LRUCache* g_decode_cache;
  extern LRUCache* g_train_cache;

  /* ---------- Priority Queue (Binary Heap) for Training ---------- */
  typedef struct {
    int idx1;
    int idx2;
    int frequency;
  } TokenPair;

  typedef struct {
    TokenPair* data;
    int size;
    int capacity;
  } PriorityQueue;

  PriorityQueue* pq_create(int capacity);
  void pq_free(PriorityQueue* pq);
  void pq_push(PriorityQueue* pq, TokenPair pair);
  TokenPair pq_pop(PriorityQueue* pq);
  int pq_empty(PriorityQueue* pq);

  /* ---------- Incremental Merge Helpers ---------- */
  int* merge_with_positions(const int* ids, int ids_size, Pair pair, int new_token, size_t* new_size, int** merge_positions, int* num_positions);
  void update_frequency_cache_for_merge(const int* ids, int ids_size, int merge_pos, int new_token);
  void clear_merged_pair_in_cache(Pair pair);

#ifdef __cplusplus
}
#endif

#endif