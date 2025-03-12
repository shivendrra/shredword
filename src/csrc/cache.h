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
#define INITIAL_CACHE_SIZE 10240  // Max size of LRU cache
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

// for training worker thread arguments
typedef struct {
  int idx1;
  int idx2;
  int freq;
} PairStat;   // used in the training worker to count occurrences of adjacent token pairs

typedef struct {
  const int* ids;
  int start;
  int end;
  PairStat* local_stats;  // array to store partial pair stats (size at least INITIAL_CACHE_SIZE)
  int* local_count;       // pointer to an int storing the number of distinct pairs found
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

  // --- LRU Cache definitions (now implemented as a hashmap) ---
  typedef struct {
    char* key;         // key string (dynamically allocated)
    void* value;       // pointer to cached data
    size_t value_size; // size of the cached data in bytes
  } LRUCacheEntry;

  typedef struct {
    LRUCacheEntry* entries; // array of entries, used in an open-addressing hash table
    int capacity;           // maximum number of entries
    int size;               // current number of entries
  } LRUCache;

  LRUCache* create_lru_cache(int capacity);
  void free_lru_cache(LRUCache* cache);
  // lru_cache_get returns a malloced copy of the value if found (caller must free), else returns NULL
  void* lru_cache_get(LRUCache* cache, const char* key, size_t* value_size);
  void lru_cache_put(LRUCache* cache, const char* key, const void* value, size_t value_size);
  void initialize_caches(); // initialize global caches
  void update_frequency_cache_for_merge(const int* ids, int ids_size, int merge_pos, int new_token);
  int* merge_with_positions(const int* ids, int ids_size, Pair pair, int new_token, size_t* new_size, int** merge_positions, int* num_positions);
  void clear_merged_pair_in_cache(Pair pair);

  // global caches for encoding, decoding, and training merges
  extern LRUCache* g_encode_cache;
  extern LRUCache* g_decode_cache;
  extern LRUCache* g_train_cache;

#ifdef __cplusplus
}
#endif

#endif