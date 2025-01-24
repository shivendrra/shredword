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

#define MAX_THREADS 6
#define INITIAL_CACHE_SIZE 1024  // max size of LRU cache
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

typedef struct CacheNode {
  char* key;  // serialized form of token pair
  int value;  // cached merged index
  struct CacheNode* prev;
  struct CacheNode* next;
} CacheNode;

typedef struct {
  CacheNode* head;  // head of body of doubly linked list
  CacheNode* tail;  // tail of body of doubly linked list
  int size, capacity; // current size, & max capacity
  CacheNode** table; // dynamically allocated hash table
} LRUCache;

extern "C" {
  void initialize_token_cache(const Shred* tokenizer);
  void* decode_worker(void* args);
  void* encode_worker(void* args);
  unsigned int hash(const char* key);
  LRUCache* init_cache(int capacity);
  void resize_cache(LRUCache* cache);
  void remove_node(LRUCache* cache, CacheNode* node);
  void add_to_front(LRUCache* cache, CacheNode* node);
  int get_from_cache(LRUCache* cache, const char* key);
  void put_in_cache(LRUCache* cache, const char* key, int value);
  void free_cache(LRUCache* cache);
}

#endif