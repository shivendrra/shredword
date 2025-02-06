#include "cache.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void initialize_token_cache(const Shred* tokenizer) {
  for (int i = 0; i < VOCAB_SIZE + MAX_MERGES; i++) {
    if (tokenizer->base.vocab[i].value) {
      token_length_cache[i] = strlen(tokenizer->base.vocab[i].value);
    } else {
      token_length_cache[i] = 0;
    }
  }
}

// worker function for decoding in parallel
void* decode_worker(void* args) {
  ThreadArgs* thread_args = (ThreadArgs*)args;
  const Shred* tokenizer = thread_args->tokenizer;
  const int* ids = thread_args->ids;
  int start = thread_args->start;
  int end = thread_args->end;

  size_t local_size = 0;

  // calculating local size
  for (int i = start; i < end; i++) {
    local_size += token_length_cache[ids[i]];
  }

  char* local_output = (char*)malloc(local_size + 1);
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
  ThreadArgs* thread_args = (ThreadArgs*)args;
  const Shred* tokenizer = thread_args->tokenizer;
  const int* ids = thread_args->ids;
  int start = thread_args->start;
  int end = thread_args->end;

  size_t local_size = end - start;
  int* local_output = (int*)malloc(local_size * sizeof(int));
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

unsigned int hash(const char* key) {
  unsigned int hash = 0;
  while (*key) {
    hash = (hash * 31) + *key++;
  }
  return hash % INITIAL_CACHE_SIZE;
}

LRUCache* init_cache(int initial_capacity) {
  LRUCache* cache = (LRUCache*)malloc(sizeof(LRUCache));
  if (!cache) {
    fprintf(stderr, "Error: Memory allocation for LRUCache failed.\n");
    exit(EXIT_FAILURE);
  }

  cache->head = NULL;
  cache->tail = NULL;
  cache->size = 0;
  cache->capacity = initial_capacity;
  cache->table = (CacheNode**)calloc(initial_capacity, sizeof(CacheNode*));
  if (!cache->table) {
    fprintf(stderr, "Error: Memory allocation for LRUCache table failed.\n");
    free(cache);
    exit(EXIT_FAILURE);
  }
  return cache;
}

void resize_cache(LRUCache* cache, int new_cap) {
  int new_capacity = new_cap;
  CacheNode** new_table = (CacheNode**)calloc(new_capacity, sizeof(CacheNode*));
  if (!new_table) {
    fprintf(stderr, "Error: Memory allocation for resizing LRUCache table failed.\n");
    exit(EXIT_FAILURE);
  }

  // rehashing existing entries into the new table
  for (int i = 0; i < cache->capacity; i++) {
    CacheNode* node = cache->table[i];
    while (node) {
      unsigned int new_index = hash(node->key) % new_capacity;
      CacheNode* next = node->next;
      node->next = new_table[new_index];
      new_table[new_index] = node;
      node = next;
    }
  }

  free(cache->table);
  cache->table = new_table;
  cache->capacity = new_capacity;
}

void remove_node(LRUCache* cache, CacheNode* node) {
  if (node->prev) {
    node->prev->next = node->next;
  } else {
    cache->head = node->next;
  }
  if (node->next) {
    node->next->prev = node->prev;
  } else {
    cache->tail = node->prev;
  }
}

void add_to_front(LRUCache* cache, CacheNode* node) {
  node->next = cache->head;
  node->prev = NULL;
  if (cache->head) {
    cache->head->prev = node;
  }
  cache->head = node;
  if (!cache->tail) {
    cache->tail = node;
  }
}

void put(LRUCache* cache, const char* key, int value) {
  unsigned int hash_index = hash(key);
  CacheNode* existing = cache->table[hash_index];

  while (existing) {
    if (strcmp(existing->key, key) == 0) {
      existing->value = value;
      add_to_front(cache, existing);
      return;
    }
    existing = existing->next;
  }

  CacheNode* new_node = (CacheNode*)malloc(sizeof(CacheNode));
  if (!new_node) {
    fprintf(stderr, "Error: Memory allocation for cache node failed.\n");
    exit(EXIT_FAILURE);
  }

  new_node->key = strdup(key);
  new_node->value = value;
  new_node->prev = NULL;
  new_node->next = cache->head;

  if (cache->head) {
    cache->head->prev = new_node;
  }
  cache->head = new_node;

  if (!cache->tail) {
    cache->tail = new_node;
  }

  cache->table[hash_index] = new_node;
  cache->size++;

  if (cache->size > cache->capacity) {
    resize_cache(cache, cache->capacity * 2);
  }
}

int get(LRUCache* cache, const char* key) {
  unsigned int idx = hash(key);
  CacheNode* node = cache->table[idx];
  while (node) {
    if (strcmp(node->key, key) == 0) {
      // move the accessed node to the front
      remove_node(cache, node);
      add_to_front(cache, node);
      return node->value;
    }
    node = node->next;
  }
  return -1; // not found
}

void free_cache(LRUCache* cache) {
  if (!cache) return;

  // free the doubly linked list nodes
  CacheNode* current = cache->head;
  while (current) {
    CacheNode* next = current->next;
    free(current->key);  // freeing the dynamically allocated key
    free(current);       // freeing the node itself
    current = next;
  }
  free(cache->table);
  free(cache);
}