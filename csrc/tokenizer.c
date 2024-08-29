#include "tokenizer.h"
#include <stdio.h>
#include <string.h>
#include <pthread.h>
#include <stdint.h>

static const char *vocabulary[] = {"a", "b", "c", "ab", "bc", "abc"};
static const size_t vocab_size = sizeof(vocabulary) / sizeof(vocabulary[0]);

typedef struct {
  const char *first;
  const char *second;
  const char *merge_result;
} MergePair;

static const MergePair merge_pairs[] = {
  {"a", "b", "ab"},
  {"b", "c", "bc"},
  {"ab", "c", "abc"}
};
static const size_t merge_pairs_size = sizeof(merge_pairs) / sizeof(merge_pairs[0]);

#define FXHASH_SEED 0x9E3779B9
static inline uint32_t fxhash(const void *data, size_t len) {
  const uint8_t *ptr = (const uint8_t *)data;
  uint32_t h = FXHASH_SEED;
  for (size_t i = 0; i < len; i++) {
    h ^= ptr[i];
    h *= 0x01000193;
  }
  return h;
}

typedef struct HashEntry {
  char *key;
  char *value;
  struct HashEntry *next;
} HashEntry;

typedef struct {
  HashEntry **buckets;
  size_t bucket_count;
} HashMap;

static HashMap *merge_cache;

static HashMap* hashmap_create(size_t bucket_count) {
  HashMap *map = malloc(sizeof(HashMap));
  map->bucket_count = bucket_count;
  map->buckets = calloc(bucket_count, sizeof(HashEntry*));
  return map;
}

static void hashmap_insert(HashMap *map, const char *key, const char *value) {
  uint32_t hash = fxhash(key, strlen(key)) % map->bucket_count;
  HashEntry *entry = malloc(sizeof(HashEntry));
  entry->key = strdup(key);
  entry->value = strdup(value);
  entry->next = map->buckets[hash];
  map->buckets[hash] = entry;
}

static const char* hashmap_lookup(HashMap *map, const char *key) {
  uint32_t hash = fxhash(key, strlen(key)) % map->bucket_count;
  HashEntry *entry = map->buckets[hash];
  while (entry) {
    if (strcmp(entry->key, key) == 0) {
      return entry->value;
    }
    entry = entry->next;
  }
  return NULL;
}

static void hashmap_free(HashMap *map) {
  for (size_t i = 0; i < map->bucket_count; i++) {
    HashEntry *entry = map->buckets[i];
    while (entry) {
      HashEntry *temp = entry;
      entry = entry->next;
      free(temp->key);
      free(temp->value);
      free(temp);
    }
  }
  free(map->buckets);
  free(map);
}

void initialize_tokenizer(const char *vocab_file, const char *merge_file) {
  merge_cache = hashmap_create(1024); // Cache with 1024 buckets for merges
}

TokenList tokenize(const char *input) {
  TokenList token_list;
  token_list.tokens = malloc(strlen(input) * sizeof(Token));
  token_list.num_tokens = 0;

  size_t i = 0;
  while (i < strlen(input)) {
    char token[256] = {0};
    size_t token_len = 0;

    for (size_t j = i; j < strlen(input); j++) {
      token[token_len++] = input[j];
      token[token_len] = '\0';

      if (hashmap_lookup(merge_cache, token)) {
        break;
      }
    }

    if (!hashmap_lookup(merge_cache, token)) {
      token[0] = input[i];
      token[1] = '\0';
      token_len = 1;
    }

    token_list.tokens[token_list.num_tokens].data = strdup(token);
    token_list.tokens[token_list.num_tokens].size = token_len;
    token_list.num_tokens++;

    i += token_len;
  }

  for (size_t i = 0; i < token_list.num_tokens - 1; i++) {
    char merge_key[512];
    snprintf(merge_key, sizeof(merge_key), "%s%s", token_list.tokens[i].data, token_list.tokens[i + 1].data);

    const char *merged = hashmap_lookup(merge_cache, merge_key);
    if (!merged) {
      merged = merge_key;
      hashmap_insert(merge_cache, merge_key, merged);
    }

    if (merged) {
      free(token_list.tokens[i].data);
      free(token_list.tokens[i + 1].data);
      token_list.tokens[i].data = strdup(merged);
      token_list.tokens[i].size = strlen(merged);

      // Shift tokens
      for (size_t j = i + 1; j < token_list.num_tokens - 1; j++) {
        token_list.tokens[j] = token_list.tokens[j + 1];
      }
      token_list.num_tokens--;
    }
  }

  return token_list;
}

void free_token_list(TokenList *token_list) {
  for (size_t i = 0; i < token_list->num_tokens; i++) {
    free(token_list->tokens[i].data);
  }
  free(token_list->tokens);
}