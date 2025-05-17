#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include "hash.h"

// --- Initialize a string map with given bucket count (power of two) ---
void strmap_init(StrMap* map, size_t nbuckets) {
  if (!map) {
    fprintf(stderr, "Pointer to Map not found!\n");
    exit(EXIT_FAILURE);
  }
  map->nbuckets = nbuckets;
  map->buckets = (StrEntry**)calloc(nbuckets, sizeof(StrEntry*));
}

// --- Increment the count for key (creates if missing) ---
void strmap_increment(StrMap* map, const char* key) {
  if (!map) {
    fprintf(stderr, "Pointer to Map not found!\n");
    exit(EXIT_FAILURE);
  }
  // djb2 hash
  size_t h = 5381;
  for (const unsigned char* s = (const unsigned char*)key; *s; s++) {
    h = ((h << 5) + h) + *s;
  }
  size_t idx = h & (map->nbuckets - 1);
  StrEntry** p = &map->buckets[idx];
  while (*p) {
    if (strcmp((*p)->key, key) == 0) {
      (*p)->value++;
      return;
    }
    p = &(*p)->next;
  }
  StrEntry* e = (StrEntry*)malloc(sizeof(StrEntry));
  e->key = strdup(key);
  e->value = 1;
  e->next = NULL;
  *p = e;
}

/**
 @brief Iterate over all entries in the map.
 * @param m The map
 * @param func Callback(key, value, user) for each entry
 * @param user Passed through to callback
*/
void strmap_iter(StrMap* map, void(*func)(const char*, uint64_t, void*), void* user) {
  if (!map) {
    fprintf(stderr, "Pointer to Map not found!\n");
    exit(EXIT_FAILURE);
  }
  
  for (size_t i = 0; i < map->nbuckets; i++) {
    for (StrEntry* e = map->buckets[i]; e; e = e->next) {
      func(e->key, e->value, user);
    }
  }
}

// --- Free all resources held by the map ---
void strmap_free(StrMap *map) {
  if (!map) {
    fprintf(stderr, "Pointer to Map not found!\n");
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < map->nbuckets; i++) {
    StrEntry* e = map->buckets[i];
    while (e) {
      StrEntry *n = e->next;
      free(e->key);
      free(e);
      e = n;
    }
  }
  free(map->buckets);
  map->buckets = NULL;
}

// --- Initialize bigram info map. ---
void bimap_init(BIMap* map, size_t nbuckets) {
  if (!map) {
    fprintf(stderr, "Pointer to Map not found!\n");
    exit(EXIT_FAILURE);
  }
  map->nbuckets = nbuckets;
  map->buckets = (BIEntry**)calloc(nbuckets, sizeof(BIEntry*));
}

// --- Retrieve or create an Info* for a given bigram key ---
Info* bimap_get(BIMap* map, PairKey key) {
  if (!map) {
    fprintf(stderr, "Pointer to Map not found!\n");
    exit(EXIT_FAILURE);
  }
  uint32_t h = ((uint32_t)key.first * 9973) ^ (uint32_t)key.second;
  size_t idx = h & (map->nbuckets - 1);

  // walk chain looking for existing entry
  BIEntry **bucket = &map->buckets[idx];
  while (*bucket) {
    if ((*bucket)->key.first  == key.first &&
        (*bucket)->key.second == key.second) {
      return &(*bucket)->info;
    }
    bucket = &(*bucket)->next;
  }

  // not found -> allocate and link new entry
  BIEntry *entry = (BIEntry*)calloc(1, sizeof(BIEntry));  // zeros all fields
  entry->key.first = key.first;
  entry->key.second = key.second;
  entry->next = NULL;

  *bucket = entry;
  return &entry->info;
}

// --- Return the current version for a key (0 if missing) ---
uint32_t bimap_version(const BIMap* map, PairKey key) {
  if (!map) {
    fprintf(stderr, "Pointer to Map not found!\n");
    exit(EXIT_FAILURE);
  }
  uint32_t h = ((uint32_t)key.first * 9973) ^ (uint32_t)key.second;
  size_t idx = h & (map->nbuckets - 1);
  for (BIEntry *e = map->buckets[idx]; e; e = e->next) {
    if (e->key.first == key.first && e->key.second == key.second)
      return e->info.version;
  }
  return 0;
}

// --- Free all resources held by the bigram map. ---
void bimap_free(BIMap *map) {
  for (size_t i = 0; i < map->nbuckets; i++) {
    BIEntry *e = map->buckets[i];
    while (e) {
      BIEntry *n = e->next;
      free(e);
      e = n;
    }
  }
  free(map->buckets);
  map->buckets = NULL;
}