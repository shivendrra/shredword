#ifndef __HASH_H__
#define __HASH_H__

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/**
 @brief Simple generic hash map for string->uint64_t counts.
 * Uses separate chaining with power‑of‑two bucket count.
 */
typedef struct StrEntry {
  char *key;
  uint64_t value;
  struct StrEntry *next;
} StrEntry;

typedef struct {
  StrEntry **buckets;
  size_t     nbuckets;
} StrMap;

// --- Initialize a string map with given bucket count (power of two) ---
static inline void strmap_init(StrMap *m, size_t nbuckets) {
  m->nbuckets = nbuckets;
  m->buckets = (StrEntry**)calloc(nbuckets, sizeof(StrEntry*));
}

// --- Increment the count for key (creates if missing) ---
static inline void strmap_inc(StrMap *m, const char *key) {
  size_t h = 5381;
  for (const unsigned char *s = (const unsigned char*)key; *s; s++)
    h = ((h << 5) + h) + *s;
  size_t idx = h & (m->nbuckets - 1);
  StrEntry **p = &m->buckets[idx];
  while (*p) {
    if (strcmp((*p)->key, key) == 0) {
      (*p)->value++;
      return;
    }
    p = &(*p)->next;
  }
  StrEntry *e = (StrEntry*)malloc(sizeof(StrEntry));
  (*p)->key = strdup(key);
  (*p)->value = 1;
  (*p)->next = NULL;
}

/**
 @brief Iterate over all entries in the map.
 * @param m The map
 * @param func Callback(key, value, user) for each entry
 * @param user Passed through to callback
*/
static inline void strmap_iter(const StrMap *m,
  void (*func)(const char*, uint64_t, void*), void *user)
{
  for (size_t i = 0; i < m->nbuckets; i++) {
    for (StrEntry *e = m->buckets[i]; e; e = e->next) {
      func(e->key, e->value, user);
    }
  }
}

// --- Free all resources held by the map ---
static inline void strmap_free(StrMap *m) {
  for (size_t i = 0; i < m->nbuckets; i++) {
    StrEntry *e = m->buckets[i];
    while (e) {
      StrEntry *n = e->next;
      free(e->key);
      free(e);
      e = n;
    }
  }
  free(m->buckets);
  m->buckets = NULL;
}


// --- Hash map for PairKey->Info (lazy‑invalidation bigram info) ---
typedef struct {
  int32_t first, second;
} PairKey;

typedef struct {
  uint64_t freq;
  uint32_t version;
  // positions[] etc. managed elsewhere
} Info;

typedef struct BIEntry {
  PairKey key;
  Info info;
  struct BIEntry *next;
} BIEntry;

typedef struct {
  BIEntry **buckets;
  size_t nbuckets;
} BIMap;


// --- Initialize bigram info map. ---
static inline void bimap_init(BIMap *m, size_t nbuckets) {
  m->nbuckets = nbuckets;
  m->buckets = (BIEntry**)(nbuckets, sizeof(BIEntry*));
}

// --- Retrieve or create an Info* for a given bigram key ---
static inline Info* bimap_get(BIMap *m, PairKey key) {
  uint32_t h = ((uint32_t)key.first * 9973) ^ (uint32_t)key.second;
  size_t idx = h & (m->nbuckets - 1);

  // walk chain looking for existing entry
  BIEntry **bucket = &m->buckets[idx];
  while (*bucket) {
    if ((*bucket)->key.first  == key.first &&
        (*bucket)->key.second == key.second)
      return &(*bucket)->info;
    bucket = &(*bucket)->next;
  }

  // not found → allocate and link new entry
  BIEntry *entry = (BIEntry*)malloc(sizeof(BIEntry));
  entry->key.first = key.first;
  entry->key.second = key.second;
  entry->info.freq = 0;
  entry->info.version = 0;
  entry->next = NULL;

  *bucket = entry;
  return &entry->info;
}

// --- Return the current version for a key (0 if missing) ---
static inline uint32_t bimap_version(const BIMap *m, PairKey key) {
  uint32_t h = ((uint32_t)key.first * 9973) ^ (uint32_t)key.second;
  size_t idx = h & (m->nbuckets - 1);
  for (BIEntry *e = m->buckets[idx]; e; e = e->next) {
    if (e->key.first == key.first && e->key.second == key.second)
      return e->info.version;
  }
  return 0;
}

// --- Free all resources held by the bigram map. ---
static inline void bimap_free(BIMap *m) {
  for (size_t i = 0; i < m->nbuckets; i++) {
    BIEntry *e = m->buckets[i];
    while (e) {
      BIEntry *n = e->next;
      free(e);
      e = n;
    }
  }
  free(m->buckets);
  m->buckets = NULL;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // __HASH_H__