/**
 @file hash.h
 @brief hasmap implementation for particularly training BPE merges

 * string -> int value hashmap, maintaining version info, etc.
 * separate hashing for Bigram related task.
*/

#ifndef __HASH_H__
#define __HASH_H__

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct StrEntry {
  char* key;  // string or character
  uint64_t value;   // int64 value of its
  struct StrEntry* next;    // linked to the next pair
} StrEntry;

typedef struct StrMap {
  StrEntry** buckets;
  size_t nbuckets;
} StrMap;

typedef struct PairKey {
  int32_t first, second;
} PairKey;

typedef struct wordPos wordPos;

typedef struct Info {
  uint64_t freq;   // frequency of a particular pair
  wordPos* positions;   // dynamic Array of occurances
  size_t pos_capacity;   // capacity of pos[]
  size_t pos_size;  // current no of occurances
  uint32_t version;   // version for lazy validation
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

extern "C" {
  // StrMap related functions ----
  void strmap_init(StrMap* map, size_t nbuckets);
  void strmap_increment(StrMap* map, const char* key);
  void strmap_iter(StrMap* map, void(*func)(const char*, uint64_t, void*), void* user);
  void strmap_free(StrMap* map);

  // BiGram Hash related functions ----
  void bimap_init(BIMap *m, size_t nbuckets);
  Info* bimap_get(BIMap *m, PairKey key);
  uint32_t bimap_version(const BIMap* map, PairKey key);
  void bimap_free(BIMap *map);
}

#endif