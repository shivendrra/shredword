#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include "tokenmap.h"

void token_map_add(TokenMap* map, const char* token, int index) {
  size_t h = 5381;
  for (const unsigned char* s = (const unsigned char*)token; *s; s++) {
    h = ((h << 5) + h) + *s;
  }
  size_t idx = h & (map->nbuckets - 1);
  
  TokenEntry* entry = (TokenEntry*)malloc(sizeof(TokenEntry));
  entry->token = strdup(token);
  entry->index = index;
  entry->next = map->buckets[idx];
  map->buckets[idx] = entry;
}

int token_map_get(const TokenMap* map, const char* token) {
  if (!map || !token) return -1;
  
  size_t h = 5381;
  for (const unsigned char* s = (const unsigned char*)token; *s; s++) {
    h = ((h << 5) + h) + *s;
  }
  size_t idx = h & (map->nbuckets - 1);
  
  TokenEntry* entry = map->buckets[idx];
  while (entry) {
    if (strcmp(entry->token, token) == 0) {
      return entry->index;
    }
    entry = entry->next;
  }
  return -1;
}

void token_map_clear(TokenMap* map) {
  for (size_t i = 0; i < map->nbuckets; i++) {
    TokenEntry* e = map->buckets[i];
    while (e) {
      TokenEntry* next = e->next;
      free(e->token);
      free(e);
      e = next;
    }
    map->buckets[i] = NULL;
  }
}