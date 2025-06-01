#ifndef __TOKENMAP__H__
#define __TOKENMAP__H__

#include <stddef.h>

typedef struct TokenEntry {
  char* token;
  int index;
  struct TokenEntry* next;
} TokenEntry;

typedef struct TokenMap {
  TokenEntry** buckets;
  size_t nbuckets;
} TokenMap;

extern "C" {
  void token_map_add(TokenMap* map, const char* token, int index);
  int token_map_get(const TokenMap* map, const char* token);
  void token_map_clear(TokenMap* map);
}

#endif  //!__TOKENMAP__H__