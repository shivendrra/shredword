/**
  @brief trie-based dtype to store possible vocabs
  
  * trie-based functions for creating, inserting & deleting tries & entries
  * word-count & printing functions
*/

#ifndef __TRIE__H__
#define __TRIE__H__

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#define NUM_CHARS 256   // Maximum initial letters, since UTF-8 so 0-255

// Forward declarations
typedef struct StringInterner StringInterner;
typedef struct HashVocabEntry HashVocabEntry;

typedef struct TrieNode {
  struct TrieNode* children[256];
  uint64_t frequency;
  uint32_t string_id;
  uint16_t depth;
  bool is_terminal;
} TrieNode; // Trie node for efficient substring storage

extern "C" {
  TrieNode* trie_create_node();
  void trie_insert(TrieNode* root, StringInterner* interner, const char* str, size_t len, size_t min_freq);
  void trie_collect_entries(TrieNode* node, HashVocabEntry* entries, size_t* count, size_t max_count, StringInterner* interner);
  void trie_free(TrieNode* node);
  int trie_count_words(TrieNode* node);
}

#endif