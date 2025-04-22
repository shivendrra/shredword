/*
  base.h
  * Base class for basic BPE functions & logics.
  * Contains functions for tokenizer initialization, normalization, and helper routines.
  * To be compiled with base.cpp containing the main logic (no regex, no caching)
*/

#ifndef __BASE__H__
#define __BASE__H__

#include <cstddef>
#include <ctype.h>

#define NUM_CHARS 256

typedef struct TrieNode {
  struct TrieNode *children[NUM_CHARS];
  bool terminal;
} TrieNode;

extern "C" {
  // functions for creating/modifying `trie`
  TrieNode *create_node();
  void trie_insert(TrieNode *root, const char* word);
  int longest_prefix(TrieNode *root, const char* text);
  void free_trie(TrieNode *node);   // freeing the trie from the memory
  void print_trie(TrieNode *node);  // prints all the nodes recursively
}

#endif