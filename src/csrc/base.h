/**
  @base.h Base class for basic BPE functions & logics.

  * functions for tokenizer initialization, normalization, and helper routines.
  * compliation with base.cpp containing the main logic (no regex, no caching)
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

  char* normalize_text(const char* input);  // normalize input text to NFKC form and replace spaces with "▁"
  void save_vocab(TrieNode* root, const char* vocab_file);
  TrieNode* load_vocab(const char* vocab_file);
}

#endif