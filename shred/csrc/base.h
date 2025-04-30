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
#define MAX_LINE_LENGTH 1024
#define MAX_TOKENS 1000000
#define MAX_SEQ_LENGTH 4096
#define MAX_SYMBOL_LEN 32

typedef struct TrieNode {
  struct TrieNode *children[NUM_CHARS];
  bool terminal;
} TrieNode;

typedef struct TokenPairs {
  int first, second;
} TokenPairs;

extern "C" {
  // functions for creating/modifying `trie`
  TrieNode *create_node();
  void trie_insert(TrieNode *root, const char* word);
  int longest_prefix(TrieNode *root, const char* text);
  void free_trie(TrieNode *node);   // freeing the trie from the memory
  void print_trie(TrieNode *node);  // prints all the nodes recursively

  // void bpe_learn(TrieNode* root, int merge_steps, const char* train_file);
  void bpe_learn(const char* train_file, int merge_steps, TrieNode* root);
  void save_vocab(TrieNode* root, const char* file_prefix);
  void load_vocab(TrieNode* root, const char* model_file);
}

#endif