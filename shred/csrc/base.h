/**
  @base.h Base class for basic BPE functions & logics.

  * functions for tokenizer initialization, normalization, and helper routines.
  * compliation with base.cpp containing the main logic (no regex, no caching)
*/

#ifndef __BASE__H__
#define __BASE__H__

#define NUM_CHARS 256   // Maximum initial letters, since UTF-8 so 256
#define MAX_LINE_LENGTH 1024  // Max length of line to be read
#define MAX_TOKENS 1000000   // Max tokens to be trained
#define MAX_SEQ_LENGTH 4096   // BUffer length for loading sentences
#define MIN_SYMBOL_LEN 32   // Symbols size (to be increased dynamically)

typedef struct TrieNode {
  struct TrieNode *children[NUM_CHARS];
  bool terminal;
} TrieNode;   // Trie-based dtype to store possible vocabs

extern "C" {
  // functions for creating/modifying `trie`
  TrieNode* create_node();
  void trie_insert(TrieNode* root, const char* word);
  int longest_prefix(TrieNode* root, const char* text);
  int trie_count_words(TrieNode* node);
  void free_trie(TrieNode* node);   // freeing the trie from the memory
  void print_trie(TrieNode* node);  // prints all the nodes recursively

  int split_to_symbols(const char* line, char*** out_symbols);
  void load_and_split(const char* train_file, char**** out_symbols, int** out_lens, int* out_size);

  // function for loading/saving vocabs
  void save_vocab(TrieNode* root, const char* file_prefix);
  void load_vocab(TrieNode* root, const char* model_file);
}

#endif