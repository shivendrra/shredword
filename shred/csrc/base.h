/**
  @base.h Base class for basic BPE functions & logics.

  * functions for tokenizer initialization, normalization, and helper routines.
  * compliation with base.cpp containing the main logic (no regex, no caching)
*/

#ifndef __BASE__H__
#define __BASE__H__

#include "trie.h"

#define MAX_LINE_LENGTH 1024  // Max length of line to be read
#define MAX_TOKENS 1000000   // Max tokens to be trained
#define MAX_SEQ_LENGTH 4096   // BUffer length for loading sentences
#define MIN_SYMBOL_LEN 32   // Symbols size (to be increased dynamically)
#define MAX_MERGES  10000

extern "C" {
  void get_stats(const int* ids, int ids_size, int stats[MAX_MERGES][3]);
  int split_to_symbols(const char* line, char*** out_symbols);
  void load_and_split(const char* train_file, char**** out_symbols, int** out_lens, int* out_size);

  // function for loading/saving vocabs
  void save_vocab(TrieNode* root, const char* file_prefix);
  void load_vocab(TrieNode* root, const char* model_file);
}

#endif