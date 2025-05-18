/**
  @brief trie-based dtype to store possible vocabs
  
  * trie-based functions for creating, inserting & deleting tries & entries
  * word-count & printing functions
*/

#ifndef __TRIE__H__
#define __TRIE__H__

#define  NUM_CHARS  256   // Maximum initial letters, since UTF-8 so 0-255

typedef struct TrieNode {
  struct TrieNode* children[256];
  bool terminal;
} TrieNode;   // Trie-based dtype to store possible vocabs

extern "C" {
  TrieNode* create_node();
  void trie_insert(TrieNode* root, const char* word);
  int longest_prefix(TrieNode* root, const char* text);
  int trie_count_words(TrieNode* node);   // freeing the trie from the memory
  void print_trie(TrieNode* node);   // prints all the nodes recursively
  void free_trie(TrieNode* node);
}

#endif