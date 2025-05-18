#include "trie.h"
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>

TrieNode* create_node() {
  TrieNode* node = (TrieNode*)malloc(sizeof(TrieNode));
  node->terminal = false;
  for (int i = 0; i < NUM_CHARS; i++) {
    node->children[i] = NULL;
  }
  return node;
}

void trie_insert(TrieNode* root, const char* word) {
  if (root == NULL) {
    fprintf(stderr, "Error: Invaild Node to insert new word.\n");
    exit(EXIT_FAILURE);
  }
  unsigned char* idx = (unsigned char*)word;
  TrieNode* temp = root;
  size_t length = strlen(word);
  for (int i = 0; i < length; i++) {
    if (temp->children[idx[i]] == NULL) {
      temp->children[idx[i]] = create_node();
    }
    temp = temp->children[idx[i]];
  }
  temp->terminal = true;
}

int longest_prefix(TrieNode* root, const char* text) {
  if (root == NULL) {
    fprintf(stderr, "Error: Invaild Node to check for length.\n");
    exit(EXIT_FAILURE);
  }
  TrieNode* temp = root;
  int max_len = 0, pos = 0;
  while (text[pos] && temp->children[(unsigned char)text[pos]]) {
    temp = temp->children[(unsigned char)text[pos]];
    pos++;
    if (temp->terminal) {
      max_len = pos;
    }
  }
  return max_len;
}

int trie_count_words(TrieNode* node) {
  if (node == NULL) {
    fprintf(stderr, "Error: TrieNode pointer is NULL.\n");
    exit(EXIT_FAILURE);
  }
  int count = node->terminal ? 1 : 0;
  for (int i = 0; i < NUM_CHARS; i++) {
    count += trie_count_words(node->children[i]);
  }
  return count;
}

void _print_trie_recusrive(TrieNode* node, unsigned char* prefix, int length) {
  unsigned char newprefix[length + 2];
  memcpy(newprefix, prefix, length);
  newprefix[length + 1] = 0;

  if (node->terminal) {
    printf("WORD: %s\n", prefix);
  }

  for (int i = 0; i < NUM_CHARS; i++) {
    if (node->children[i] != NULL) {
      newprefix[length] = i;
      _print_trie_recusrive(node->children[i], newprefix, length+1);
    }
  }
}

void print_trie(TrieNode* node) {
  if (node == NULL) {
    fprintf(stderr, "Error: Invaild Trie-Node to print.\n");
    exit(EXIT_FAILURE);
  }
  _print_trie_recusrive(node, NULL, 0);
}

void free_trie(TrieNode* node) {
  if (!node) return;  // return silently (usually throws error when trie is initialized as NULL pointer)

  for (int i = 0; i < NUM_CHARS; ++i) {
    if (node->children[i]) {
      free_trie(node->children[i]);
    }
  }
  free(node);
}