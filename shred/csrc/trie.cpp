#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdlib.h>
#include "trie.h"
#include "unigram/hash.h"

TrieNode* trie_create_node() {
  TrieNode* node = (TrieNode*)calloc(1, sizeof(TrieNode));
  node->string_id = UINT32_MAX;
  return node;
}

void trie_insert(TrieNode* root, StringInterner* interner, const char* str, size_t len, size_t min_freq) {
  if (len == 0) return;
  
  TrieNode* current = root;
  for (size_t i = 0; i < len; i++) {
    unsigned char c = (unsigned char)str[i];
    if (!current->children[c]) {
      current->children[c] = trie_create_node();
      current->children[c]->depth = current->depth + 1;
    }
    current = current->children[c];
    current->frequency++;
    
    // early termination for rare patterns
    if (i > 0 && current->frequency < min_freq && current->depth > 3) {
      return;
    }
  }
  
  if (current->frequency >= min_freq) {
    current->is_terminal = true;
    if (current->string_id == UINT32_MAX) {
      current->string_id = interner_add(interner, str, len);
    }
  }
}

void trie_collect_entries(TrieNode* node, HashVocabEntry* entries, size_t* count, size_t max_count, StringInterner* interner) {
  if (!node || *count >= max_count) return;
  
  if (node->is_terminal && node->string_id != UINT32_MAX) {
    entries[*count] = (HashVocabEntry){
      .string_id = node->string_id,
      .frequency = node->frequency,
      .score = 0.0,
      .length = interner_get_length(interner, node->string_id)
    };
    (*count)++;
  }
  
  for (int i = 0; i < 256; i++) {
    if (node->children[i]) {
      trie_collect_entries(node->children[i], entries, count, max_count, interner);
    }
  }
}

void trie_free(TrieNode* node) {
  if (!node) return;
  
  for (int i = 0; i < 256; i++) {
    if (node->children[i]) {
      trie_free(node->children[i]);
    }
  }
  free(node);
}

int longest_prefix(TrieNode* root, const char* text) {
  if (root == NULL) {
    fprintf(stderr, "Error: Invalid Node to check for length.\n");
    exit(EXIT_FAILURE);
  }
  TrieNode* temp = root;
  int max_len = 0, pos = 0;
  while (text[pos] && temp->children[(unsigned char)text[pos]]) {
    temp = temp->children[(unsigned char)text[pos]];
    pos++;
    if (temp->is_terminal) {
      max_len = pos;
    }
  }
  return max_len;
}

int trie_count_words(TrieNode* node) {
  if (node == NULL) {
    return 0;
  }
  int count = node->is_terminal ? 1 : 0;
  for (int i = 0; i < NUM_CHARS; i++) {
    if (node->children[i]) {
      count += trie_count_words(node->children[i]);
    }
  }
  return count;
}

void _print_trie_recursive(TrieNode* node, unsigned char* prefix, int length) {
  if (!node) return;
  
  unsigned char newprefix[length + 2];
  if (prefix) {
    memcpy(newprefix, prefix, length);
  }
  newprefix[length + 1] = 0;

  if (node->is_terminal) {
    printf("WORD: %s\n", (prefix) ? (char*)prefix : "");
  }

  for (int i = 0; i < NUM_CHARS; i++) {
    if (node->children[i] != NULL) {
      newprefix[length] = i;
      _print_trie_recursive(node->children[i], newprefix, length + 1);
    }
  }
}

void print_trie(TrieNode* node) {
  if (node == NULL) {
    fprintf(stderr, "Error: Invalid Trie-Node to print.\n");
    return;
  }
  unsigned char empty_prefix = 0;
  _print_trie_recursive(node, &empty_prefix, 0);
}

void free_trie(TrieNode* node) {
  if (!node) return;

  for (int i = 0; i < NUM_CHARS; ++i) {
    if (node->children[i]) {
      free_trie(node->children[i]);
    }
  }
  free(node);
}