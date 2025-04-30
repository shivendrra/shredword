#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "base.h"

TrieNode *create_node() {
  TrieNode* node = (TrieNode*)malloc(sizeof(TrieNode));
  node->terminal = false;
  for (int i = 0; i < NUM_CHARS; i++) {
    node->children[i] = NULL;
  }
  return node;
}

void trie_insert(TrieNode *root, const char *word) {
  if (root == NULL) {
    root = create_node();
  }
  unsigned char *idx = (unsigned char *)word;
  TrieNode *temp = root;
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

void print_trie_recursively(TrieNode *node, unsigned char *prefix, int length) {
  unsigned char newprefix[length+2];
  memcpy(newprefix, prefix, length);
  newprefix[length+1] = 0;

  if (node->terminal) {
    printf("WORD: %s\n", prefix);
  }

  for (int i = 0; i < NUM_CHARS; i++) {
    if (node->children[i] != NULL) {
      newprefix[length] = i;
      print_trie_recursively(node->children[i], newprefix, length+1);
    }
  }
}

void print_trie(TrieNode *node) {
  if (node == NULL) {
    fprintf(stderr, "Error: Invaild Trie-Node to print.\n");
    exit(EXIT_FAILURE);
  }
  print_trie_recursively(node, NULL, 0);
}

void free_trie(TrieNode *node) {
  if (node == NULL) {
    fprintf(stderr, "Error: Invaild Trie-Node to free from memory.\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < NUM_CHARS; ++i) {
    free_trie(node->children[i]);
  }
  delete node;
}

// helper function for saving new vocabs in tries, recursively
static void save_vocab_rec(TrieNode* node, FILE* out, unsigned char* buf, int depth) {
  if (node->terminal) {
    buf[depth] = '\0';
    fprintf(out, "%s\n", buf);
  }
  for (int c = 0; c < NUM_CHARS; c++) {
    if (node->children[c]) {
      buf[depth] = (unsigned char)c;
      save_vocab_rec(node->children[c], out, buf, depth + 1);
    }
  }
}

void save_vocab(TrieNode* root, const char* vocab_file) {
  FILE* out = fopen(vocab_file, "w");
  if (!out) { perror("fopen(vocab_file)"); return; }
  unsigned char* buffer = (unsigned char*)malloc(1024);
  save_vocab_rec(root, out, buffer, 0);
  free(buffer);
  fclose(out);
}

void load_vocab(TrieNode* root, const char* model_file) {
  if (!root) {
    fprintf(stderr, "Error: Trie pointer is null.\n");
    exit(EXIT_FAILURE);
  }
  if (!model_file) {
    fprintf(stderr, "Error: model_file pointer is null.\n");
    exit(EXIT_FAILURE);
  }

  printf("Loading vocab & model from: '%s' \n", model_file);
  FILE* fp = fopen(model_file, "r");
  if (!fp) {
    fprintf(stderr, "Error: Could not open model file: %s\n", model_file);
    exit(EXIT_FAILURE);
  }
  char line[MAX_LINE_LENGTH];
  fgets(line, MAX_LINE_LENGTH, fp); // version
  while (fgets(line, sizeof(line), fp)) {
    line[strcspn(line, "\r\n")] = '\0';
    if (line[0]) trie_insert(root, line);
  }
  printf("Loaded saved merges successfully!\n");
  fclose(fp);
}