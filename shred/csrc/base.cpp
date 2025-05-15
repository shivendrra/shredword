#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "base.h"
#include "trie.h"

// Read a line, split on U+2581 marker into symbols array
int split_to_symbols(const char* line, char*** out_symbols) {
  // worst case every byte is a separate UTF-8 symbol → allocate MAX_SEQ_LENGTH pointers
  char** symbols = (char**)malloc(sizeof(char*) * MAX_SEQ_LENGTH);
  int n = 0, i = 0, L = strlen(line);
  while (i < L) {
    // detect marker 0xE2 0x96 0x81
    if (i+2 < L && (unsigned char)line[i] == 0xE2 && (unsigned char)line[i+1] == 0x96 && (unsigned char)line[i+2] == 0x81) {
      symbols[n++] = strdup("▁");
      i += 3;
    } else {
      // grabing one UTF-8 codepoint
      int len = 1;
      unsigned char c = line[i];
      if (c >= 0xC0) {
        if ((c & 0xE0) == 0xC0) len = 2;
        else if ((c & 0xF0) == 0xE0) len = 3;
        else if ((c & 0xF8) == 0xF0) len = 4;
      }
      char temp[MIN_SYMBOL_LEN];
      strncpy(temp, line + i, len);
      temp[len] = '\0';
      symbols[n++] = strdup(temp);
      i += len;
    }
  }
  *out_symbols = symbols;
  return n;
}

// helper function to open and load a file for training
void load_and_split(const char* train_file, char**** out_symbols, int** out_lens, int* out_size) {
  FILE* f = fopen(train_file,"r");
  char buf[4096];
  int capacity = 1024, corpus_size = 0;
  char*** seq_syms = (char***)malloc(sizeof(char**) * capacity);
  int* seq_lens = (int*)malloc(sizeof(int) * capacity);
  
  while (fgets(buf, sizeof(buf), f)) {
    buf[strcspn(buf, "\r\n")] = '\0';
    if (!buf[0]) continue;
    if (corpus_size == capacity) {
      capacity *= 2;
      seq_syms = (char***)realloc(seq_syms, sizeof(char**) * capacity);
      seq_lens = (int*)realloc(seq_lens, sizeof(int) * capacity);
    }
    seq_lens[corpus_size] = split_to_symbols(buf, &seq_syms[corpus_size]);
    corpus_size++;
    if (corpus_size % 10000 == 0) {
      printf("[DEBUG] read %d lines\n", corpus_size);
    }
  }
  fclose(f);
  printf("[DEBUG] total lines: %d\n", corpus_size);

  // allocating the new computed values to each of the respective variable
  *out_symbols = seq_syms;
  *out_lens = seq_lens;
  *out_size = corpus_size;
}

// helper function for saving new vocabs in tries, recursively
static void _save(TrieNode* n, FILE* f, char* buf, int depth) {
  if (n->terminal) {
    buf[depth] = '\0';
    fprintf(f, "%s\n", buf);
  }
  for (int c = 0; c < NUM_CHARS; c++) {
    if (n->children[c]) {
      buf[depth] = (char)c;
      _save(n->children[c], f, buf, depth+1);
    }
  }
}

void save_vocab(TrieNode* root, const char* vocab_file) {
  if (!root) {
    fprintf(stderr, "Error: Trie pointer is null.\n");
    exit(EXIT_FAILURE);
  }
  if (!vocab_file) {
    fprintf(stderr, "Error: vocab_file pointer is null.\n");
    exit(EXIT_FAILURE);
  }

  FILE* f = fopen(vocab_file, "w");
  if (!f) { perror(vocab_file); return; }
  char buf[1024];
  _save(root, f, buf, 0);
  fclose(f);
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