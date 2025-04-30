#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "base.h"
#include "inc/khash.h"

KHASH_MAP_INIT_STR(str_int, int)  // Map from char* to int
KHASH_MAP_INIT_STR(pair_int, int) // Map from pair string ("A\0B") to int

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
  free(node);
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

// Read a line, split on U+2581 marker into symbols array
static int split_to_symbols(const char* line, char*** out_symbols) {
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
      char temp[MAX_SYMBOL_LEN];
      strncpy(temp, line + i, len);
      temp[len] = '\0';
      symbols[n++] = strdup(temp);
      i += len;
    }
  }
  *out_symbols = symbols;
  return n;
}

void bpe_learn(const char* train_file, int merge_steps, TrieNode* root) {
  // 1) Read all lines, build corpus of symbol-sequences
  printf("[DEBUG] bpe_learn(): file=\"%s\", steps=%d\n", train_file, merge_steps);
  FILE* f = fopen(train_file,"r");
  if (!f) { perror("[ERROR] fopen train_file"); return; }
  int  capacity    = 1024;
  int  corpus_size = 0;
  char*** seq_syms = (char***)malloc(sizeof(char**) * capacity);
  int*   seq_lens  = (int*)   malloc(sizeof(int)    * capacity);

  char buf[4096];
  while (fgets(buf, sizeof(buf), f)) {
    buf[strcspn(buf, "\r\n")] = '\0';
    if (!buf[0]) continue;
    // growing the size  if needed
    if (corpus_size == capacity) {
      capacity   *= 2;
      seq_syms    = (char***)realloc(seq_syms, sizeof(char**) * capacity);
      seq_lens    = (int*)   realloc(seq_lens, sizeof(int)    * capacity);
    }
    // storinh
    seq_lens[corpus_size] = split_to_symbols(buf, &seq_syms[corpus_size]);
    corpus_size++;
    if (corpus_size % 1000 == 0) printf("[DEBUG] read %d lines\n", corpus_size);
  }
  fclose(f);
  printf("[DEBUG] total lines read: %d\n", corpus_size);

  // 2) BPE merge iterations
  for (int step = 0; step < merge_steps; step++) {
    printf("[DEBUG] counting pairs for step %d\n", step+1);
    // count all adjacent pairs
    khash_t(pair_int)* pc = kh_init(pair_int);
    for (int i = 0; i < corpus_size; i++) {
      for (int j = 0; j +1 < seq_lens[i]; j++) {
      // key = sym[j] + '\t' + sym[j+1]
      char key[MAX_SYMBOL_LEN*2+1];
      snprintf(key, sizeof(key), "%s\t%s", seq_syms[i][j], seq_syms[i][j+1]);
      int ret; khiter_t k = kh_put(pair_int, pc, strdup(key), &ret);
      kh_val(pc,k) += 1;
      }
    }
    // find best_k
    khiter_t best_k = kh_end(pc);
    int best_c = 0;
    for (khiter_t k = kh_begin(pc); k != kh_end(pc); k++) {
      if (!kh_exist(pc, k)) continue;
      if (kh_val(pc, k) > best_c) {
      best_c = kh_val(pc, k);
      best_k = k;
      }
    }
    if (best_k == kh_end(pc)) { kh_destroy(pair_int, pc); break; }
    const char* raw_key = kh_key(pc, best_k);
    
    // split key back
    char* key_copy = strdup(raw_key);
    char* A = strtok(key_copy, "\t"), *B = strtok(NULL, "\t");
    char merged[MAX_SYMBOL_LEN*2];
    snprintf(merged, sizeof(merged), "%s%s", A, B);
    printf("[DEBUG] best pair: %s+%s count=%d\n", A, B, best_c);
    printf("Merge %d: %s+%s (%d)\n", step+1, A, B, best_c);

    // apply merge to corpus
    for (int i = 0; i < corpus_size; i++) {
      char** in = seq_syms[i];
      int L = seq_lens[i];
      char** out = (char**)malloc(sizeof(char*) * (L));
      int m = 0;
      for (int j=0;j<L;j++) {
        if (j+1 < L && strcmp(in[j], A)==0 && strcmp(in[j+1], B)==0) {
          out[m++] = strdup(merged);
          j++; 
        } else {
          out[m++] = strdup(in[j]);
        }
      }

      // freeing old seq
      for (int z = 0; z < L; z++) free(in[z]);
      free(in);
      seq_syms[i] = out;
      seq_lens[i] = m;
    }

    // cleaning-up pair-count map
    for (khiter_t k = kh_begin(pc); k != kh_end(pc); k++)
    if (kh_exist(pc, k)) free((char*)kh_key(pc, k));
    kh_destroy(pair_int, pc);
  }

  // 3) collecting all symbols into trie
  printf("[DEBUG] inserting %d symbols into trie\n", corpus_size * 10);
  for (int i=0; i < corpus_size; i++) {
    for (int j=0; j < seq_lens[i]; j++) {
      trie_insert(root, seq_syms[i][j]);
      free(seq_syms[i][j]);
    }
    free(seq_syms[i]);
  }
}