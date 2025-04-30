#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "inc/khash.h"
#include "train.h"
#include "base.h"

KHASH_MAP_INIT_STR(str_int, int)  // Map from char* to int
KHASH_MAP_INIT_STR(pair_int, int) // Map from pair string ("A\0B") to int

static int trie_count_words(TrieNode* node) {
  if (!node) return 0;
  int count = node->terminal ? 1 : 0;
  for (int i = 0; i < NUM_CHARS; ++i) {
    count += trie_count_words(node->children[i]);
  }
  return count;
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

void train_vocab(const char* train_file, const char* vocab_file, int vocab_limit) {
  TrieNode* root = create_node();
  FILE* test = fopen(vocab_file, "r");
  if (test) { fclose(test); load_vocab(root, vocab_file); }

  FILE* fin = fopen(train_file, "r");
  if (!fin) { perror("fopen(train_file)"); return; }

  char line[4096];
  char token[4096];
  int step = 0;

  while (fgets(line, sizeof(line), fin)) {
    line[strcspn(line, "\r\n")] = '\0';
    if (!line[0]) continue;

    const char* norm = line;  // already normalized
    size_t L = strlen(norm), i = 0;
    while (i < L) {
      if (i + 2 < L && (unsigned char)norm[i]==0xE2 && (unsigned char)norm[i+1]==0x96 && (unsigned char)norm[i+2]==0x81) {
        i += 3;
        continue;
      }
      size_t start = i;
      while (i < L && !(i+2 < L && (unsigned char)norm[i]==0xE2 && (unsigned char)norm[i+1]==0x96 && (unsigned char)norm[i+2]==0x81)) {
        i++;
      }
      size_t len = i - start;
      if (len > 0 && len < sizeof(token)) {
        memcpy(token, norm + start, len);
        token[len] = '\0';
        trie_insert(root, token);

        int current_vocab_size = trie_count_words(root);
        step++;
        if (step % 500 == 0 || current_vocab_size == vocab_limit) {
          printf("[step %d] Current vocab size: %d\n", step, current_vocab_size);
        }
        if (current_vocab_size >= vocab_limit) {
          printf("Reached vocab limit of %d tokens.\n", vocab_limit);
          goto end_training;
        }
      }
    }
  }

end_training:
  fclose(fin);
  save_vocab(root, vocab_file);
  free_trie(root);
}

/**
  @brief Train new vocabulary from a pre-normalized text file.
  * Each line should already be normalized and use ▁ (U+2581) to indicate word boundaries.
  * Extract raw tokens and symbols
  * Trains a new vocab using the extracted symbols & characters using BPE

   * @param train_file Path to the training text file (pre-normalized).
   * @param vocab_file Path to the output vocabulary file.
   * @param vocab_limit Maximum number of unique tokens to extract.
*/
void train_vocab_bpe(const char* train_file, const char* vocab_file, int merge_steps) {
  printf("[DEBUG] train_vocab(): \"%s\" -> \"%s\" (%d steps)\n", train_file, vocab_file, merge_steps);

  TrieNode* root = create_node();

  FILE* f = fopen(train_file,"r");
  if (!f) { perror("[ERROR] fopen train_file"); return; }
  int capacity = 1024, corpus_size = 0;
  char*** seq_syms = (char***)malloc(sizeof(char**) * capacity);
  int* seq_lens = (int*)malloc(sizeof(int) * capacity);

  char buf[4096];
  while (fgets(buf, sizeof(buf), f)) {
    buf[strcspn(buf, "\r\n")] = '\0';
    if (!buf[0]) continue;
    // growing the size  if needed
    if (corpus_size == capacity) {
      capacity *= 2;
      seq_syms = (char***)realloc(seq_syms, sizeof(char**) * capacity);
      seq_lens = (int*)realloc(seq_lens, sizeof(int) * capacity);
    }
    // storinh
    seq_lens[corpus_size] = split_to_symbols(buf, &seq_syms[corpus_size]);
    corpus_size++;
    if (corpus_size % 10000 == 0) printf("[DEBUG] read %d lines\n", corpus_size);
  }
  fclose(f);
  printf("[DEBUG] total lines read: %d\n", corpus_size);
  
  printf("[DEBUG] Starting BPE learning...\n");
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
      for (int j = 0; j < L; j++) {
        if (j + 1 < L && strcmp(in[j], A)==0 && strcmp(in[j + 1], B)==0) {
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
  printf("[DEBUG] inserting %d symbols into trie\n", corpus_size * 10);
  for (int i = 0; i < corpus_size; i++) {
    for (int j = 0; j < seq_lens[i]; j++) {
      trie_insert(root, seq_syms[i][j]);
      free(seq_syms[i][j]);
    }
    free(seq_syms[i]);
  }
  printf("[DEBUG] BPE learning done.\n");

  printf("[DEBUG] Saving vocab to \"%s\"...\n", vocab_file);
  save_vocab(root, vocab_file);
  printf("[DEBUG] Vocab saved.\n");

  printf("[DEBUG] Freeing trie...\n");
  free_trie(root);
  printf("[DEBUG] train_vocab() complete.\n");
}