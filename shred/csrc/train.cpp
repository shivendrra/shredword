#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "inc/khash.h"
#include "train.h"
#include "base.h"
#include "heap.h"

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
  printf("[DEBUG] bpe_learn('%s', %d)\n", train_file, merge_steps);

  // --- Read & split corpus ---
  FILE* f = fopen(train_file, "r");
  if (!f) { perror("fopen train_file"); return; }

  int capacity = 1024;
  int corpus_size = 0;
  char*** seq_syms = (char***)malloc(sizeof(char**) * capacity);
  int* seq_lens = (int*)malloc(sizeof(int) * capacity);
  char buf[4096];

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

  TrieNode* root = create_node();

  // --- Initial pair counting ---
  khash_t(pair_int)* pc = kh_init(pair_int);
  for (int i = 0; i < corpus_size; i++) {
    for (int j = 0; j + 1 < seq_lens[i]; j++) {
      char key[MAX_SYMBOL_LEN*2 + 2];
      snprintf(key, sizeof(key), "%s\t%s", seq_syms[i][j], seq_syms[i][j+1]);
      int ret;
      khiter_t k = kh_put(pair_int, pc, strdup(key), &ret);
      kh_val(pc, k)++;
    }
  }

  // --- Build heap of all pairs ---
  MaxHeap heap;
  int init_cap = (int)kh_size(pc);
  if (init_cap < 1) init_cap = 1;
  heap_init(&heap, init_cap);
  for (khiter_t k = kh_begin(pc); k != kh_end(pc); k++) {
    if (!kh_exist(pc, k)) continue;
    char* raw = strdup(kh_key(pc, k));
    int f = kh_val(pc, k);
    heap_push(&heap, raw, f);
  }

  // --- BPE merge loop ---
  for (int step = 0; step < merge_steps && !heap_empty(&heap); step++) {
    HeapEntry he = heap_pop(&heap);
    char *A = strtok(he.key, "\t"), *B = strtok(NULL, "\t");
    char merged[MAX_SYMBOL_LEN*2];
    
    snprintf(merged, sizeof(merged), "%s%s", A, B);
    printf("[DEBUG] Merge %d: %s+%s (%d)\n", step+1, A, B, he.freq);
    free(he.key);

    // apply merge to each sequence
    for (int i = 0; i < corpus_size; i++) {
      char** in = seq_syms[i];
      int L = seq_lens[i];
      char** out = (char**)malloc(sizeof(char*) * L);
      int m = 0;
      for (int j = 0; j < L; j++) {
        if (j+1 < L && strcmp(in[j], A)==0 && strcmp(in[j+1], B)==0) {
          out[m++] = strdup(merged);
          j++;
        } else {
          out[m++] = strdup(in[j]);
        }
      }
      for (int z = 0; z < L; z++) free(in[z]);
      free(in);
      seq_syms[i] = out;
      seq_lens[i] = m;
    }

    // every 50 merges, rebuild the heap from fresh counts
    if ((step+1) % 50 == 0) {
      printf("[DEBUG] Rebuilding pair heap at merge %d\n", step+1);
      // clear khash & heap
      for (khiter_t kk = kh_begin(pc); kk != kh_end(pc); kk++) {
        if (kh_exist(pc,kk)) free((char*)kh_key(pc,kk));
      }
      kh_clear(pair_int, pc);
      heap_free(&heap);
      int init_cap = (int)kh_size(pc);
      if (init_cap < 1) init_cap = 1;
      heap_init(&heap, init_cap);

      // recount pairs
      for (int i = 0; i < corpus_size; i++) {
        for (int j = 0; j + 1 < seq_lens[i]; j++) {
          char key2[MAX_SYMBOL_LEN*2 + 2];
          snprintf(key2, sizeof(key2), "%s\t%s", seq_syms[i][j], seq_syms[i][j+1]);
          int ret; khiter_t kk = kh_put(pair_int, pc, strdup(key2), &ret);
          kh_val(pc, kk)++;
        }
      }
      // repopulate heap
      for (khiter_t kk = kh_begin(pc); kk != kh_end(pc); kk++) {
        if (!kh_exist(pc,kk)) continue;
        char* raw2 = strdup(kh_key(pc,kk));
        int f2 = kh_val(pc,kk);
        heap_push(&heap, raw2, f2);
      }
    }
  }

  // --- Insert into trie & cleanup ---
  printf("[DEBUG] inserting symbols into trie\n");
  for (int i = 0; i < corpus_size; i++) {
    for (int j = 0; j < seq_lens[i]; j++) {
      trie_insert(root, seq_syms[i][j]);
      free(seq_syms[i][j]);
    }
    free(seq_syms[i]);
  }

  save_vocab(root, vocab_file);  // saving the trained vocabs

  free(seq_syms);
  free(seq_lens);
  free_trie(root);

  // free heap and khash
  heap_free(&heap);
  for (khiter_t k = kh_begin(pc); k != kh_end(pc); k++)
    if (kh_exist(pc,k)) free((char*)kh_key(pc,k));
  kh_destroy(pair_int, pc);

  printf("[DEBUG] bpe learning complete.\n");
}