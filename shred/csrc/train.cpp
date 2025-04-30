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

static void save_vocab_recursively(TrieNode* node, FILE* out, unsigned char* buffer, int depth) {
  if (node->terminal) {
    buffer[depth] = '\0';
    fprintf(out, "%s\n", buffer);
  }
  for (int c = 0; c < NUM_CHARS; c++) {
    if (node->children[c]) {
      buffer[depth] = (unsigned char)c;
      save_vocab_recursively(node->children[c], out, buffer, depth + 1);
    }
  }
}

/**
  @brief Train new vocabulary from a pre-normalized text file.
  * Each line should already be normalized and use ▁ (U+2581) to indicate word boundaries.
  * Extracts tokens and builds a trie-based vocabulary, which is then saved.

   * @param train_file Path to the training text file (pre-normalized).
   * @param vocab_file Path to the output vocabulary file.
   * @param vocab_limit Maximum number of unique tokens to extract.
*/
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

void train_vocab(const char* train_file,
  const char* vocab_file,
  int merge_steps)
{
  printf("[DEBUG] train_vocab(): \"%s\" -> \"%s\" (%d steps)\n",
    train_file, vocab_file, merge_steps);

  // 1) Create the trie root
  printf("[DEBUG] Creating trie root...\n");
  TrieNode* root = create_node();

  // 2) Run BPE learning
  printf("[DEBUG] Starting BPE learning...\n");
  bpe_learn(train_file, merge_steps, root);
  printf("[DEBUG] BPE learning done.\n");

  // 3) Save the resulting vocabulary
  printf("[DEBUG] Saving vocab to \"%s\"...\n", vocab_file);
  save_vocab(root, vocab_file);
  printf("[DEBUG] Vocab saved.\n");

  // 4) Clean up
  printf("[DEBUG] Freeing trie...\n");
  free_trie(root);
  printf("[DEBUG] train_vocab() complete.\n");
}