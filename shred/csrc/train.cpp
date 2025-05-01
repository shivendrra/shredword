#include <cstring>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <stdatomic.h>
#include "train.h"
#include "base.h"
#include "heap.h"
#include "threads.h"
#include "inc/khash.h"

static char* my_strndup(const char* s, size_t n) {
  char* p = (char*)malloc(n+1);
  memcpy(p, s, n);
  p[n] = '\0';
  return p;
}

// Fallback atomics using GCC built-ins
static inline int atomic_fetch_add_int(int *ptr, int val) {
  return __sync_fetch_and_add(ptr, val);
}

static inline void atomic_store_int(int *ptr, int val) {
  // full barrier store
  __sync_lock_test_and_set(ptr, val);
}

static int trie_count_words(TrieNode* node) {
  if (!node) return 0;
  int count = node->terminal ? 1 : 0;
  for (int i = 0; i < NUM_CHARS; ++i) {
    count += trie_count_words(node->children[i]);
  }
  return count;
}

int get_symbol_id(const char* sym) {
  khiter_t k = kh_get(str_int, sym2id, sym);
  if (k != kh_end(sym2id)) return kh_val(sym2id, k);
  // new symbol
  int id = atomic_fetch_add_int(&sym_count, 1);
  if (id + 1 > sym_capacity) {
    int nc = sym_capacity * 2;
    id2sym = (char**)realloc(id2sym, sizeof(char*) * nc);
    sym_capacity = nc;
  }
  id2sym[id] = strdup(sym);
  int ret;
  k = kh_put(str_int, sym2id, strdup(sym), &ret);
  kh_val(sym2id, k) = id;
  return id;
}

void train_vocab_naive(const char* train_file, const char* vocab_file, int vocab_limit) {
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
    if (corpus_size % 100000 == 0) {
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
      char key[256];  // plenty for 2 merged symbols and a tab
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

void train_bpe_fast(const char* train_file, const char* vocab_file, int merge_steps) {
  printf("[FAST BPE] loading & splitting corpus...\n");
  char*** seq_syms;
  int* seq_lens;
  int corpus_size;
  load_and_split(train_file, &seq_syms, &seq_lens, &corpus_size);

  printf("[FAST BPE] initializing symbol tables...\n");
  sym_capacity = 1024;
  id2sym = (char**)malloc(sizeof(char*)*sym_capacity);
  sym2id = kh_init(str_int);
  atomic_store_int(&sym_count, 0);
  
  // 1) parallel map
  initialize_threads(); // initializing threads first
  int num_threads = get_max_threads();
  pthread_t* threads = (pthread_t*)malloc(sizeof(pthread_t) * num_threads);
  ThreadArg* args = (ThreadArg*)malloc(sizeof(ThreadArg) * num_threads);
  for (int t = 0; t < num_threads; t++) {
    args[t] = (ThreadArg){t,(char*)train_file, seq_syms, seq_lens, corpus_size, NULL};
    pthread_create(&threads[t], NULL, thread_count_pairs, &args[t]);
  }
  printf("[DEBUG] corpus_size = %d\n", corpus_size);
  for (int i = 0; i < 10; i++) {
    printf("[DEBUG] line %d symbol count = %d\n", i, seq_lens[i]);
    for (int j = 0; j < seq_lens[i]; j++) {
      if (seq_syms[i][j] == NULL) {
        printf("[ERROR] NULL symbol at seq_syms[%d][%d]\n", i, j);
      } else {
        printf("  %s", seq_syms[i][j]);
      }
    }
    printf("\n");
  }
  khash_t(pair_int)* global_map = kh_init(pair_int);
  for (int t = 0; t < num_threads; t++) {
    pthread_join(threads[t], NULL); // wait for thread t

    // reduce this thread’s local map
    khash_t(pair_int)* m = args[t].local_map;
    for (khiter_t k = kh_begin(m); k != kh_end(m); k++) {
      if (!kh_exist(m, k)) continue;
      const char* pk = kh_key(m, k);
      int v = kh_val(m, k);

      // interpret the 8‐byte key as an integer for logging
      uint64_t raw_pair;
      memcpy(&raw_pair, pk, sizeof(uint64_t));

      // duplicate raw binary key (8 bytes) + null
      char* key_copy = (char*)malloc(sizeof(uint64_t) + 1);
      if (!key_copy) {
        fprintf(stderr, "[ERROR] malloc failed for key_copy in thread %d\n", t);
        exit(EXIT_FAILURE);
      }
      memcpy(key_copy, pk, sizeof(uint64_t));
      key_copy[sizeof(uint64_t)] = '\0';

      int ret;
      khiter_t g = kh_put(pair_int, global_map, key_copy, &ret);
      if (ret < 0) {
        fprintf(stderr, "[ERROR] kh_put failed for key_copy in thread %d\n", t);
        exit(EXIT_FAILURE);
      }
      kh_val(global_map, g) += v;
    }

    // free all keys in this local map
    for (khiter_t k = kh_begin(m); k != kh_end(m); k++) {
      if (!kh_exist(m, k)) continue;
      const char* pk = kh_key(m, k);
      uint64_t raw_pair;
      memcpy(&raw_pair, pk, sizeof(uint64_t));
      free((char*)pk);
    }

    kh_destroy(pair_int, m);
  }
  free(threads);
  free(args);

  // 2) build heap
  MaxHeap heap;
  int initial = kh_size(global_map);
  if (initial < 1) initial = 1;
  heap_init(&heap, initial);
  for (khiter_t k = kh_begin(global_map); k != kh_end(global_map); k++) {
    if (!kh_exist(global_map,k)) continue;
    uint64_t p; memcpy(&p, kh_key(global_map,k), sizeof(p));
    char* key_copy = my_strndup((char*)&p,8);
    heap_push(&heap, key_copy, kh_val(global_map,k));
  }

  // 3) merges (serial for clarity; can parallelize similar to map)
  TrieNode* root = create_node();
  for (int step = 0; step < merge_steps && !heap_empty(&heap); step++) {
    HeapEntry he = heap_pop(&heap);
    uint64_t p;
    memcpy(&p, he.key, sizeof(p));
    
    int A,B;
    unpack_pair(p,&A,&B);
    char *symA = id2sym[A], *symB = id2sym[B];
    size_t lenA = strlen(symA), lenB = strlen(symB);
    char* merged_str = (char*)malloc(lenA + lenB + 1);
    if (!merged_str) {
      perror("malloc merged_str");
      exit(1);
    }
    // build merged symbol
    memcpy(merged_str, symA, lenA);
    memcpy(merged_str + lenA, symB, lenB + 1);  // includes '\0'

    snprintf(merged_str, lenA + lenB + 1, "%s%s", symA, symB);
    int AB = get_symbol_id(merged_str);
    printf("[FAST BPE] Merge %d: %s+%s (%d)\n", step+1, symA, symB, he.freq);
    free(he.key);

    // apply only neighbor updates
    // .... A B C .... --> .... AB C ....
    // remove count of A+B, remove count of B+C, & add count of AB+C
    for (int i = 0; i < corpus_size; i++) {
      int L = seq_lens[i];
      char** seq = seq_syms[i];
    
      // allocate a new sequence with same or fewer tokens
      char** new_seq = (char**)malloc(sizeof(char*) * L);
      int new_len = 0;
    
      for (int j = 0; j < L; j++) {
        if (j + 1 < L && strcmp(seq[j], symA) == 0 && strcmp(seq[j + 1], symB) == 0) {
          // match found → replace A+B with merged
          new_seq[new_len++] = strdup(id2sym[AB]);
          j++;  // skip B
        } else {
          new_seq[new_len++] = strdup(seq[j]);
        }
      }
    
      // cleanup old seq
      for (int j = 0; j < L; j++) free(seq[j]);
      free(seq);
    
      // store new
      seq_syms[i] = new_seq;
      seq_lens[i] = new_len;
    }

    free(merged_str);
  }

  // 4) insert into trie & save
  for (int i = 0; i < sym_count; i++) trie_insert(root, id2sym[i]);
  save_vocab(root, vocab_file);
  free_trie(root);

  // cleanup
  heap_free(&heap);
  for (khiter_t k = kh_begin(global_map); k != kh_end(global_map); k++)
    if (kh_exist(global_map,k)) free((char*)kh_key(global_map,k));
  kh_destroy(pair_int, global_map);
}