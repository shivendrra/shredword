#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "bpe.h"
#include "heap.h"
#include "hash.h"

// callback context for iterating the StrMap
struct load_ctx {
  BpeTrainer* trainer;
  size_t idx;
};

typedef struct {
  BpeTrainer *trainer;
  size_t *idx;
  bool *keep_char;
} BuildCtx;

typedef struct {
  char c;
  uint64_t count;
} CharCount;

typedef struct {
  CharCount *arr;
  size_t idx;
} CharCountCtx;

static void build_symbol_cb(const char* w, uint64_t cnt, void* u) {
  BuildCtx* ctx = (BuildCtx*)u;
  BpeTrainer* trainer = ctx->trainer;
  size_t pos = *(ctx->idx);

  Symbol *head = NULL, *prev = NULL;
  for (const unsigned char* p = (const unsigned char*)w; *p; ++p) {
    Symbol* s = (Symbol*)malloc(sizeof(Symbol));
    int32_t id = ctx->keep_char[*p] ? (int32_t)*p : trainer->config.unk_id;
    s->id = id;
    s->prev = prev;
    s->next = NULL;
    if (prev) prev->next = s;
    else head = s;
    prev = s;
  }

  trainer->corpus.words[pos] = head;
  trainer->corpus.word_counts[pos] = cnt;
  (*(ctx->idx))++;
}

// Callback to build char histogram from each word
static void char_hist_cb(const char* word, uint64_t wcount, void* u) {
  StrMap *cmap = (StrMap*)u;
  for (const unsigned char *p = (const unsigned char*)word; *p; ++p) {
    char tmp[2] = { (char)*p, 0 };
    strmap_increment(cmap, tmp);
  }
}

// Callback to collect CharCount entries into the context
static void collect_char_cb(const char* kc, uint64_t vc, void* u) {
  CharCountCtx *ctx = (CharCountCtx*)u;
  ctx->arr[ctx->idx].c = kc[0];
  ctx->arr[ctx->idx].count = vc;
  ctx->idx++;
}

// qsort comparator for CharCount descending
static int charcount_cmp(const void *a, const void *b) {
  const CharCount *ca = (const CharCount*)a;
  const CharCount *cb = (const CharCount*)b;
  if (cb->count > ca->count) return 1;
  if (cb->count < ca->count) return -1;
  return 0;
}

// --- helper called for each (key, count) --- 
static void load_entry_cb(const char* key, uint64_t val, void* user) {
  struct load_ctx* ctx = (struct load_ctx*)user;
  Symbol *head = NULL, *prev = NULL;
  
  // build linked list of raw bytes
  for (const unsigned char* p = (const unsigned char*)key; *p; ++p) {
    Symbol* s = (Symbol*)malloc(sizeof(Symbol));
    s->id = (int32_t)*p;
    s->prev = prev, s->next = NULL;
    if (prev) prev->next = s;
    else head = s;
    prev = s;
  }
  // store into corpus
  ctx->trainer->corpus.words[ctx->idx] = head;
  ctx->trainer->corpus.word_counts[ctx->idx] = val;
  ctx->idx++;
}

void remove_occurrence(Info* info, size_t word_index, Symbol* pos) {
  for (size_t i = 0; i < info->pos_size; ++i) {
    if (info->positions[i].word_index == word_index && info->positions[i].pos == pos) {
      info->positions[i] = info->positions[--info->pos_size];  // Swap-and-pop
      return;
    }
  }
}

void append_occurrence(Info* info, size_t word_index, Symbol* pos) {
  if (info->pos_size == info->pos_capacity) {
    size_t new_cap = info->pos_capacity ? info->pos_capacity * 2 : 4;
    info->positions = (wordPos*)realloc(info->positions, new_cap * sizeof(wordPos));
    info->pos_capacity = new_cap;
  }
  info->positions[info->pos_size++] = (wordPos){ word_index, pos };
}

uint64_t recompute_freq(PairKey key, Info* info, BpeTrainer* trainer) {
  uint64_t freq = 0;
  size_t write_idx = 0;

  for (size_t i = 0; i < info->pos_size; ++i) {
    wordPos wp = info->positions[i];
    Symbol* s = wp.pos;

    if (s && s->next &&
        s->id == key.first && s->next->id == key.second) {
      freq += trainer->corpus.word_counts[wp.word_index];
      info->positions[write_idx++] = wp;
    }
  }
  info->pos_size = write_idx;
  return freq;
}

/**
 @brief Allocate and initialize a new trainer.
 * @param config Pointer to user-supplied configuration struct.
 * @returns A heap-backed trainer handle, or NULL on failure.
*/
BpeTrainer* bpe_trainer_create(const BPEConfig* config) {
  if (!config) {
    fprintf(stderr, "[ERROR]\t Config fot BPE not provided!\n");
    exit(EXIT_FAILURE);
  }
  // zero‐init entire struct
  BpeTrainer* trainer = (BpeTrainer*)calloc(1, sizeof(BpeTrainer));
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t Pointer allocation failed!\n");
    exit(EXIT_FAILURE);
  }
  trainer->config = *config;    // copy config

  // Default thresholds if user didn't set them
  if (trainer->config.character_coverage <= 0.0 ||
      trainer->config.character_coverage > 1.0) {
    trainer->config.character_coverage = 0.955;  // keep 95.5% chars
  }
  if (trainer->config.min_pair_freq == 0) {
    trainer->config.min_pair_freq = 10;          // only merge pairs seen >= twice
  }

  // Initialize other fields
  trainer->initial_vocab_size = 256;
  trainer->merge_ops = (PairKey*)malloc(sizeof(PairKey) * config->target_vocab);
  trainer->num_merges = 0;
  heap_init(&trainer->heap, MIN_HEAP_SIZE);
  printf("[INFO]\t BPE trainer initialized. Heap initialized successfully.\n");

  return trainer;
}

/**
  @brief Tear down and free all resources in the trainer.
 * @param trainer  Handle previously returned by bpe_trainer_create.
*/
void bpe_trainer_destroy(BpeTrainer* trainer) {
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t No Trainer pointer found to destroy!\n");
    exit(EXIT_FAILURE);
  }
  // freeing corpus arrays (if loaded)
  free(trainer->corpus.words);
  free(trainer->corpus.word_counts);

  // free heap storage
  heap_free(&trainer->heap);

  // finally, free the trainer struct itself
  free(trainer);
}

int bpe_loadCorpus(BpeTrainer* trainer, const char* input_path) {
  if (!trainer || !input_path) {
    fprintf(stderr, "[ERROR]\t NULL trainer or input_path\n");
    exit(EXIT_FAILURE);
  }

  // Build word freq map
  StrMap freq_map;
  strmap_init(&freq_map, 4096);
  FILE *fp = fopen(input_path, "r");
  if (!fp) return -2;
  char line[4096];
  while (fgets(line, sizeof(line), fp)) {
    char *tok = strtok(line, " \t\r\n");
    while (tok) {
      strmap_increment(&freq_map, tok);
      tok = strtok(NULL, " \t\r\n");
    }
  }
  fclose(fp);

  // Build character histogram
  StrMap char_map;
  strmap_init(&char_map, 256);
  strmap_iter(&freq_map, char_hist_cb, &char_map);

  // Collect & sort CharCount
  CharCount* counts = (CharCount*)malloc(256 * sizeof(CharCount));
  if (!counts) {
    fprintf(stderr, "[ERROR]\t OOM allocating character counts\n");
    exit(EXIT_FAILURE);
  }
  CharCountCtx ctx = { counts, 0 };
  strmap_iter(&char_map, collect_char_cb, &ctx);
  size_t C = ctx.idx;
  qsort(counts, C, sizeof(CharCount), charcount_cmp);

  // Determine kept characters
  size_t keep = (size_t)(C * trainer->config.character_coverage);
  bool keep_char[256] = {0};
  for (size_t i = 0; i < keep; ++i) {
    keep_char[(unsigned char)counts[i].c] = true;
  }
  free(counts);
  strmap_free(&char_map);

  // Count unique tokens
  size_t N = 0;
  strmap_iter(&freq_map,
    // reuse simple lambda‐like callback in C
    [](const char* k, uint64_t v, void* u){
      (*(size_t*)u)++;
    },
    &N
  );
  trainer->corpus.vocab_size = N;
  trainer->corpus.words = (Symbol**)malloc(N * sizeof(Symbol*));
  trainer->corpus.word_counts = (uint64_t*)malloc(N * sizeof(uint64_t));

  // Populate symbol chains, mapping rare chars to UNK
  size_t idx = 0;
  BuildCtx c_btx = { trainer, &idx, keep_char };
  strmap_iter(&freq_map, build_symbol_cb, &c_btx);

  strmap_free(&freq_map);
  bimap_init(&bigram_map, 4096);
  return 0;
}

// --- Return the current version for a bigram key. Used by heap_pop to skip stale entries ---
uint32_t bpe_get_current_version(PairKey key) {
  return bimap_version(&bigram_map, key);
}

// --- initializes the bpe trainer & populate the heap with frequencies from
// normalized loaded corpus ---
void bpe_initialize(BpeTrainer* trainer) {
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t Trainer pointer is NULL!\n");
    exit(EXIT_FAILURE);
  }
  // clearing the bigram map first & then initializing
  bimap_free(&bigram_map);
  bimap_init(&bigram_map, MIN_HEAP_SIZE);

  // clearing the heap first & then initializing
  heap_free(&trainer->heap);
  heap_init(&trainer->heap, MIN_HEAP_SIZE);

  // counting all bigrams
  bpe_count_bigrams(trainer);
}

void bpe_count_bigrams(BpeTrainer* trainer) {
  size_t V = trainer->corpus.vocab_size;
  uint64_t min_freq = trainer->config.min_pair_freq;
  size_t total_pairs = 0;

  for (size_t wi = 0; wi < V; ++wi) {
    Symbol *s = trainer->corpus.words[wi];
    uint64_t wcount = trainer->corpus.word_counts[wi];

    while (s && s->next) {
      total_pairs++;
      PairKey key = { s->id, s->next->id };
      Info *info = bimap_get(&bigram_map, key);

      // grow positions
      if (info->pos_size == info->pos_capacity) {
        size_t new_cap = info->pos_capacity ? info->pos_capacity * 2 : 4;
        info->positions = (wordPos*)realloc(info->positions, new_cap * sizeof(wordPos));
        info->pos_capacity = new_cap;
      }
      info->positions[ info->pos_size++ ] = (wordPos){ wi, s };
      info->freq += wcount;

      // Only push if freq >= threshold
      if (info->freq >= min_freq) {
        heap_push(&trainer->heap, key, info->freq, info->version);
      }
      s = s->next;
    }
  }
  printf("[INFO]\t Counted all bigrams: %zu occurrences (skipping freq<%llu)\n", total_pairs, (unsigned long long)min_freq);
}

/**
 * Perform one BPE merge step:
 *  1. Pop the highest‐frequency bigram (skipping stale entries)
 *  2. Merge every occurrence in-place in the Symbol chains
 *  3. For each merged spot, update the two adjacent bigrams:
      - bump their version
      - recompute freq
      - push into the heap
  @returns 0 on success, or −1 if the heap is empty (no more merges)
*/
int bpe_merge(BpeTrainer* trainer) {
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t Trainer pointer is NULL!\n");
    exit(EXIT_FAILURE);
  }
  if (heap_empty(&trainer->heap)) return -1;

  HeapEntry top = heap_pop(&trainer->heap);
  PairKey key = top.key;
  uint64_t freq = top.freq;
  int32_t new_id = (int32_t)(trainer->initial_vocab_size + trainer->num_merges);    // <- make new_id an int32_t to match PairKey
  printf("[MERGE]\t (%d / %d) Merging (%d,%d) freq=%llu -> new_id=%d\n", new_id - trainer->initial_vocab_size, trainer->num_merges, key.first, key.second, (unsigned long long)freq, new_id);

  // Remember the op
  if (trainer->num_merges < trainer->config.target_vocab)
    trainer->merge_ops[trainer->num_merges] = key;

  Info* info = bimap_get(&bigram_map, key);
  info->freq = recompute_freq(key, info, trainer);
  size_t occs = info->pos_size;
  wordPos* pts = info->positions;

  printf("[STEP]\t Before merge: occs=%zu, info->freq=%llu\n", occs, (unsigned long long)info->freq);
  for (size_t i = 0; i < occs; ++i) {
    wordPos wp = pts[i];
    Symbol* a = wp.pos;
    if (!a || !a->next) continue;  // safety
    Symbol* b = a->next;
    if (!b) continue;

    uint64_t wc = trainer->corpus.word_counts[wp.word_index];
    if (!a || !a->next) {
      printf("[SKIP] Invalid pointer at occurrence %zu\n", i);
      continue;
    }
    if (a == a->next) {
      printf("[WARNING] Self-loop detected at word %zu\n", wp.word_index);
      break;
    }

    a->id = new_id;
    a->next = b->next;
    if (b->next) b->next->prev = a;
    free(b);

    // left-context bigram
    if (a->prev) {
      PairKey lk = { a->prev->id, new_id };
      Info* li = bimap_get(&bigram_map, lk);
      // remove_occurrence(li, wp.word_index, a->prev);
      // append_occurrence(li, wp.word_index, a->prev);
      li->freq = recompute_freq(lk, li, trainer);
      li->version++;
      if (li->freq >= trainer->config.min_pair_freq) {
        heap_push(&trainer->heap, lk, li->freq, li->version);
      }
    }
    
    // right-context bigram
    if (a->next) {
      PairKey rk = { new_id, a->next->id };
      Info* ri = bimap_get(&bigram_map, rk);
      // remove_occurrence(ri, wp.word_index, a);
      // append_occurrence(ri, wp.word_index, a);
      ri->freq = recompute_freq(rk, ri, trainer);
      ri->version++;
      if (ri->freq >= trainer->config.min_pair_freq) {
        heap_push(&trainer->heap, rk, ri->freq, ri->version);
      }
    }
  }
  info->pos_size = 0;
  info->freq = 0;
  info->version++;
  trainer->num_merges++;
  return 0;
}

/**
 * Performs one merge and then fully rebuilds the bigram map & heap.
 * This matches “standard” BPE exactly (but is O(N) per merge).
 */
int bpe_merge_full(BpeTrainer* trainer) {
  // 1) Do the normal lazy merge step:
  int ret = bpe_merge(trainer);
  if (ret != 0) return ret;

  // 2) Rebuild bigram map & heap from scratch:
  bimap_free(&bigram_map);
  bimap_init(&bigram_map, MIN_HEAP_SIZE);

  heap_free(&trainer->heap);
  heap_init(&trainer->heap, MIN_HEAP_SIZE);

  bpe_count_bigrams(trainer);
  return 0;
}

/**
 * Run the full training loop:
 *  - initialize bigrams & heap
 *  - repeatedly merge until target_vocab_size reached
  @returns number of merges performed, or -1 on error
*/
int bpe_train(BpeTrainer* trainer) {
  if(!trainer) {
    fprintf(stderr, "[ERROR]\t Trainer pointer is NULL!\n");
    exit(EXIT_FAILURE);
  }
  printf("[INFO]\t ----------------------- Starting the training process -----------------------\n");
  bpe_initialize(trainer);  // Prepare the heap and bigram map
  // Merge until we have created target_vocab_size new tokens
  size_t merges = 0;
  while (merges < trainer->config.target_vocab) {
    if (bpe_merge_full(trainer) != 0) break;
    // if (bpe_merge(trainer) != 0) break;
    merges++;
  }
  printf("[INFO]\t Training completed successfully. Merges: %d\n", (int)merges);
  return (int)merges;
}

/**
  @brief Serialize the final vocabulary and merge list to disk.
 * Writes:
 *  - vocab_path: one token per line, “<token_string> <frequency>\n”
 *  - model_path: merge operations, one per line “<id1> <id2> <new_id>\n”
*/
void bpe_save(const BpeTrainer* trainer, const char* model_path, const char* vocab_path) {
  size_t V0 = trainer->initial_vocab_size;
  size_t M = trainer->num_merges;
  size_t T = V0 + M;

  // Build token strings
  char** toks = (char**)calloc(T, sizeof(char*));
  for (size_t i = 0; i < V0; ++i) {
    toks[i] = (char*)malloc(2);
    toks[i][0] = (char)i; toks[i][1] = '\0';
  }
  for (size_t m = 0; m < M; ++m) {
    PairKey op = trainer->merge_ops[m];
    size_t id = V0 + m;
    char* A = toks[op.first];
    char* B = toks[op.second];
    size_t aL = strlen(A), bL = strlen(B);
    toks[id] = (char*)malloc(aL+bL+1);
    memcpy(toks[id], A, aL);
    memcpy(toks[id]+aL, B, bL+1);
  }

  // Count real frequencies
  uint64_t* freq = (uint64_t*)calloc(T, sizeof(uint64_t));
  for (size_t w = 0; w < trainer->corpus.vocab_size; ++w) {
    uint64_t wc = trainer->corpus.word_counts[w];
    for (Symbol* s = trainer->corpus.words[w]; s; s = s->next) {
      freq[s->id] += wc;
    }
  }

  // Write vocab
  FILE* vf = fopen(vocab_path, "w");
  for (size_t i = 0; i < T; ++i) {
    fprintf(vf, "[%s] \t -%llu\n", toks[i], (unsigned long long)freq[i]);
  }
  fclose(vf);

  // Write merges
  FILE* mf = fopen(model_path, "wb");
  fprintf(mf, "bpe: v1.1\n\n");
  for (size_t m = 0; m < M; ++m) {
    PairKey op = trainer->merge_ops[m];
    fprintf(mf, "%d \t %d \t %zu\n", op.first, op.second, V0 + m);
  }
  fclose(mf);

  // Cleanup
  for (size_t i = 0; i < T; ++i) free(toks[i]);
  free(toks);
  free(freq);

  printf("[INFO]\tSaved %zu-token vocab to %s and %zu merges to %s\n", T, vocab_path, M, model_path);
}