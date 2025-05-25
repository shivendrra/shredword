#include <stdio.h>
#include <stdlib.h>
#include "../inc/hash.h"
#include "../inc/heap.h"
#include "histogram.h"
#include "bpe.h"

// helper function to compute neighbouring frequencies
uint64_t recompute_freq(PairKey key, Info* info, Trainer* trainer) {
  uint64_t freq = 0;
  size_t write_idx = 0;

  for (size_t i = 0; i < info->pos_size; ++i) {
    WordPos wp = info->positions[i];
    Symbol* s = wp.pos;

    if (s && s->next &&
        s->id == key.first &&
        s->next->id == key.second) {
      freq += trainer->corpus.word_counts[wp.word_index];
      info->positions[write_idx++] = wp;
    }
  }
  info->pos_size = write_idx;
  return freq;
}

Trainer* create_trainer(const BPEConfig* config) {
  if (config == NULL) {
    fprintf(stderr, "[ERROR]\t Config pointer is NULL\n");
    exit(EXIT_FAILURE);
  }
  Trainer* trainer = (Trainer*)malloc(sizeof(Trainer));
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t Couldn't allocate Memory to Trainer\n");
    exit(EXIT_FAILURE);
  }
  trainer->config = *config;
  // defaulting character coverage value
  if (trainer->config.character_coverage <= 0.0 || trainer->config.character_coverage >= 1.0) {
    trainer->config.character_coverage = 0.995;
  }
  // defaulting min pair freq value
  if (trainer->config.min_pair_freq == 0) {
    trainer->config.min_pair_freq = 100;
  }
  trainer->num_merges = 0;
  trainer->merge_ops = (PairKey*)malloc(sizeof(PairKey) * trainer->config.target_vocab_size);
  heap_init(&trainer->heap, MIN_HEAP_SIZE);   // initialized heap
  printf("[INFO]\t BPE trainer initialized. Heap initialized successfully.\n");
  return trainer;
}

void bpe_trainer_destroy(Trainer* trainer) {
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t No Trainer pointer found to destroy!\n");
    exit(EXIT_FAILURE);
  }
  // freeing corpus arrays (if loaded)
  free(trainer->corpus.words);
  free(trainer->corpus.word_counts);
  heap_free(&trainer->heap);  // free heap storage
  free(trainer);    // finally, free the trainer struct itself
}

/**
  @brief Reads a text file line by line and counts the frequency of each unique token,
        where tokens are separated by tabs, carriage returns, or newlines.
  * This approach uses strtok to split each line in-place, which is memory efficient
  * because it does not require allocating additional memory for each token.
  * By modifying the original line buffer, we avoid unnecessary string copies,
  * making it well-suited for processing large files or lines with many tokens.
  @param input_path [in] Path to the input file to be processed.
  @return void
*/
int bpe_load_corpus(Trainer* trainer, const char* input_path) {
  if (!trainer || !input_path) {
    fprintf(stderr, "[ERROR]\t NULL trainer & input path pointers\n");
    exit(EXIT_FAILURE);
  }
  StrMap freq_map;
  strmap_init(&freq_map, INITIAL_STR_BUFFER);
  FILE* fp = fopen(input_path, "r");
  if (!fp) {
    fprintf(stderr, "[ERROR]\t Couldn't open the file\n");
    exit(EXIT_FAILURE);
  }
  char line[INITIAL_STR_BUFFER];
  while (fgets(line, sizeof(line), fp)) {
    char* tok = strtok(line, "\t\r\n");
    while (tok) {
      strmap_increment(&freq_map, tok);
      tok = strtok(NULL, "\t\r\n");
    }
  }
  fclose(fp);

  // building character histogram
  StrMap char_map;
  strmap_init(&char_map, INITIAL_VOCAB_SIZE);
  strmap_iter(&char_map, char_hist, &char_map);

  // collecting & sorting CharCount
  CharCount* counts = (CharCount*)malloc(INITIAL_VOCAB_SIZE * sizeof(CharCount));
  if (!counts) {
    fprintf(stderr, "[ERROR]\t Failed allocation of character counts\n");
    exit(EXIT_FAILURE);
  }
  CharCountCtx ctx = {counts, 0};
  strmap_iter(&char_map, collect_char, &ctx);
  size_t c = ctx.idx;
  qsort(counts, c, sizeof(CharCount), charcount_cmp);

  // determining the kept characters
  size_t keep = (size_t)(c * trainer->config.character_coverage);
  bool keep_char[INITIAL_VOCAB_SIZE] = {0};
  for (size_t i = 0; i < keep; i++) {
    keep_char[(unsigned char)counts[i].c] = true;
  }
  free(counts);
  strmap_free(&char_map);
  
  // counting unique tokens
  size_t N = 0;
  strmap_iter(&freq_map, [](const char* k, uint64_t v, void* u){
    (*(size_t*)u)++;
  }, &N);
  trainer->corpus.vocab_size = N;
  trainer->corpus.words = (Symbol**)malloc(N * sizeof(Symbol*));
  trainer->corpus.word_counts = (uint64_t*)malloc(N * sizeof(uint64_t));

  // populating symbol chains, mapping rare chars to UNK
  size_t idx = 0;
  BuildCtx c_btx = { trainer, &idx, keep_char };
  strmap_iter(&freq_map, build_symbol_cb, &c_btx);

  strmap_free(&freq_map);
  bimap_init(&trainer->bigram_map, MIN_HEAP_SIZE);
  return 0;
}

void bpe_init(Trainer* trainer) {
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t NULL trainer pointer\n");
    exit(EXIT_FAILURE);
  }
  // bimap re-init
  bimap_free(&trainer->bigram_map);
  bimap_init(&trainer->bigram_map, MIN_HEAP_SIZE);

  // heap re-init
  heap_free(&trainer->heap);
  heap_init(&trainer->heap, MIN_HEAP_SIZE);

  bpe_count_bigrams(trainer);
}

void bpe_count_bigrams(Trainer* trainer) {
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t NULL trainer pointer\n");
    exit(EXIT_FAILURE);
  }
  size_t v = trainer->corpus.vocab_size;
  uint64_t min_freq = trainer->config.min_pair_freq;
  size_t total_pairs = 0;

  for (size_t wi = 0; wi < v; wi++) {
    Symbol* s = trainer->corpus.words[wi];
    uint64_t  wcount = trainer->corpus.word_counts[wi];

    while (s && s->next) {
      total_pairs++;
      PairKey key = { s->id, s->next->id };
      Info* info = bimap_get(&trainer->bigram_map, key);

      if (info->pos_size == info->pos_capacity) {
        size_t new_cap = info->pos_capacity ? info->pos_capacity * 2 : 4;
        info->positions = (WordPos*)realloc(info->positions, new_cap * sizeof(WordPos));
        info->pos_capacity = new_cap;
      }
      info->positions[info->pos_size++] = (WordPos){ wi, s };
      info->freq += wcount;
      if (info->freq >= min_freq) {
        heap_push(&trainer->heap, key, info->freq, info->version);
      }
      s = s->next;
    }
  }
  printf("[INFO]\t Counted all bigrams: %zu occurrences (skipping freq<%llu)\n", total_pairs, (unsigned long long)min_freq);
}

int bpe_merge_batch(Trainer* trainer, int batch_size) {
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t Trainer pointer is NULL!\n");
    exit(EXIT_FAILURE);
  }
  if (heap_empty(&trainer->heap)) return -1;

  int merges_done = 0;
  while (merges_done < batch_size && !heap_empty(&trainer->heap)) {
    HeapEntry top = heap_pop(&trainer->heap);
    PairKey key = top.key;

    Info* info = bimap_get(&trainer->bigram_map, key);
    uint64_t freq = recompute_freq(key, info, trainer);

    if (freq < trainer->config.min_pair_freq) continue;

    int32_t new_id = (int32_t)(INITIAL_VOCAB_SIZE + trainer->num_merges);
    printf("[MERGE]\t Merging (%d,%d) freq=%llu -> new_id=%d\n", key.first, key.second, (unsigned long long)freq, new_id);

    if (trainer->num_merges < trainer->config.target_vocab_size)
      trainer->merge_ops[trainer->num_merges] = key;

    WordPos* pts = info->positions;
    size_t occs = info->pos_size;

    for (size_t i = 0; i < occs; ++i) {
      WordPos wp = pts[i];
      Symbol* a = wp.pos;
      Symbol* b = a->next;
      uint64_t wc = trainer->corpus.word_counts[wp.word_index];

      if (!b || a->id != key.first || b->id != key.second) continue;

      // merge in-place
      a->id = new_id;
      a->next = b->next;
      if (b->next) b->next->prev = a;
      free(b);

      // update left bigram
      if (a->prev) {
        PairKey left = { a->prev->id, new_id };
        Info* linfo = bimap_get(&trainer->bigram_map, left);
        linfo->version++;
        linfo->freq = recompute_freq(left, linfo, trainer);
        if (linfo->freq >= trainer->config.min_pair_freq)
          heap_push(&trainer->heap, left, linfo->freq, linfo->version);
      }

      // update right bigram
      if (a->next) {
        PairKey right = { new_id, a->next->id };
        Info* rinfo = bimap_get(&trainer->bigram_map, right);
        rinfo->version++;
        rinfo->freq = recompute_freq(right, rinfo, trainer);
        if (rinfo->freq >= trainer->config.min_pair_freq)
          heap_push(&trainer->heap, right, rinfo->freq, rinfo->version);
      }
    }

    info->pos_size = 0;
    info->freq = 0;
    info->version++;

    trainer->num_merges++;
    merges_done++;
  }

  return merges_done;
}

int bpe_train(Trainer* trainer) {
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t Trainer pointer is NULL!\n");
    exit(EXIT_FAILURE);
  }

  printf("[INFO]\t ----------------------- Starting the training process -----------------------\n");
  bpe_init(trainer);  // reset bigram map and heap
  int total_merges = 0;

  while (total_merges < (int)trainer->config.target_vocab_size) {
    if (heap_empty(&trainer->heap)) break;

    // peek top bigram to decide batch size adaptively
    HeapEntry top = trainer->heap.data[0];  // max-heap root
    uint64_t freq = top.freq;
    int batch_size;

    // adaptive batch size training
    if (freq > 100000) batch_size = 25;
    else if (freq > 50000) batch_size = 15;
    else if (freq > 20000) batch_size = 8;
    else if (freq > 10000) batch_size = 5;
    else batch_size = 2;

    int merged = bpe_merge_batch(trainer, batch_size);
    if (merged <= 0) break;
    total_merges += merged;
  }

  printf("[INFO]\t Training completed successfully. Total merges: %d\n", total_merges);
  return (int)total_merges;
}

void bpe_save(const Trainer* trainer, const char* model_path, const char* vocab_path) {
    if (!trainer) {
    fprintf(stderr, "[ERROR]\t Trainer pointer is NULL!\n");
    exit(EXIT_FAILURE);
  }
  size_t M = trainer->num_merges;
  size_t T = INITIAL_VOCAB_SIZE + M;

  // building token strings
  char** toks = (char**)calloc(T, sizeof(char*));
  for (size_t i = 0; i < INITIAL_VOCAB_SIZE; ++i) {
    toks[i] = (char*)malloc(2);
    toks[i][0] = (char)i; toks[i][1] = '\0';
  }
  for (size_t m = 0; m < M; ++m) {
    PairKey ops = trainer->merge_ops[m];
    size_t id = INITIAL_VOCAB_SIZE + m;
    char *A = toks[ops.first], *B = toks[ops.second];
    size_t aL = strlen(A), bL = strlen(B);
    toks[id] = (char*)malloc(aL + bL + 1);
    memcpy(toks[id], A, aL);
    memcpy(toks[id] + aL, B, bL + 1);
  }

  // count real frequencies
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
    fprintf(vf, "%s %llu\n", toks[i], (unsigned long long)freq[i]);
  }
  fclose(vf);

  // Write merges
  FILE* mf = fopen(model_path, "wb");
  for (size_t m = 0; m < M; ++m) {
    PairKey op = trainer->merge_ops[m];
    fprintf(mf, "%d %d %zu\n", op.first, op.second, INITIAL_VOCAB_SIZE + m);
  }
  fclose(mf);

  // cleanup
  for (size_t i = 0; i < T; ++i) free(toks[i]);
  free(toks);
  free(freq);

  printf("[INFO]\tSaved %zu-token vocab to %s and %zu merges to %s\n", T, vocab_path, M, model_path);
}