#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "bpe.h"
#include "trie.h"
#include "inc/chash.h"

struct BpeTrainer {
  BPEConfig config;   // user-settings
  MaxHeap heap;   // bigram merge items
  Corpus corpus;    // word-> countmap & symbol chains
};

// callback context for iterating the StrMap
struct load_ctx {
  BpeTrainer* trainer;
  size_t idx;
};

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

/**
 @brief Allocate and initialize a new trainer.
 * @param config Pointer to user-supplied configuration struct.
 * @returns A heap-backed trainer handle, or NULL on failure.
*/
BpeTrainer* bpe_trainer_create(const BPEConfig* config) {
  if (!config) {
    fprintf(stderr, "Config fot BPE not provided!\n");
    exit(EXIT_FAILURE);
  }
  BpeTrainer* trainer = (BpeTrainer*)malloc(sizeof(BpeTrainer));
  if (!trainer) {
    fprintf(stderr, "Pointer allocation failed!\n");
    exit(EXIT_FAILURE);
  }

  memcpy(&trainer->config, config, sizeof(*config));  // copying config to trainer
  trainer->corpus.words = NULL;
  trainer->corpus.word_counts = NULL;
  trainer->corpus.vocab_size = 0;

  heap_init(&trainer->heap, MIN_HEAP_SIZE);
  return trainer;
}

/**
  @brief Tear down and free all resources in the trainer.
 * @param trainer  Handle previously returned by bpe_trainer_create.
*/
void bpe_trainer_destroy(BpeTrainer* trainer) {
  if (!trainer) {
    fprintf(stderr, "No Trainer pointer found to destroy!\n");
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
    fprintf(stderr, "Trainer pointer or Input path pointer is NULL!\n");
    exit(EXIT_FAILURE);
  };

  // 1) building temporary string->count map
  StrMap freq_map;
  strmap_init(&freq_map, 4096);

  FILE* fp = fopen(input_path, "r");
  if (!fp) return -2;

  char line[4096];
  while (fgets(line, sizeof(line), fp)) {
    char* tok = strtok(line, " \t\r\n");
    while (tok) {
      strmap_inc(&freq_map, tok);
      tok = strtok(NULL, " \t\r\n");
    }
  }
  fclose(fp);

  // 2) count unique tokens
  size_t N = 0;
  strmap_iter(&freq_map, 
    [](const char* key, uint64_t val, void* user){
      size_t* cnt = (size_t*)user;
      (*cnt)++;
    }, &N);

  // 3) allocating trainer->corpus arrays
  trainer->corpus.words = (Symbol**)(sizeof(Symbol*) * N);
  trainer->corpus.word_counts = (uint64_t*)malloc(sizeof(uint64_t) * N);
  trainer->corpus.vocab_size  = N;

  // 4) populating Symbol lists & counts
  size_t idx = 0;
  struct load_ctx ctx = { trainer, 0 };
  strmap_iter(&freq_map, load_entry_cb, &ctx);

  // 5) Clean up temporary map
  strmap_free(&freq_map);

  // 6) Initialize global bigram map
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
    fprintf(stderr, "Trainer pointer is NULL!\n");
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
  if (!trainer) {
    fprintf(stderr, "Trainer pointer is NULL!\n");
    exit(EXIT_FAILURE);
  }

  size_t V = trainer->corpus.vocab_size;
  for (size_t wi = 0; wi < V; ++wi) {
    Symbol* s = trainer->corpus.words[wi];
    uint64_t wcount = trainer->corpus.word_counts[wi];

    while(s && s->next) {
      PairKey key = { s->id, s->next->id };
      Info *info = bigram_map_get(&bigram_map, key);
      
      // grow positions array if needed
      if (info->pos_size == info->pos_capacity) {
        size_t new_capactiy = info->pos_capacity ? info->pos_capacity * 2 : 4;
        info->positions = (wordPos*)realloc(info->positions, new_capactiy * sizeof(wordPos));
        info->pos_capacity = new_capactiy;
      }

      // record this occurances
      info->positions[info->pos_size].word_index = wi;
      info->positions[info->pos_size].pos = s;
      info->pos_size++;
      info->freq += wcount; // accumulate the freq
      heap_push(&trainer->heap, key, info->freq, info->version);  // pusing the item into heap
      s = s->next;   // proceed
    }
  }
}

// --- merges the max frequencey pairs, updates only the neighbours & returns the merged idx ---
int bpe_merge(BpeTrainer* trainer) {
  if(!trainer) {
    fprintf(stderr, "Trainer pointer is NULL!\n");
    exit(EXIT_FAILURE);
  }
}