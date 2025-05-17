#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include "bpe.h"
#include "trie.h"
#include "hash.h"

struct BpeTrainer {
  BPEConfig config;   // user-settings
  MaxHeap heap;   // bigram merge items
  Corpus corpus;    // word-> countmap & symbol chains
  Info bigram_map;  // bigram -> Info (freq, pair, version)
  size_t next_token_id; // next unused subword ID (starts at initial vocab size)
  size_t initial_vocab_size;
  size_t num_merges;    // how many merges have been applied
  PairKey *merge_ops;     // array of length num_merges
  char **token_strs;    // maps token ID -> UTF‑8 string
  uint64_t *token_freqs;   // maps token ID -> frequency
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
      strmap_increment(&freq_map, tok);
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
  if(!trainer) {
    fprintf(stderr, "Trainer pointer is NULL!\n");
    exit(EXIT_FAILURE);
  }
  if (heap_empty(&trainer->heap)) return -1;

  HeapEntry top = heap_pop(&trainer->heap);
  PairKey key = top.key;
  int32_t new_id;

  Info* info = bigram_map_get(&bigram_map, key);
  size_t occur_count = info->pos_size;
  for (size_t i = 0; i < occur_count; ++i) {
    wordPos wpos = info->positions[i];
    Symbol* sym1 = wpos.pos;
    Symbol* sym2 = sym1->next;

    // splice out sym2, update sym1 to new_Id
    sym1->id = new_id;
    sym1->next = sym2->next;
    if (sym2->next) sym2->next->prev = sym1;
    free(sym2);

    if (sym1->prev) {
      PairKey left_key = { sym1->prev->id, new_id};
      Info* li = bigram_map_get(&bigram_map, left_key);
      li->version++;

      // recompute total frequencies from occurances
      li->freq = 0;
      for (size_t j = 0; j < li->pos_size; i++) {
        li->freq += trainer->corpus.word_counts[ li->positions[j].word_index];
      }
      heap_push(&trainer->heap, left_key, li->freq, li->version);
    }

    if (sym1->next) {
      PairKey right_key = {new_id, sym1->next->id};
      Info* ri = bigram_map_get(&bigram_map, right_key);
      ri->version++;
      ri->freq = 0;   // recompute total freq from occurances
      for (size_t j = 0; j < ri->pos_size; i++) {
        ri->freq += trainer->corpus.word_counts[ ri->positions[j].word_index];
      }
      heap_push(&trainer->heap, right_key, ri->freq, ri->version);
    }
  }

  info->pos_size = 0;
  info->freq = 0;
  info->version++;
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
    fprintf(stderr, "Trainer pointer is NULL!\n");
    exit(EXIT_FAILURE);
  }
  bpe_initialize(trainer);  // Prepare the heap and bigram map
  // Merge until we have created target_vocab_size new tokens
  size_t merges = 0;
  while (merges < trainer->config.target_vocab) {
    if (bpe_merge(trainer) != 0) break;
    merges++;
  }

  return (int)merges;
}

/**
  @brief Serialize the final vocabulary and merge list to disk.
 * Writes:
 *  - vocab_path: one token per line, “<token_string> <frequency>\n”
 *  - model_path: merge operations, one per line “<id1> <id2> <new_id>\n”

 * Assumes you have in BpeTrainer:
 *   size_t   num_tokens;
 *   char   **token_strs;    // maps token ID -> UTF-8 string
 *   uint64_t *token_freqs;   // maps token ID -> frequency
 *   size_t   num_merges;
 *   PairKey *merge_ops;      // length num_merges: (id1,id2)
*/
int bpe_save_model(const BpeTrainer *trainer, const char *model_path, const char *vocab_path) {
  if (!trainer || !model_path || !vocab_path) {
    fprintf(stderr, "Trainer or model_path or vocab_path pointers are NULL!\n");
    exit(EXIT_FAILURE);
  }

  // Writing vocabulary file
  FILE *vf = fopen(vocab_path, "w");
  if (!vf) return -2;
  size_t total_tokens = trainer->initial_vocab_size + trainer->num_merges;
  for (size_t id = 0; id < total_tokens; ++id) {
    fprintf(vf, "%s %llu\n", trainer->token_strs[id], (unsigned long long)trainer->token_freqs[id]);
  }
  fclose(vf);

  // Writing merge operations
  FILE *mf = fopen(model_path, "w");
  if (!mf) return -3;
  for (size_t i = 0; i < trainer->num_merges; ++i) {
    PairKey op = trainer->merge_ops[i];
    // new_id for this merge is typically (initial_vocab_size + i)
    fprintf(mf, "%d %d %zu\n", op.first, op.second, trainer->initial_vocab_size + i);
  }
  fclose(mf);

  return 0;
}