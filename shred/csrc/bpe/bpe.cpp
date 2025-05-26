#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../inc/hash.h"
#include "../inc/heap.h"
#include "histogram.h"
#include "bpe.h"

// Simple hash table for tracking frequency changes during merges
typedef struct FreqChange {
  uint64_t pair_hash;
  int64_t delta;
  struct FreqChange* next;
} FreqChange;

// Simple hash table for frequency changes
#define FREQ_CHANGE_BUCKETS 1024

typedef struct FreqChangeMap {
  FreqChange* buckets[FREQ_CHANGE_BUCKETS];
} FreqChangeMap;

static void freq_change_init(FreqChangeMap* map) {
  for (int i = 0; i < FREQ_CHANGE_BUCKETS; i++) {
    map->buckets[i] = NULL;
  }
}

static void freq_change_add(FreqChangeMap* map, uint64_t pair_hash, int64_t delta) {
  size_t bucket = pair_hash % FREQ_CHANGE_BUCKETS;
  
  // Check if entry already exists
  for (FreqChange* fc = map->buckets[bucket]; fc; fc = fc->next) {
    if (fc->pair_hash == pair_hash) {
      fc->delta += delta;
      return;
    }
  }
  
  // Create new entry
  FreqChange* new_fc = (FreqChange*)malloc(sizeof(FreqChange));
  new_fc->pair_hash = pair_hash;
  new_fc->delta = delta;
  new_fc->next = map->buckets[bucket];
  map->buckets[bucket] = new_fc;
}

static void freq_change_free(FreqChangeMap* map) {
  for (int i = 0; i < FREQ_CHANGE_BUCKETS; i++) {
    FreqChange* fc = map->buckets[i];
    while (fc) {
      FreqChange* next = fc->next;
      free(fc);
      fc = next;
    }
    map->buckets[i] = NULL;
  }
}

/**
 @brief Recomputes the frequency of a given bigram in the current corpus.
 *
 * This function scans all words in the corpus and sums the frequency of a specific bigram 
 * (represented by `key.first` and `key.second`) by counting how many times it appears 
 * across all symbol sequences, weighted by the frequency of the word in which it appears.
 *
 * Deleted symbols (marked via `s->deleted`) are skipped to ensure only live bigrams are counted.
 * This helps maintain accurate frequency statistics during merge steps when the symbol list
 * is modified in-place.
 *
 @param key     The bigram to count, defined by its `first` and `second` symbol IDs.
 @param info    Pointer to the Info struct associated with this bigram (not modified here).
 @param trainer Pointer to the Trainer object that contains the corpus.
 @return The total frequency of the bigram across the corpus.
 *
 @note Bigrams containing unknown token IDs (`unk_id`) are ignored and return zero,
 *       as they are considered invalid for merging.
*/
uint64_t recompute_freq(PairKey key, Info* info, Trainer* trainer) {
  if (key.first == trainer->config.unk_id || key.second == trainer->config.unk_id) return 0;

  uint64_t freq = 0;
  size_t vocab_size = trainer->corpus.vocab_size;

  for (size_t wi = 0; wi < vocab_size; ++wi) {
    Symbol* s = trainer->corpus.words[wi];
    uint64_t count = trainer->corpus.word_counts[wi];
        
    while (s && s->next) {
      if (!s->deleted && !s->next->deleted &&
        s->id == key.first && s->next->id == key.second) {
        freq += count;
      }
      s = s->next;
    }
  }
  return freq;
}

/**
 @brief Allocates and initializes a new BPE trainer instance with user-specified or default configuration.
 *
 * This function sets up a new `Trainer` object for BPE training by copying the given configuration,
 * initializing internal data structures such as the heap and merge operations array,
 * and applying defaults for missing parameters such as `character_coverage` and `min_pair_freq`.
 *
 @param config Pointer to the `BPEConfig` struct containing user-supplied settings.
 @return Pointer to an initialized `Trainer` struct.
 *
 @note Exits the program on allocation failure or if the config pointer is NULL.
*/
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
    trainer->config.min_pair_freq = MIN_PAIR_FREQ;
  }
  trainer->num_merges = 0;
  trainer->merge_ops = (PairKey*)malloc(sizeof(PairKey) * trainer->config.target_vocab_size);
  heap_init(&trainer->heap, MIN_HEAP_SIZE);
  printf("[INFO]\t BPE trainer initialized. Heap initialized successfully.\n");
  return trainer;
}

/**
 @brief Frees all resources associated with a BPE trainer instance.
 *
 * This function deallocates the internal corpus arrays (word symbols and counts),
 * frees the internal heap structure, and finally frees the trainer object itself.
 *
 @param trainer Pointer to the `Trainer` struct to be destroyed.
 *
 @note Exits the program if the passed trainer pointer is NULL.
*/
void bpe_trainer_destroy(Trainer* trainer) {
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t No Trainer pointer found to destroy!\n");
    exit(EXIT_FAILURE);
  }
  // freeing corpus arrays (if loaded)
  free(trainer->corpus.words);
  free(trainer->corpus.word_counts);
  heap_free(&trainer->heap);
  free(trainer);
}

/**
 @brief Reinitializes the trainer's heap and bigram map, and performs the initial bigram counting pass.
 *
 * This function clears and reinitializes the `heap` and `bigram_map` fields in the `Trainer` struct
 * to ensure a clean state before training or after a reset. It then calls `bpe_count_bigrams()`
 * to repopulate the heap with valid bigrams based on current corpus state.
 *
 @param trainer Pointer to the `Trainer` struct.
 *
 @note Exits the program if the trainer pointer is NULL.
*/
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

/**
 @brief Loads the training corpus from a text file and constructs the initial vocabulary and character histogram.
 *
 * This function performs the following steps:
 *  1. Reads the file line by line and splits it into tokens using tab, newline, space, and carriage return as delimiters.
 *  2. Builds a frequency map of unique words using a `StrMap`.
 *  3. Constructs a histogram of character frequencies across all words and determines which characters to retain
 *     based on the `character_coverage` parameter in the configuration.
 *  4. Initializes the corpus vocabulary:
 *     - Assigns known characters their byte ID.
 *     - Maps all rare or unknown characters to the special UNK token.
 *     - Stores tokenized words as linked lists of `Symbol` nodes.
 *  5. Allocates and sets up the initial `bigram_map`.
 *
 @param trainer Pointer to the `Trainer` object being initialized.
 @param input_path Path to the input corpus file.
 @return 0 on success, -1 on failure (e.g., file not found, memory allocation failure).
 *
 @note This is a prerequisite step before BPE training can start. The function is optimized
 *       for handling large files and long lines using dynamic buffer resizing.
*/
int bpe_load_corpus(Trainer* trainer, const char* input_path) {
  if (!trainer || !input_path) {
    fprintf(stderr, "[ERROR]\t NULL trainer or input path pointers\n");
    return -1;
  }
  StrMap freq_map;
  strmap_init(&freq_map, INITIAL_STR_BUFFER);
  FILE* fp = fopen(input_path, "r");
  if (!fp) {
    fprintf(stderr, "[ERROR]\t Couldn't open file: %s\n", input_path);
    strmap_free(&freq_map);
    return -1;
  }

  char* line = (char*)malloc(INITIAL_STR_BUFFER);
  if (!line) {
    fprintf(stderr, "[ERROR]\t Memory allocation failed\n");
    fclose(fp);
    strmap_free(&freq_map);
    return -1;
  }
  size_t line_cap = INITIAL_STR_BUFFER;
  while (fgets(line, line_cap, fp)) {
    size_t len = strlen(line);
    while (len == line_cap - 1 && line[len-1] != '\n') {
      line_cap *= 2;
      char* new_line = (char*)realloc(line, line_cap);
      if (!new_line) {
        fprintf(stderr, "[ERROR]\t Memory reallocation failed\n");
        free(line);
        fclose(fp);
        strmap_free(&freq_map);
        return -1;
      }
      line = new_line;
      if (!fgets(line + len, line_cap - len, fp)) break;
      len = strlen(line);
    }
    if (len > 0 && line[len-1] == '\n') { line[len-1] = '\0'; }
    char* tok = strtok(line, "\t\r\n ");
    while (tok) {
      strmap_increment(&freq_map, tok);
      tok = strtok(NULL, "\t\r\n ");
    }
  }
  free(line);
  fclose(fp);

  // building character histogram
  StrMap char_map;
  strmap_init(&char_map, INITIAL_VOCAB_SIZE);
  strmap_iter(&freq_map, char_hist, &char_map);
  
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
  printf("[DEBUG]\t Character histogram built with %zu unique characters.\n", c);
  
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
  strmap_iter(&freq_map, [](const char* k, uint64_t v, void* u){(*(size_t*)u)++;}, &N);
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

/**
 @brief Scans the current corpus and counts the frequency of all valid bigrams, populating the heap for training.
 *
 * This function processes every pair of consecutive non-deleted symbols across all words in the corpus,
 * counting how often each bigram (pair of IDs) occurs. It maintains a hash map (`bigram_map`) for fast lookup.
 * It then pushes all bigrams whose frequency is greater than or equal to `min_pair_freq` into the trainer's max-heap.
 *
 * The function performs two passes:
 *  - First pass: Scans the entire corpus and accumulates frequencies in the `bigram_map`.
 *  - Second pass: Pushes qualifying pairs into the heap based on their frequency threshold.
 *
 @param trainer Pointer to the initialized `Trainer` containing the corpus.
 *
 @note This is the core pre-processing step that makes bigram statistics available for the merge loop.
 *       It also logs progress periodically for large datasets (every 10k words).
*/
void bpe_count_bigrams(Trainer* trainer) {
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t NULL trainer pointer\n");
    exit(EXIT_FAILURE);
  }

  size_t v = trainer->corpus.vocab_size;
  uint64_t min_freq = trainer->config.min_pair_freq;
  uint64_t total_pairs = 0;
  size_t unique_pairs = 0;

  printf("[INFO]\t Counting bigrams from %zu words...\n", v);

  // First pass: count all bigram frequencies
  for (size_t wi = 0; wi < v; wi++) {
    Symbol* s = trainer->corpus.words[wi];
    uint64_t wcount = trainer->corpus.word_counts[wi];
    while (s && s->next) {
      if (s->deleted || s->next->deleted || 
        s->id == trainer->config.unk_id || s->next->id == trainer->config.unk_id) {
        s = s->next;
        continue;
      }
        
      PairKey key = { s->id, s->next->id };
      Info* info = bimap_get(&trainer->bigram_map, key);
        
      if (info->freq == 0) {
        unique_pairs++;
        info->version = 0; // Initialize version
      }
        
      info->freq += wcount;
      total_pairs += wcount;  
      s = s->next;
    }
    
    if (wi % 10000 == 0 && wi > 0) {
      printf("[DEBUG]\t Processed %zu/%zu words, found %zu unique pairs\n", wi, v, unique_pairs);
    }
  }

  // Second pass: populate heap with frequent pairs
  size_t heap_entries = 0;
  for (size_t i = 0; i < trainer->bigram_map.nbuckets; i++) {
    for (BIEntry* e = trainer->bigram_map.buckets[i]; e; e = e->next) {
      if (e->info.freq >= min_freq) {
        heap_push(&trainer->heap, e->key, e->info.freq, e->info.version);
        heap_entries++;
      }
    }
  }

  printf("[INFO]\t Counted %llu total bigram occurrences, %zu unique pairs\n", (unsigned long long)total_pairs, unique_pairs);
  printf("[INFO]\t Added %zu pairs to heap (freq >= %llu)\n", heap_entries, (unsigned long long)min_freq);
}

/**
 @brief Perform a batch of BPE merges based on the most frequent bigrams.
 *
 * This function repeatedly pops the most frequent valid bigram from the heap and merges it,
 * updating affected bigrams' frequencies using a hash-based difference tracking mechanism.
 * This avoids expensive rescans or linear recomputations.
 *
 * The function maintains:
 * - Lazy validation: Skips stale heap entries by checking version mismatch.
 * - In-place merges: Symbol chains are modified directly.
 * - Frequency tracking: Uses 64-bit hash keys to track deltas in neighbor frequencies.
 * - Efficient heap updates: Only pushes new or changed bigrams above threshold.
 *
 @param trainer Pointer to the initialized Trainer instance.
 @param batch_size Number of merges to perform in one go.
 @return Number of successful merges performed.
 *
 * Example: Merges ("th", "e") → "the" and updates ("a", "th") → ("a", "the") accordingly.
*/
int bpe_merge_batch(Trainer* trainer, int batch_size) {
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t Trainer pointer is NULL!\n");
    return -1;
  }
  if (heap_empty(&trainer->heap)) {
    printf("[INFO]\t Heap is empty, no more merges possible\n");
    return 0;
  }

  int merges_done = 0;
  int stale_entries = 0;
  uint64_t min_freq = trainer->config.min_pair_freq;
  
  while (merges_done < batch_size && !heap_empty(&trainer->heap)) {
    HeapEntry top = heap_pop(&trainer->heap);
    PairKey key = top.key;

    Info* info = bimap_get(&trainer->bigram_map, key);

    // Skip stale heap entries (version mismatch)
    if (top.version != info->version) {
      stale_entries++;
      continue;
    }

    // Verify frequency is still above threshold
    uint64_t current_freq = info->freq;
    if (current_freq < min_freq) {
      continue;
    }

    // Proceed with merge
    int32_t new_id = INITIAL_VOCAB_SIZE + trainer->num_merges;
    printf("[MERGE]\t Merging (%d,%d) freq=%llu -> new_id=%d (merge %zu)\n", key.first, key.second, (unsigned long long)current_freq, new_id, trainer->num_merges + 1);

    if (trainer->num_merges < trainer->config.target_vocab_size) {
      trainer->merge_ops[trainer->num_merges] = key;
    }

    // Track affected pairs and their frequency changes using our C hash table
    FreqChangeMap freq_changes;
    freq_change_init(&freq_changes);
    uint64_t total_merge_count = 0;
        
    // Perform merges in all words
    for (size_t wi = 0; wi < trainer->corpus.vocab_size; ++wi) {
      Symbol* s = trainer->corpus.words[wi];
      uint64_t word_count = trainer->corpus.word_counts[wi];
      
      while (s && s->next) {
        if (s->deleted || s->next->deleted ||
          s->id != key.first || s->next->id != key.second) {
          s = s->next;
          continue;
        }

        // Count this merge
        total_merge_count += word_count;
        
        // Track frequency changes for neighboring pairs
        // Left neighbor
        if (s->prev && !s->prev->deleted) {
          PairKey old_left = {s->prev->id, s->id};
          PairKey new_left = {s->prev->id, new_id};
          uint64_t old_hash = ((uint64_t)old_left.first << 32) | (uint64_t)old_left.second;
          uint64_t new_hash = ((uint64_t)new_left.first << 32) | (uint64_t)new_left.second;
          freq_change_add(&freq_changes, old_hash, -(int64_t)word_count);
          freq_change_add(&freq_changes, new_hash, (int64_t)word_count);
        }
        
        // Right neighbor  
        if (s->next->next && !s->next->next->deleted) {
          PairKey old_right = {s->next->id, s->next->next->id};
          PairKey new_right = {new_id, s->next->next->id};
          uint64_t old_hash = ((uint64_t)old_right.first << 32) | (uint64_t)old_right.second;
          uint64_t new_hash = ((uint64_t)new_right.first << 32) | (uint64_t)new_right.second;
          freq_change_add(&freq_changes, old_hash, -(int64_t)word_count);
          freq_change_add(&freq_changes, new_hash, (int64_t)word_count);
        }

        // Perform the actual merge
        Symbol* b = s->next;
        s->id = new_id;
        s->next = b->next;
        if (b->next) {
          b->next->prev = s;
        }
        b->deleted = true;
        // Continue from the merged symbol
        // Don't advance s here - let the outer loop handle it
      }
    }

    // Apply frequency changes
    for (int i = 0; i < FREQ_CHANGE_BUCKETS; i++) {
      for (FreqChange* fc = freq_changes.buckets[i]; fc; fc = fc->next) {
        uint64_t pair_hash = fc->pair_hash;
        int64_t delta = fc->delta;
        
        PairKey pk = {(int32_t)(pair_hash >> 32), (int32_t)(pair_hash & 0xFFFFFFFF)};
        
        // Skip if this is the pair we just merged
        if (pk.first == key.first && pk.second == key.second) {
          continue;
        }
        
        Info* pair_info = bimap_get(&trainer->bigram_map, pk);
        // Apply frequency change safely
        if (delta < 0) {
          uint64_t abs_delta = (uint64_t)(-delta);
          if (pair_info->freq >= abs_delta) {
            pair_info->freq -= abs_delta;
          } else {
            pair_info->freq = 0;
          }
        } else {
          pair_info->freq += (uint64_t)delta;
        }

        // Add to heap if frequency meets threshold
        if (pair_info->freq >= min_freq) {
          pair_info->version++;
          heap_push(&trainer->heap, pk, pair_info->freq, pair_info->version);
        }
      }
    }

    // Clean up frequency changes map
    freq_change_free(&freq_changes);

    // Mark the merged pair as processed
    info->freq = 0;
    info->version++;
    trainer->num_merges++;
    merges_done++;
    
    printf("[DEBUG]\t Merged %llu occurrences in corpus\n", (unsigned long long)total_merge_count);
  }
  if (stale_entries > 0) {
    printf("[DEBUG]\t Skipped %d stale heap entries\n", stale_entries);
  }

  return merges_done;
}

/**
 @brief Safely remove all symbols marked as `deleted` from the corpus.
 *
 * This function scans all words in the corpus and frees memory of symbols marked deleted
 * during BPE merges. It also updates the `prev` and `next` pointers to maintain
 * linked list consistency for remaining symbols.
 *
 * Should be called periodically (e.g., every few thousand merges) to avoid memory bloat
 * and dangling pointer references.
 *
 @param trainer Pointer to the Trainer whose corpus contains the symbols.
 *
 * Example: After merging "th" and "e", the original "e" Symbol is marked deleted.
 * This function will safely unlink and free it.
*/
void free_deleted_symbols(Trainer* trainer) {
  if (!trainer) return;
  for (size_t wi = 0; wi < trainer->corpus.vocab_size; ++wi) {
    Symbol* s = trainer->corpus.words[wi];
    Symbol* prev = NULL;
    while (s) {
      if (s->deleted) {
        Symbol* to_free = s;
        if (prev) {
          prev->next = s->next;
        } else {
          trainer->corpus.words[wi] = s->next;
        }
        if (s->next) { s->next->prev = prev; }
        s = s->next;
        free(to_free);
      } else {
        prev = s;
        s = s->next;
      }
    }
  }
}

/**
 @brief Executes the BPE training loop until the target vocabulary size is reached.
 *
 * This function drives the main training process by:
 *  - Initializing the bigram heap and frequency map
 *  - Dynamically determining a batch size for each iteration based on the top bigram frequency
 *  - Performing batch merges via `bpe_merge_batch`
 *  - Periodically freeing deleted symbols to reduce memory footprint
 *
 * The batch size is adaptively chosen to balance merge speed and accuracy.
 * It ensures the training progresses efficiently while still considering
 * changes in frequency distribution due to recent merges.
 *
 * The function stops if:
 *  - The heap is empty (no more eligible merges)
 *  - The target vocabulary size has been reached
 *  - A batch merge results in no actual merges (convergence)
 *
 @param trainer A pointer to the initialized Trainer structure
 @return The total number of merges performed during training
*/
int bpe_train(Trainer* trainer) {
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t Trainer pointer is NULL!\n");
    return -1;
  }
  printf("[INFO]\t Starting BPE training (target vocab size: %zu)\n", trainer->config.target_vocab_size);  
  bpe_init(trainer);
  int total_merges = 0;
  int target_merges = (int)trainer->config.target_vocab_size - INITIAL_VOCAB_SIZE;
  printf("[INFO]\t Need to perform %d merges to reach target vocab size\n", target_merges);

  while (total_merges < target_merges) {
    if (heap_empty(&trainer->heap)) {
      printf("[INFO]\t Heap exhausted, stopping training\n");
      break;
    }

    // Determine batch size based on current heap top frequency
    HeapEntry top = trainer->heap.data[0];
    uint64_t top_freq = top.freq;
    int batch_size;

    // More conservative batch sizing for accuracy
    if (top_freq > 100000) batch_size = 50;
    else if (top_freq > 50000) batch_size = 20;
    else if (top_freq > 20000) batch_size = 10;
    else if (top_freq > 10000) batch_size = 5;
    else if (top_freq > 5000) batch_size = 3;
    else if (top_freq > 2000) batch_size = 2;
    else batch_size = 1;

    // Don't exceed remaining merges
    batch_size = (batch_size > target_merges - total_merges) ? target_merges - total_merges : batch_size;
    printf("[INFO]\t Processing batch of %d merges (completed: %d/%d, heap size: %zu, top freq: %llu)\n", 
           batch_size, total_merges, target_merges, trainer->heap.size, (unsigned long long)top_freq);
    int merged = bpe_merge_batch(trainer, batch_size);
    if (merged <= 0) {
      printf("[WARNING]\t No merges performed, stopping\n");
      break;
    }
    total_merges += merged;

    // Periodic cleanup
    if (total_merges % 100 == 0) {
      printf("[DEBUG]\t Cleaning up deleted symbols after %d merges\n", total_merges);
      free_deleted_symbols(trainer);
    }

    // Progress reporting
    if (total_merges % 50 == 0 || merged < batch_size) {
        printf("[PROGRESS]\t Completed %d/%d merges (%.1f%%)\n", total_merges, target_merges, 
               100.0 * total_merges / target_merges);
    }
  }
  printf("[INFO]\t Final cleanup of deleted symbols\n");
  free_deleted_symbols(trainer);
  printf("[INFO]\t Training completed. Performed %d merges\n", total_merges);
  return total_merges;
}

/**
 @brief Serializes the trained BPE model to disk by saving the final vocabulary and merge operations.
 *
 * This function generates:
 *  - A vocabulary file (`vocab_path`) containing each token string and its frequency
 *  - A merge operations file (`model_path`) listing each merge (left_id, right_id, new_id)
 *
 * The function reconstructs merged token strings by concatenating their component strings,
 * which are tracked using an in-memory array `toks[]` indexed by token ID.
 *
 * Frequencies are computed by iterating over the final corpus and summing the
 * token counts across all words, skipping deleted symbols.
 *
 @param trainer A pointer to the trained BPE model
 @param model_path Output path for writing the merge operations (one per line)
 @param vocab_path Output path for writing the vocabulary (token and frequency per line)
 *
 * Example output:
 * vocab.txt: "th 5000", "the 3200", ...
 * model.txt: "116 104 256"  (id 116 + id 104 = new token id 256)
*/
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
    toks[i][0] = (char)i; 
    toks[i][1] = '\0';
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

  // count actual token frequencies in final corpus
  uint64_t* freq = (uint64_t*)calloc(T, sizeof(uint64_t));
  for (size_t w = 0; w < trainer->corpus.vocab_size; ++w) {
    uint64_t wc = trainer->corpus.word_counts[w];
    for (Symbol* s = trainer->corpus.words[w]; s; s = s->next) {
      if (!s->deleted) {
        freq[s->id] += wc;
      }
    }
  }

  // Write vocabulary with frequencies
  FILE* vf = fopen(vocab_path, "w");
  for (size_t i = 0; i < T; ++i) {
    fprintf(vf, "%s %llu\n", toks[i], (unsigned long long)freq[i]);
  }
  fclose(vf);

  // Write merge operations
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