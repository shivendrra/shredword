#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include "core.h"
#include "normalize.h"
#include "hash.h"
#include "tokenmap.h"

UnigramModel* create_unigram_model(size_t capacity) {
  if (capacity == 0) {
    capacity = MAX_VOCAB_SIZE;
  }
  UnigramModel* model = (UnigramModel*)malloc(sizeof(UnigramModel));
  if (!model) {
    fprintf(stderr, "[ERROR] Couldn't allocate UnigramModel\n");
    exit(EXIT_FAILURE);
  }
  
  model->entries = (UnigramEntry*)calloc(capacity, sizeof(UnigramEntry));
  if (!model->entries) {
    fprintf(stderr, "[ERROR] Couldn't allocate UnigramEntry array\n");
    free(model);
    exit(EXIT_FAILURE);
  }
  
  // Initialize token map
  size_t bucket_count = 1;
  while (bucket_count < capacity / 4) bucket_count <<= 1;
  if (bucket_count < 1024) bucket_count = 1024;
  
  model->token_map.nbuckets = bucket_count;
  model->token_map.buckets = (TokenEntry**)calloc(bucket_count, sizeof(TokenEntry*));
  if (!model->token_map.buckets) {
    fprintf(stderr, "[ERROR] Couldn't allocate token map buckets\n");
    free(model->entries);
    free(model);
    exit(EXIT_FAILURE);
  }
  
  // Initialize hash table
  model->hash_table = (int*)malloc(sizeof(int) * HASH_TABLE_SIZE);
  model->next_in_bucket = (int*)malloc(sizeof(int) * capacity);
  if (!model->hash_table || !model->next_in_bucket) {
    fprintf(stderr, "[ERROR] Couldn't allocate hash tables\n");
    free(model->entries);
    free(model->token_map.buckets);
    free(model->hash_table);
    free(model->next_in_bucket);
    free(model);
    exit(EXIT_FAILURE);
  }
  
  // Initialize hash table with -1 (empty)
  for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
    model->hash_table[i] = -1;
  }
  for (size_t i = 0; i < capacity; i++) {
    model->next_in_bucket[i] = -1;
  }
  
  model->capacity = capacity;
  model->size = 0;
  return model;
}

void free_unigram_model(UnigramModel* model) {
  if (!model) return;
  
  if (model->entries) {
    for (size_t i = 0; i < model->size; i++) {
      if (model->entries[i].subword) {
        free(model->entries[i].subword);
      }
    }
    free(model->entries);
  }

  // Free token map
  token_map_clear(&model->token_map);
  if (model->token_map.buckets) {
    free(model->token_map.buckets);
  }
  
  free(model->hash_table);
  free(model->next_in_bucket);
  free(model);
}

static inline int compare_freq_desc(const void* a, const void* b) {
  const UnigramEntry* entry_a = (const UnigramEntry*)a;
  const UnigramEntry* entry_b = (const UnigramEntry*)b;
  return (entry_a->freq < entry_b->freq) - (entry_a->freq > entry_b->freq);
}

static inline int compare_score_desc(const void* a, const void* b) {
  const UnigramEntry* ua = (const UnigramEntry*)a;
  const UnigramEntry* ub = (const UnigramEntry*)b;
  return (ua->score < ub->score) - (ua->score > ub->score);
}

void initialize_from_hash_vocab(UnigramModel* model, HashVocabEntry* entries, size_t count, StringInterner* interner) {
  if (!model || !entries || count == 0) {
    fprintf(stderr, "[ERROR] Invalid arguments to initialize_from_hash_vocab\n");
    return;
  }
  
  uint64_t total_freq = 0;
  
  // Sort by frequency first
  qsort(entries, count, sizeof(HashVocabEntry), [](const void* a, const void* b) {
    const HashVocabEntry* ea = (const HashVocabEntry*)a;
    const HashVocabEntry* eb = (const HashVocabEntry*)b;
    return (ea->frequency < eb->frequency) - (ea->frequency > eb->frequency);
  });
  
  for (size_t i = 0; i < count && model->size < model->capacity; i++) {
    const char* str = interner_get_string(interner, entries[i].string_id);
    if (!str) continue;
    
    size_t len = interner_get_length(interner, entries[i].string_id);
    char* subword_copy = (char*)malloc(len + 1);
    if (!subword_copy) continue;
    
    memcpy(subword_copy, str, len);
    subword_copy[len] = '\0';
    
    model->entries[model->size].subword = subword_copy;
    model->entries[model->size].freq = entries[i].frequency;
    model->entries[model->size].len = (uint16_t)len;
    model->entries[model->size].hash = model_hash(str, len);
    
    total_freq += entries[i].frequency;
    model->size++;
  }
  
  // Normalize and assign scores as log-probs
  if (total_freq > 0) {
    double log_total = log((double)total_freq);
    for (size_t i = 0; i < model->size; i++) {
      model->entries[i].score = log((double)model->entries[i].freq) - log_total;
    }
  }
  
  rebuild_token_map(model);
  rebuild_hash_table(model);
}

void initialize_from_vocab_table(UnigramModel* model, VocabTable* vocab_table) {
  // Placeholder implementation - needs actual VocabTable structure
  fprintf(stderr, "[WARNING] initialize_from_vocab_table not fully implemented\n");
}

void rebuild_token_map(UnigramModel* model) {
  if (!model) return;
  
  // Clear existing token map
  token_map_clear(&model->token_map);
  
  // Rebuild with current entries
  for (size_t i = 0; i < model->size; i++) {
    if (model->entries[i].subword) {
      token_map_add(&model->token_map, model->entries[i].subword, (int)i);
    }
  }
}

void rebuild_hash_table(UnigramModel* model) {
  if (!model) return;
  
  // Clear hash table
  for (size_t i = 0; i < HASH_TABLE_SIZE; i++) {
    model->hash_table[i] = -1;
  }
  for (size_t i = 0; i < model->capacity; i++) {
    model->next_in_bucket[i] = -1;
  }
  
  // Rebuild hash table
  for (size_t i = 0; i < model->size; i++) {
    if (model->entries[i].subword) {
      uint32_t hash = model->entries[i].hash;
      size_t bucket = hash % HASH_TABLE_SIZE;
      
      model->next_in_bucket[i] = model->hash_table[bucket];
      model->hash_table[bucket] = (int)i;
    }
  }
}

int fast_token_lookup(const UnigramModel* model, const char* token, size_t len) {
  if (!model || !token) return -1;
  
  uint32_t hash = model_hash(token, len);
  size_t bucket = hash % HASH_TABLE_SIZE;
  
  int idx = model->hash_table[bucket];
  while (idx >= 0) {
    if (model->entries[idx].hash == hash && 
        model->entries[idx].len == len &&
        memcmp(model->entries[idx].subword, token, len) == 0) {
      return idx;
    }
    idx = model->next_in_bucket[idx];
  }
  
  return -1;
}

int get_token_id(const UnigramModel* model, const char* token) {
  if (!model || !token) return -1;
  return fast_token_lookup(model, token, strlen(token));
}

void prune_unigram_model(UnigramModel* model, size_t target_vocab_size) {
  if (!model || model->size <= target_vocab_size) return;

  qsort(model->entries, model->size, sizeof(UnigramEntry), compare_score_desc);
  
  for (size_t i = target_vocab_size; i < model->size; ++i) {
    if (model->entries[i].subword) {
      free(model->entries[i].subword);
      model->entries[i].subword = NULL;
    }
  }
  
  model->size = target_vocab_size;
  rebuild_token_map(model);
  rebuild_hash_table(model);
}

void dump_unigram_model(UnigramModel* model) {
  if (!model) return;
  
  for (size_t i = 0; i < model->size; i++) {
    if (model->entries[i].subword) {
      printf("%s %.5f\n", model->entries[i].subword, model->entries[i].score);
    }
  }
}

void save_unigram_model(const UnigramModel* model, const char* filepath) {
  if (!model || !filepath) {
    fprintf(stderr, "[ERROR] Invalid arguments to save_unigram_model\n");
    return;
  }

  FILE* fp = fopen(filepath, "w");
  if (!fp) {
    fprintf(stderr, "[ERROR] Failed to open model file: %s\n", filepath);
    return;
  }

  for (size_t i = 0; i < model->size; i++) {
    if (model->entries[i].subword) {
      fprintf(fp, "%s\t%.8f\n", model->entries[i].subword, model->entries[i].score);
    }  
  }

  fclose(fp);
}

UnigramModel* create_model_from_text(const char** lines, size_t num_lines, size_t max_subword_len, size_t min_frequency, size_t target_vocab_size) {
  if (!lines || num_lines == 0) {
    fprintf(stderr, "[ERROR] Invalid input to create_model_from_text\n");
    return NULL;
  }

  // Create vocabulary builder using hash-based system
  VocabBuilder* builder = vocab_builder_create(target_vocab_size * 2, min_frequency);
  if (!builder) {
    fprintf(stderr, "[ERROR] Failed to create vocabulary builder\n");
    return NULL;
  }

  // Process all lines
  char normalized_line[MAX_LINE];
  for (size_t i = 0; i < num_lines; i++) {
    if (!lines[i]) continue;
    
    // Normalize the line using existing normalize_line function
    int norm_len = normalize_line(lines[i], normalized_line, sizeof(normalized_line));
    if (norm_len > 0) {
      vocab_builder_add_line(builder, normalized_line, max_subword_len);
    }
  }

  // Finalize vocabulary
  HashVocabEntry* entries = NULL;
  size_t count = vocab_builder_finalize(builder, &entries);
  
  if (!entries || count == 0) {
    fprintf(stderr, "[WARNING] No vocabulary entries generated\n");
    vocab_builder_free(builder);
    return NULL;
  }

  // Create and initialize model
  UnigramModel* model = create_unigram_model(target_vocab_size);
  initialize_from_hash_vocab(model, entries, count, builder->interner);

  // Prune if necessary
  if (model->size > target_vocab_size) {
    prune_unigram_model(model, target_vocab_size);
  }

  // Clean up
  free(entries);
  vocab_builder_free(builder);
  
  return model;
}