#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include "unigram.h"

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
  if (model->token_map.buckets) {
    for (size_t i = 0; i < model->token_map.nbuckets; i++) {
      TokenEntry* e = model->token_map.buckets[i];
      while (e) {
        TokenEntry* next = e->next;
        free(e->token);
        free(e);
        e = next;
      }
    }
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

static void token_map_add(TokenMap* map, const char* token, int index) {
  size_t h = 5381;
  for (const unsigned char* s = (const unsigned char*)token; *s; s++) {
    h = ((h << 5) + h) + *s;
  }
  size_t idx = h & (map->nbuckets - 1);
  
  TokenEntry* entry = (TokenEntry*)malloc(sizeof(TokenEntry));
  entry->token = strdup(token);
  entry->index = index;
  entry->next = map->buckets[idx];
  map->buckets[idx] = entry;
}

static int token_map_get(const TokenMap* map, const char* token) {
  if (!map || !token) return -1;
  
  size_t h = 5381;
  for (const unsigned char* s = (const unsigned char*)token; *s; s++) {
    h = ((h << 5) + h) + *s;
  }
  size_t idx = h & (map->nbuckets - 1);
  
  TokenEntry* entry = map->buckets[idx];
  while (entry) {
    if (strcmp(entry->token, token) == 0) {
      return entry->index;
    }
    entry = entry->next;
  }
  return -1;
}

static void token_map_clear(TokenMap* map) {
  for (size_t i = 0; i < map->nbuckets; i++) {
    TokenEntry* e = map->buckets[i];
    while (e) {
      TokenEntry* next = e->next;
      free(e->token);
      free(e);
      e = next;
    }
    map->buckets[i] = NULL;
  }
}

void rebuild_token_map(UnigramModel* model) {
  if (!model) return;
  
  token_map_clear(&model->token_map);
  
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
      uint32_t hash = fast_hash(model->entries[i].subword, model->entries[i].len);
      model->entries[i].hash = hash;
      uint32_t hash_key = hash % HASH_TABLE_SIZE;
      
      model->next_in_bucket[i] = model->hash_table[hash_key];
      model->hash_table[hash_key] = (int)i;
    }
  }
}

static void transfer_entry_from_builder(UnigramModel* model, HashVocabEntry* entries, size_t count) {
  uint64_t total_freq = 0;
  
  for (size_t i = 0; i < count && i < model->capacity; i++) {
    const char* str = interner_get_string(NULL, entries[i].string_id); // This needs proper interner reference
    if (!str) continue;
    
    size_t len = interner_get_length(NULL, entries[i].string_id); // This needs proper interner reference
    char* subword_copy = (char*)malloc(len + 1);
    if (!subword_copy) continue;
    
    memcpy(subword_copy, str, len + 1);
    
    model->entries[model->size].subword = subword_copy;
    model->entries[model->size].freq = (int)entries[i].frequency;
    model->entries[model->size].len = (uint16_t)len;
    model->entries[model->size].hash = 0;
    
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
}

void initialize_from_vocab(UnigramModel* model, VocabBuilder* builder) {
  if (!model || !builder) {
    fprintf(stderr, "[ERROR] Model or VocabBuilder pointer is NULL\n");
    return;
  }

  HashVocabEntry* entries = NULL;
  size_t count = vocab_builder_finalize(builder, &entries);
  
  if (!entries || count == 0) {
    fprintf(stderr, "[WARNING] No entries from VocabBuilder\n");
    return;
  }

  // Sort by frequency
  qsort(entries, count, sizeof(HashVocabEntry), 
        [](const void* a, const void* b) -> int {
          const HashVocabEntry* ea = (const HashVocabEntry*)a;
          const HashVocabEntry* eb = (const HashVocabEntry*)b;
          return (ea->frequency < eb->frequency) - (ea->frequency > eb->frequency);
        });

  // Limit to MAX_VOCAB_SIZE
  if (count > MAX_VOCAB_SIZE) {
    count = MAX_VOCAB_SIZE;
  }

  transfer_entry_from_builder(model, entries, count);
  free(entries);

  rebuild_token_map(model);
  rebuild_hash_table(model);
}

int fast_token_lookup(const UnigramModel* model, const char* token, size_t len) {
  uint32_t hash = fast_hash(token, len);
  uint32_t hash_key = hash % HASH_TABLE_SIZE;
  
  int idx = model->hash_table[hash_key];
  while (idx != -1) {
    const UnigramEntry* entry = &model->entries[idx];
    if (entry->hash == hash && entry->len == len && 
        memcmp(entry->subword, token, len) == 0) {
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

static inline double log_sum_exp(double a, double b) {
  if (a == -DBL_MAX) return b;
  if (b == -DBL_MAX) return a;
  
  double diff = a - b;
  if (diff > 50.0) return a;
  if (diff < -50.0) return b;
  
  return (a > b) ? a + log1p(exp(diff)) : b + log1p(exp(-diff));
}

static double forward(const UnigramModel* model, const char* line, double* alpha, size_t len) {
  alpha[0] = 0.0;
  for (size_t i = 1; i <= len; ++i) {
    alpha[i] = -DBL_MAX;
  }

  for (size_t pos = 1; pos <= len; ++pos) {
    size_t max_k = (MAX_SUBWORD_LEN < pos) ? MAX_SUBWORD_LEN : pos;
    
    for (size_t k = 1; k <= max_k; ++k) {
      size_t start = pos - k;
      
      int token_idx = fast_token_lookup(model, line + start, k);
      if (token_idx >= 0) {
        alpha[pos] = log_sum_exp(alpha[pos], alpha[start] + model->entries[token_idx].score);
      }
    }
  }
  return alpha[len];
}

void run_em_training(UnigramModel* model, const char* const* corpus_lines, size_t num_lines, size_t max_steps) {
  if (!model || !corpus_lines || num_lines == 0) {
    fprintf(stderr, "[ERROR] Invalid arguments to EM trainer\n");
    return;
  }

  double* alpha = (double*)malloc(sizeof(double) * MAX_LINE);
  double* expected_counts = (double*)calloc(model->size, sizeof(double));
  if (!alpha || !expected_counts) {
    fprintf(stderr, "[ERROR] Failed to allocate memory for EM arrays\n");
    free(alpha);
    free(expected_counts);
    return;
  }

  const double SMOOTHING = 1e-8;
  double prev_log_likelihood = -DBL_MAX;
  
  for (size_t step = 0; step < max_steps; ++step) {
    memset(expected_counts, 0, sizeof(double) * model->size);
    double total_log_prob = 0.0;
    size_t valid_lines = 0;

    // E-step
    for (size_t l = 0; l < num_lines; ++l) {
      const char* line = corpus_lines[l];
      size_t len = strlen(line);
      if (len == 0 || len >= MAX_LINE) continue;

      double logZ = forward(model, line, alpha, len);
      
      if (!isfinite(logZ) || logZ == -DBL_MAX) continue;
      
      total_log_prob += logZ;
      valid_lines++;

      // Backward pass for expected counts
      for (size_t pos = len; pos > 0; --pos) {
        size_t max_k = (MAX_SUBWORD_LEN < pos) ? MAX_SUBWORD_LEN : pos;
        
        for (size_t k = 1; k <= max_k; ++k) {
          size_t start = pos - k;
          
          int token_idx = fast_token_lookup(model, line + start, k);
          if (token_idx >= 0) {
            double contrib = alpha[start] + model->entries[token_idx].score - logZ;
            
            if (contrib > -30.0 && isfinite(contrib)) {
              double exp_contrib = exp(contrib);
              if (isfinite(exp_contrib)) {
                expected_counts[token_idx] += exp_contrib;
              }
            }
          }
        }
      }
    }

    if (valid_lines == 0) {
      fprintf(stderr, "[ERROR] No valid lines processed in EM step %zu\n", step + 1);
      break;
    }

    double avg_log_likelihood = total_log_prob / valid_lines;
    if (!isfinite(avg_log_likelihood)) {
      fprintf(stderr, "[ERROR] Non-finite log likelihood at step %zu\n", step + 1);
      break;
    }

    // M-step
    double total_count = SMOOTHING * model->size;
    for (size_t i = 0; i < model->size; ++i) {
      total_count += expected_counts[i];
    }

    if (total_count > 0) {
      double log_total = log(total_count);
      for (size_t i = 0; i < model->size; ++i) {
        double smoothed_count = expected_counts[i] + SMOOTHING;
        double new_score = log(smoothed_count) - log_total;
        
        if (isfinite(new_score) && new_score > -50.0) {
          model->entries[i].score = new_score;
        } else {
          model->entries[i].score = -30.0;
        }
      }
    } else {
      fprintf(stderr, "[ERROR] Zero total count at step %zu\n", step + 1);
      break;
    }

    printf("[EM] Step %zu: Avg log likelihood = %.5f (valid lines: %zu)\n", 
           step + 1, avg_log_likelihood, valid_lines);
    
    // Check convergence
    if (step > 0) {
      double improvement = avg_log_likelihood - prev_log_likelihood;
      if (improvement < 1e-6 && improvement > -1e-6) {
        printf("[EM] Converged after %zu steps\n", step + 1);
        break;
      }
    }
    prev_log_likelihood = avg_log_likelihood;
  }

  free(alpha);
  free(expected_counts);
}

char** viterbi_tokenize(const UnigramModel* model, const char* line, size_t* out_token_count) {
  if (!model || !line || !out_token_count) {
    if (out_token_count) *out_token_count = 0;
    return NULL;
  }

  size_t len = strlen(line);
  if (len == 0) {
    *out_token_count = 0;
    return NULL;
  }

  ViterbiCell* dp = (ViterbiCell*)malloc(sizeof(ViterbiCell) * (len + 1));
  if (!dp) {
    fprintf(stderr, "[ERROR] Couldn't allocate ViterbiCell array\n");
    *out_token_count = 0;
    return NULL;
  }

  // Initialize DP table
  dp[0].score = 0.0;
  dp[0].prev = -1;
  dp[0].token_index = -1;
  
  for (size_t i = 1; i <= len; i++) {
    dp[i].score = -DBL_MAX;
    dp[i].prev = -1;
    dp[i].token_index = -1;
  }

  // Fill DP table
  for (size_t i = 1; i <= len; i++) {
    size_t max_k = (MAX_SUBWORD_LEN < i) ? MAX_SUBWORD_LEN : i;
    
    for (size_t k = 1; k <= max_k; k++) {
      size_t start = i - k;
      
      int token_idx = fast_token_lookup(model, line + start, k);
      if (token_idx >= 0) {
        double score = dp[start].score + model->entries[token_idx].score;
        if (score > dp[i].score) {
          dp[i].score = score;
          dp[i].prev = (int)start;
          dp[i].token_index = token_idx;
        }
      }
    }
  }

  // Count tokens
  size_t count = 0;
  for (int pos = (int)len; pos > 0 && dp[pos].prev >= 0; pos = dp[pos].prev) {
    count++;
  }

  if (count == 0) {
    free(dp);
    *out_token_count = 0;
    return NULL;
  }

  char** result = (char**)malloc(sizeof(char*) * count);
  if (!result) {
    fprintf(stderr, "[ERROR] Couldn't allocate result array\n");
    free(dp);
    *out_token_count = 0;
    return NULL;
  }

  // Fill result array
  size_t idx = count;
  for (int pos = (int)len; pos > 0 && dp[pos].prev >= 0; pos = dp[pos].prev) {
    int token_idx = dp[pos].token_index;
    if (token_idx >= 0) {
      --idx;
      result[idx] = strdup(model->entries[token_idx].subword);
    }
  }

  *out_token_count = count;
  free(dp);
  return result;
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

int* encode_to_ids(const UnigramModel* model, const char* line, size_t* out_count) {
  if (!model || !line || !out_count) {
    if (out_count) *out_count = 0;
    return NULL;
  }
  
  size_t count = 0;
  char** tokens = viterbi_tokenize(model, line, &count);
  if (!tokens || count == 0) {
    *out_count = 0;
    return NULL;
  }

  int* ids = (int*)malloc(sizeof(int) * count);
  if (!ids) {
    for (size_t i = 0; i < count; ++i) {
      free(tokens[i]);
    }
    free(tokens);
    *out_count = 0;
    return NULL;
  }

  for (size_t i = 0; i < count; ++i) {
    ids[i] = get_token_id(model, tokens[i]);
    free(tokens[i]);
  }

  *out_count = count;
  free(tokens);
  return ids;
}