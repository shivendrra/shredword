#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>  // for DBL_MAX
#include "unigram.h"

UnigramModel* create_unigram_model(size_t capacity) {
  if (capacity == 0) {
    capacity = MAX_VOCAB_SIZE;
  }
  UnigramModel* model = (UnigramModel*)malloc(sizeof(UnigramModel));
  if (!model) {
    fprintf(stderr, "[ERROR]\t Couldn't allocate UnigramModel\n");
    exit(EXIT_FAILURE);
  }
  model->entries = (UnigramEntry*)calloc(capacity, sizeof(UnigramEntry));
  if (!model->entries) {
    fprintf(stderr, "[ERROR]\t Couldn't allocate UnigramEntry array\n");
    free(model);
    exit(EXIT_FAILURE);
  }
  model->capacity = capacity;
  model->size = 0;
  return model;
}

void free_unigram_model(UnigramModel* model) {
  if (!model) {
    return;
  }
  if (model->entries) {
    for (size_t i = 0; i < model->size; i++) {
      if (model->entries[i].subword)
        free(model->entries[i].subword);
    }
    free(model->entries);
  }
  free(model);
}

static int compare_freq_desc(const void* a, const void* b) {
  const UnigramEntry* entry_a = (const UnigramEntry*)a;
  const UnigramEntry* entry_b = (const UnigramEntry*)b;
  if (entry_b->freq > entry_a->freq) return 1;
  if (entry_b->freq < entry_a->freq) return -1;
  return 0;
}

void initialize_from_vocab(UnigramModel* model, VocabTable* table) {
  if (!model || !table) {
    fprintf(stderr, "[ERROR]\t Table or Model pointer is NULL\n");
    return;
  }
  size_t count = 0, total_freq = 0;
  // transfer entries from VocabTable to UnigramModel
  for (size_t i = 0; i < table->capacity; i++) {
    if (table->entries[i].str && count < model->capacity) {
      char* subword_copy = strdup(table->entries[i].str);
      if (!subword_copy) {
        fprintf(stderr, "[ERROR]\t Failed to allocate memory for subword\n");
        continue;  // skipping this entry if allocation fails
      }
      model->entries[count].subword = subword_copy;
      model->entries[count].freq = (int)table->entries[i].count;
      total_freq += table->entries[i].count;
      count++;
    }
  }

  model->size = count;
  
  if (model->size == 0) {
    fprintf(stderr, "[WARNING]\t No entries transferred to UnigramModel\n");
    return;
  }
  
  qsort(model->entries, model->size, sizeof(UnigramEntry), compare_freq_desc);  // Sort by frequency descending

  // trimming to MAX_VOCAB_SIZE
  if (model->size > MAX_VOCAB_SIZE) {
    // freeing extra entries before trimming
    for (size_t i = MAX_VOCAB_SIZE; i < model->size; i++) {
      if (model->entries[i].subword) {
        free(model->entries[i].subword);
        model->entries[i].subword = NULL;
      }
    }
    model->size = MAX_VOCAB_SIZE;
  }

  // Normalize and assign scores as log-probs
  if (total_freq > 0) {
    for (size_t i = 0; i < model->size; i++) {
      double prob = (double)model->entries[i].freq / total_freq;
      model->entries[i].score = log(prob);
    }
  } else {
    fprintf(stderr, "[WARNING]\t Total frequency is zero, cannot assign scores\n");
  }
}

void dump_unigram_model(UnigramModel* model) {
  if (!model) {
    fprintf(stderr, "[ERROR]\t Model pointer is NULL\n");
    return;
  }
  for (size_t i = 0; i < model->size; i++) {
    if (model->entries[i].subword) {
      printf("%s %.5f\n", model->entries[i].subword, model->entries[i].score);
    }
  }
}

static double log_sum_exp(double a, double b) {
  if (a == -DBL_MAX) return b;
  if (b == -DBL_MAX) return a;
  return (a > b) ? a + log1p(exp(b - a)) : b + log1p(exp(a - b));
}

static double forward(const UnigramModel* model, const char* line, double* alpha, size_t len) {
  for (size_t i = 0; i <= len; ++i) alpha[i] = -DBL_MAX;
  alpha[0] = 0.0; // start of sentence

  for (size_t pos = 1; pos <= len; ++pos) {
    for (size_t k = 1; k <= MAX_SUBWORD_LEN && k <= pos; ++k) {
      size_t start = pos - k;
      const char* substr = line + start;
      for (size_t i = 0; i < model->size; ++i) {
        if (model->entries[i].subword &&
            strlen(model->entries[i].subword) == k &&
            strncmp(model->entries[i].subword, substr, k) == 0) {
          alpha[pos] = log_sum_exp(alpha[pos], alpha[start] + model->entries[i].score);
        }
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
  size_t* expected_counts = (size_t*)calloc(model->size, sizeof(size_t));
  if (!alpha || !expected_counts) {
    fprintf(stderr, "[ERROR] Failed to allocate memory for EM arrays\n");
    return;
  }

  for (size_t step = 0; step < max_steps; ++step) {
    memset(expected_counts, 0, sizeof(size_t) * model->size);
    double total_log_prob = 0.0;

    for (size_t l = 0; l < num_lines; ++l) {
      const char* line = corpus_lines[l];
      size_t len = strlen(line);
      if (len == 0) continue;

      double logZ = forward(model, line, alpha, len);
      total_log_prob += logZ;

      for (size_t pos = len; pos > 0; --pos) {
        for (size_t k = 1; k <= MAX_SUBWORD_LEN && k <= pos; ++k) {
          size_t start = pos - k;
          const char* substr = line + start;

          for (size_t i = 0; i < model->size; ++i) {
            if (model->entries[i].subword &&
                strlen(model->entries[i].subword) == k &&
                strncmp(model->entries[i].subword, substr, k) == 0) {
              double contrib = alpha[start] + model->entries[i].score - logZ;
              expected_counts[i] += (size_t)exp(contrib);
            }
          }
        }
      }
    }

    // M-step: Update scores
    size_t total_count = 0;
    for (size_t i = 0; i < model->size; ++i) {
      total_count += expected_counts[i];
    }

    for (size_t i = 0; i < model->size; ++i) {
      if (expected_counts[i] > 0) {
        double prob = (double)expected_counts[i] / total_count;
        model->entries[i].score = log(prob);
      } else {
        model->entries[i].score = -DBL_MAX;  // effectively disabled
      }
    }

    printf("[EM] Step %zu: Avg log likelihood = %.5f\n", step + 1, total_log_prob / num_lines);
  }

  free(alpha);
  free(expected_counts);
}