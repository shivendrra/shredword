#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <float.h>
#include <limits.h>
#include "training.h"
#include "core.h"

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

    printf("[EM] Step %zu: Avg log likelihood = %.5f (valid lines: %zu)\n", step + 1, avg_log_likelihood, valid_lines);
    
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