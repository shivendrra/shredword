// Enhanced test program for Unigram tokenizer with comprehensive testing
// Compilation: g++ -o run unigram_test.cpp ../shred/csrc/unigram/unigram.cpp ../shred/csrc/unigram/normalize.cpp ../shred/csrc/unigram/hash.cpp trie.cpp
// Usage: -> ./run <corpus_file> [max_subword_length] [target_vocab_size] [em_steps] [--verbose] [--benchmark]
//        -> run captions.txt 10 300 6 --verbose --benchmark

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <vector>
#include <string>
#include <algorithm>
#include <chrono>
#include <map>
#include <set>
#include <iomanip>
#include <iostream>
#include "../shred/csrc/unigram/normalize.h"
#include "../shred/csrc/unigram/unigram.h"
#include "../shred/csrc/unigram/hash.h"
#include "trie.h"

// Buffer sizes
#define READ_BUFFER_SIZE 8192
#define NORMALIZED_BUFFER_SIZE (READ_BUFFER_SIZE * 4)
#define MAX_CORPUS_LINES 100000

// Test configuration
struct TestConfig {
  bool verbose;
  bool benchmark;
  bool save_intermediate;
  bool run_quality_tests;
  const char* output_dir;
};

static void print_usage(const char *program_name) {
  printf("Usage: %s <corpus_file> [max_subword_length] [target_vocab_size] [em_steps] [options]\n", program_name);
  printf("  corpus_file: Input text file to process\n");
  printf("  max_subword_length: Maximum subword length (default: 10, max: %d)\n", MAX_SUBWORD_LEN - 1);
  printf("  target_vocab_size: Target vocabulary size after pruning (default: 8000)\n");
  printf("  em_steps: Number of EM training steps (default: 5)\n");
  printf("\nOptions:\n");
  printf("  --verbose: Enable detailed logging\n");
  printf("  --benchmark: Run performance benchmarks\n");
  printf("  --save-intermediate: Save model at each EM step\n");
  printf("  --quality-tests: Run comprehensive quality tests\n");
  printf("  --output-dir DIR: Specify output directory (default: current)\n");
}

static void print_vocab_stats(const VocabBuilder *builder, const TestConfig& config) {
  if (!builder) return;
    
  fprintf(stderr, "\n=== Initial Vocabulary Statistics ===\n");
  fprintf(stderr, "  VocabBuilder created with max_entries: %zu\n", builder->max_entries);
  fprintf(stderr, "  Minimum frequency threshold: %zu\n", builder->min_frequency);
  
  if (config.verbose) {
    fprintf(stderr, "  (VocabBuilder contains trie-based vocabulary)\n");
  }
}

static void print_model_stats(const UnigramModel *model, const TestConfig& config) {
  if (!model) return;
    
  fprintf(stderr, "\n=== Unigram Model Statistics ===\n");
  fprintf(stderr, "  Model vocabulary size: %zu\n", model->size);
  fprintf(stderr, "  Model capacity: %zu\n", model->capacity);
  fprintf(stderr, "  Token map buckets: %zu\n", model->token_map.nbuckets);
  
  // Calculate score statistics
  if (model->size > 0) {
    double min_score = model->entries[0].score;
    double max_score = model->entries[0].score;
    double sum_score = 0.0;
    
    for (size_t i = 0; i < model->size; i++) {
      double score = model->entries[i].score;
      min_score = std::min(min_score, score);
      max_score = std::max(max_score, score);
      sum_score += score;
    }
    
    fprintf(stderr, "  Score range: [%.6f, %.6f]\n", min_score, max_score);
    fprintf(stderr, "  Average score: %.6f\n", sum_score / model->size);
  }
  
  // Show top tokens by score
  size_t top_count = config.verbose ? 20 : 10;
  fprintf(stderr, "\n=== Top %zu Tokens by Score ===\n", top_count);
  for (size_t i = 0; i < model->size && i < top_count; i++) {
    if (model->entries[i].subword) {
      fprintf(stderr, "  %2zu: '%s' (score: %.6f, freq: %d, len: %d)\n", 
              i + 1, model->entries[i].subword, 
              model->entries[i].score, model->entries[i].freq,
              model->entries[i].len);
    }
  }
  
  if (config.verbose) {
    // Show frequency distribution
    std::map<int, int> freq_bins;
    for (size_t i = 0; i < model->size; i++) {
      int freq = model->entries[i].freq;
      int bin = freq < 10 ? freq : (freq < 100 ? 10 : (freq < 1000 ? 100 : 1000));
      freq_bins[bin]++;
    }
    
    fprintf(stderr, "\n=== Frequency Distribution ===\n");
    for (const auto& pair : freq_bins) {
      fprintf(stderr, "  Freq %s%d: %d tokens\n", 
              pair.first >= 1000 ? ">=" : 
              pair.first >= 100 ? "100-999: " : 
              pair.first >= 10 ? "10-99: " : "=", 
              pair.first, pair.second);
    }
  }
}

static void benchmark_tokenization(const UnigramModel *model, const char **test_sentences, size_t num_sentences, const TestConfig& config) {
  if (!config.benchmark) return;
  fprintf(stderr, "\n=== Tokenization Benchmark ===\n");
  
  auto start = std::chrono::high_resolution_clock::now();
  size_t total_tokens = 0;
  size_t total_chars = 0;
  
  for (int run = 0; run < 1000; run++) {
    for (size_t i = 0; i < num_sentences; i++) {
      size_t token_count = 0;
      char **tokens = viterbi_tokenize(model, test_sentences[i], &token_count);
      
      if (tokens) {
        total_tokens += token_count;
        total_chars += strlen(test_sentences[i]);
        
        for (size_t j = 0; j < token_count; j++) {
          free(tokens[j]);
        }
        free(tokens);
      }
    }
  }
  
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  double seconds = duration.count() / 1000000.0;
  fprintf(stderr, "  1000 runs completed in %.3f seconds\n", seconds);
  fprintf(stderr, "  Average: %.3f μs per sentence\n", duration.count() / (1000.0 * num_sentences));
  fprintf(stderr, "  Throughput: %.1f chars/sec, %.1f tokens/sec\n", 
          total_chars / seconds, total_tokens / seconds);
}

static void test_tokenization_quality(const UnigramModel *model, const TestConfig& config) {
  if (!config.run_quality_tests) return;
  
  fprintf(stderr, "\n=== Quality Assessment Tests ===\n");
  
  // Test cases with expected behaviors
  const char* quality_tests[] = {
    "hello",           // Simple word
    "hello world",     // Two words
    "preprocessing",   // Should split into meaningful parts
    "unhappiness",     // Prefix + root + suffix
    "internationalization", // Long word
    "COVID-19",        // Mixed alphanumeric
    "don't",          // Contraction
    "U.S.A.",         // Abbreviation
    "123-456-7890",   // Numbers with separators
    "hello@world.com", // Email-like
    "",               // Empty string
    "   ",            // Whitespace only
    "日本語",          // Non-Latin (if supported)
    "café résumé",    // Accented characters
  };
  
  size_t num_quality_tests = sizeof(quality_tests) / sizeof(quality_tests[0]);
  
  for (size_t i = 0; i < num_quality_tests; i++) {
    fprintf(stderr, "\nQuality Test %zu: \"%s\"\n", i + 1, quality_tests[i]);
    
    size_t token_count = 0;
    char **tokens = viterbi_tokenize(model, quality_tests[i], &token_count);
    
    if (tokens && token_count > 0) {
      fprintf(stderr, "  Tokens (%zu): [", token_count);
      size_t total_len = 0;
      for (size_t j = 0; j < token_count; j++) {
        fprintf(stderr, "\"%s\"", tokens[j]);
        total_len += strlen(tokens[j]);
        if (j < token_count - 1) fprintf(stderr, ", ");
        free(tokens[j]);
      }
      fprintf(stderr, "]\n");
      
      // Check if reconstruction is possible
      size_t input_len = strlen(quality_tests[i]);
      if (total_len != input_len) {
        fprintf(stderr, "  WARNING: Token length mismatch (input: %zu, tokens: %zu)\n", 
                input_len, total_len);
      }
      
      free(tokens);
    } else {
      fprintf(stderr, "  FAILED: No tokenization produced\n");
    }
    
    // Test round-trip with IDs
    size_t id_count = 0;
    int *ids = encode_to_ids(model, quality_tests[i], &id_count);
    
    if (ids && id_count > 0) {
      fprintf(stderr, "  Token IDs (%zu): [", id_count);
      for (size_t j = 0; j < id_count; j++) {
        fprintf(stderr, "%d", ids[j]);
        if (j < id_count - 1) fprintf(stderr, ", ");
      }
      fprintf(stderr, "]\n");
      free(ids);
    } else {
      fprintf(stderr, "  FAILED: No ID encoding produced\n");
    }
  }
}

static void test_tokenization(const UnigramModel *model, const char **test_sentences, 
                             size_t num_sentences, const TestConfig& config) {
  fprintf(stderr, "\n=== Standard Tokenization Test ===\n");
  
  for (size_t i = 0; i < num_sentences; i++) {
    fprintf(stderr, "\nInput %zu: \"%s\"\n", i + 1, test_sentences[i]);
    
    // Test Viterbi tokenization
    size_t token_count = 0;
    char **tokens = viterbi_tokenize(model, test_sentences[i], &token_count);
    
    if (tokens && token_count > 0) {
      fprintf(stderr, "Tokens (%zu): [", token_count);
      for (size_t j = 0; j < token_count; j++) {
        fprintf(stderr, "\"%s\"", tokens[j]);
        if (j < token_count - 1) fprintf(stderr, ", ");
        free(tokens[j]);
      }
      fprintf(stderr, "]\n");
      free(tokens);
    } else {
      fprintf(stderr, "Tokenization failed!\n");
    }
    
    // Test ID encoding
    size_t id_count = 0;
    int *ids = encode_to_ids(model, test_sentences[i], &id_count);
    
    if (ids && id_count > 0) {
      fprintf(stderr, "Token IDs (%zu): [", id_count);
      for (size_t j = 0; j < id_count; j++) {
        fprintf(stderr, "%d", ids[j]);
        if (j < id_count - 1) fprintf(stderr, ", ");
      }
      fprintf(stderr, "]\n");
      free(ids);
    } else {
      fprintf(stderr, "ID encoding failed!\n");
    }
  }
  
  // Run additional tests
  benchmark_tokenization(model, test_sentences, num_sentences, config);
  test_tokenization_quality(model, config);
}

static char **load_corpus_lines(const char *filename, size_t *num_lines, const TestConfig& config) {
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Error opening file '%s': %s\n", filename, strerror(errno));
    return NULL;
  }
  
  char **lines = (char**)malloc(sizeof(char*) * MAX_CORPUS_LINES);
  char *buffer = (char*)malloc(READ_BUFFER_SIZE);
  char *normalized = (char*)malloc(NORMALIZED_BUFFER_SIZE);
  
  if (!lines || !buffer || !normalized) {
    fprintf(stderr, "Error: Failed to allocate memory for corpus loading\n");
    free(lines);
    free(buffer);
    free(normalized);
    fclose(fp);
    return NULL;
  }
  
  size_t count = 0;
  size_t empty_lines = 0;
  size_t long_lines = 0;
  
  while (fgets(buffer, READ_BUFFER_SIZE, fp) && count < MAX_CORPUS_LINES) {
    // Remove trailing newline
    size_t len = strlen(buffer);
    if (len > 0 && buffer[len - 1] == '\n') {
      buffer[len - 1] = '\0';
      len--;
    }
    
    // Track statistics
    if (len == 0) {
      empty_lines++;
      continue;
    }
    
    if (len >= READ_BUFFER_SIZE - 1) {
      long_lines++;
      if (config.verbose) {
        fprintf(stderr, "Warning: Line %zu may be truncated (length: %zu)\n", count + 1, len);
      }
    }
    
    // Normalize the line
    if (normalize_line(buffer, normalized, NORMALIZED_BUFFER_SIZE) < 0) {
      if (config.verbose) {
        fprintf(stderr, "Warning: Failed to normalize line %zu\n", count + 1);
      }
      continue;
    }
    
    // Store normalized line
    lines[count] = strdup(normalized);
    if (!lines[count]) {
      fprintf(stderr, "Error: Failed to allocate memory for line %zu\n", count + 1);
      break;
    }
    count++;
    
    if (config.verbose && count % 1000 == 0) {
      fprintf(stderr, "Loaded %zu lines...\n", count);
    }
  }
  
  free(buffer);
  free(normalized);
  fclose(fp);
  
  if (config.verbose) {
    fprintf(stderr, "Corpus loading complete:\n");
    fprintf(stderr, "  Valid lines: %zu\n", count);
    fprintf(stderr, "  Empty lines skipped: %zu\n", empty_lines);
    fprintf(stderr, "  Potentially truncated lines: %zu\n", long_lines);
  }
  
  *num_lines = count;
  return lines;
}

static void free_corpus_lines(char **lines, size_t num_lines) {
  if (!lines) return;
  for (size_t i = 0; i < num_lines; i++) {
    free(lines[i]);
  }
  free(lines);
}

static void save_intermediate_model(const UnigramModel* model, const char* base_filename, 
                                   int step, const TestConfig& config) {
  if (!config.save_intermediate) return;
  
  char filename[512];
  snprintf(filename, sizeof(filename), "%s%s.step%d.model", 
           config.output_dir ? config.output_dir : "", base_filename, step);
  
  save_unigram_model(model, filename);
  if (config.verbose) {
    fprintf(stderr, "Saved intermediate model: %s\n", filename);
  }
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    print_usage(argv[0]);
    return 1;
  }

  // Parse command line arguments
  const char *filename = argv[1];
  size_t max_subword_len = 10;
  size_t target_vocab_size = 8000;
  size_t em_steps = 5;
  
  TestConfig config = {false, false, false, false, nullptr};
  
  int arg_idx = 2;
  
  // Parse numeric arguments
  if (arg_idx < argc && argv[arg_idx][0] != '-') {
    char *endptr;
    long parsed_len = strtol(argv[arg_idx], &endptr, 10);
    if (*endptr != '\0' || parsed_len < 1 || parsed_len >= MAX_SUBWORD_LEN) {
      fprintf(stderr, "Error: Invalid max_subword_length. Must be between 1 and %d\n", 
              MAX_SUBWORD_LEN - 1);
      return 1;
    }
    max_subword_len = (size_t)parsed_len;
    arg_idx++;
  }

  if (arg_idx < argc && argv[arg_idx][0] != '-') {
    char *endptr;
    long parsed_size = strtol(argv[arg_idx], &endptr, 10);
    if (*endptr != '\0' || parsed_size < 100 || parsed_size > MAX_VOCAB_SIZE) {
      fprintf(stderr, "Error: Invalid target_vocab_size. Must be between 100 and %d\n", MAX_VOCAB_SIZE);
      return 1;
    }
    target_vocab_size = (size_t)parsed_size;
    arg_idx++;
  }

  if (arg_idx < argc && argv[arg_idx][0] != '-') {
    char *endptr;
    long parsed_steps = strtol(argv[arg_idx], &endptr, 10);
    if (*endptr != '\0' || parsed_steps < 1 || parsed_steps > 50) {
      fprintf(stderr, "Error: Invalid em_steps. Must be between 1 and 50\n");
      return 1;
    }
    em_steps = (size_t)parsed_steps;
    arg_idx++;
  }
  
  // Parse options
  for (int i = arg_idx; i < argc; i++) {
    if (strcmp(argv[i], "--verbose") == 0) {
      config.verbose = true;
    } else if (strcmp(argv[i], "--benchmark") == 0) {
      config.benchmark = true;
    } else if (strcmp(argv[i], "--save-intermediate") == 0) {
      config.save_intermediate = true;
    } else if (strcmp(argv[i], "--quality-tests") == 0) {
      config.run_quality_tests = true;
    } else if (strcmp(argv[i], "--output-dir") == 0 && i + 1 < argc) {
      config.output_dir = argv[++i];
    } else {
      fprintf(stderr, "Unknown option: %s\n", argv[i]);
      print_usage(argv[0]);
      return 1;
    }
  }

  fprintf(stderr, "=== Enhanced Unigram Tokenizer Training & Testing ===\n");
  fprintf(stderr, "Corpus file: %s\n", filename);
  fprintf(stderr, "Max subword length: %zu\n", max_subword_len);
  fprintf(stderr, "Target vocabulary size: %zu\n", target_vocab_size);
  fprintf(stderr, "EM training steps: %zu\n", em_steps);
  fprintf(stderr, "Verbose mode: %s\n", config.verbose ? "ON" : "OFF");
  fprintf(stderr, "Benchmark mode: %s\n", config.benchmark ? "ON" : "OFF");
  fprintf(stderr, "Quality tests: %s\n", config.run_quality_tests ? "ON" : "OFF");

  auto total_start = std::chrono::high_resolution_clock::now();

  // Step 1: Build initial vocabulary from corpus using VocabBuilder
  fprintf(stderr, "\n--- Step 1: Building initial vocabulary ---\n");
  
  // Create VocabBuilder instead of VocabTable
  VocabBuilder *builder = vocab_builder_create(200000, 1);  // max_entries, min_frequency
  if (!builder) {
    fprintf(stderr, "Error: Failed to create vocabulary builder\n");
    return 1;
  }

  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Error opening file '%s': %s\n", filename, strerror(errno));
    vocab_builder_free(builder);
    return 1;
  }

  char *line = (char *)malloc(READ_BUFFER_SIZE);
  char *normalized = (char *)malloc(NORMALIZED_BUFFER_SIZE);
  
  if (!line || !normalized) {
    fprintf(stderr, "Error: Failed to allocate memory buffers\n");
    free(line);
    free(normalized);
    vocab_builder_free(builder);
    fclose(fp);
    return 1;
  }

  size_t line_count = 0;
  size_t error_count = 0;
  
  auto vocab_start = std::chrono::high_resolution_clock::now();
  
  while (fgets(line, READ_BUFFER_SIZE, fp)) {
    line_count++;
    
    size_t len = strlen(line);
    if (len > 0 && line[len - 1] == '\n') {
      line[len - 1] = '\0';
    }
    
    if (line[0] == '\0') continue;
    
    if (normalize_line(line, normalized, NORMALIZED_BUFFER_SIZE) < 0) {
      error_count++;
      continue;
    }
    
    // Use vocab_builder_add_line instead of add_subwords
    vocab_builder_add_line(builder, normalized, max_subword_len);
    
    if (line_count % 5000 == 0) {
      fprintf(stderr, "Processed %zu lines\n", line_count);
    }
  }
  
  fclose(fp);
  free(line);
  free(normalized);

  auto vocab_end = std::chrono::high_resolution_clock::now();
  auto vocab_duration = std::chrono::duration_cast<std::chrono::milliseconds>(vocab_end - vocab_start);

  fprintf(stderr, "Vocabulary building complete in %lld ms\n", vocab_duration.count());
  fprintf(stderr, "Lines processed: %zu (errors: %zu)\n", line_count, error_count);
  print_vocab_stats(builder, config);

  // Step 2: Initialize Unigram model
  fprintf(stderr, "\n--- Step 2: Initializing Unigram model ---\n");
  
  UnigramModel *model = create_unigram_model(200000);  // Large initial capacity
  if (!model) {
    fprintf(stderr, "Error: Failed to create Unigram model\n");
    vocab_builder_free(builder);
    return 1;
  }

  // Use VocabBuilder instead of VocabTable
  initialize_from_vocab(model, builder);
  print_model_stats(model, config);

  // Step 3: Load corpus for EM training
  fprintf(stderr, "\n--- Step 3: Loading corpus for EM training ---\n");
  
  size_t num_corpus_lines = 0;
  char **corpus_lines = load_corpus_lines(filename, &num_corpus_lines, config);
  
  if (!corpus_lines || num_corpus_lines == 0) {
    fprintf(stderr, "Error: Failed to load corpus lines\n");
    vocab_builder_free(builder);
    free_unigram_model(model);
    return 1;
  }
  
  fprintf(stderr, "Loaded %zu lines for EM training\n", num_corpus_lines);

  // Step 4: EM Training with intermediate saves
  fprintf(stderr, "\n--- Step 4: EM Training ---\n");
  
  auto em_start = std::chrono::high_resolution_clock::now();
  
  if (config.save_intermediate) {
    save_intermediate_model(model, filename, 0, config);
  }
  
  run_em_training(model, (const char* const*)corpus_lines, num_corpus_lines, em_steps);
  
  auto em_end = std::chrono::high_resolution_clock::now();
  auto em_duration = std::chrono::duration_cast<std::chrono::milliseconds>(em_end - em_start);
  
  fprintf(stderr, "EM training completed in %lld ms\n", em_duration.count());

  // Step 5: Prune vocabulary
  fprintf(stderr, "\n--- Step 5: Pruning vocabulary ---\n");
  
  if (model->size > target_vocab_size) {
    fprintf(stderr, "Pruning vocabulary from %zu to %zu tokens\n", model->size, target_vocab_size);
    prune_unigram_model(model, target_vocab_size);
  } else {
    fprintf(stderr, "No pruning needed (current size: %zu <= target: %zu)\n", 
            model->size, target_vocab_size);
  }

  print_model_stats(model, config);

  // Step 6: Test tokenization
  const char *test_sentences[] = {
    "Hello world!",
    "This is a test sentence for tokenization.",
    "Machine learning is fascinating and powerful.",
    "The quick brown fox jumps over the lazy dog.",
    "Natural language processing with subword tokenization enables better handling of out-of-vocabulary words.",
    "Preprocessing, tokenization, and normalization are essential steps.",
    "COVID-19 pandemic changed the world significantly.",
    "artificial intelligence, machine learning, deep learning"
  };
  
  test_tokenization(model, test_sentences, 8, config);

  // Step 7: Save final model
  fprintf(stderr, "\n--- Step 7: Saving final model ---\n");
  
  char model_filename[512];
  snprintf(model_filename, sizeof(model_filename), "%s%s.final.model", 
           config.output_dir ? config.output_dir : "", filename);
  save_unigram_model(model, model_filename);
  fprintf(stderr, "Final model saved to: %s\n", model_filename);

  // Step 8: Output final vocabulary
  fprintf(stderr, "\n--- Step 8: Final vocabulary output ---\n");
  fprintf(stderr, "Outputting final vocabulary to stdout...\n");
  
  printf("# Enhanced Unigram Model Vocabulary (token score frequency)\n");
  printf("# Vocabulary size: %zu\n", model->size);
  printf("# Training corpus: %s\n", filename);
  printf("# Max subword length: %zu\n", max_subword_len);
  printf("# EM steps: %zu\n", em_steps);
  printf("# Target vocab size: %zu\n", target_vocab_size);
  dump_unigram_model(model);

  auto total_end = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start);
  
  fprintf(stderr, "\n=== Training Complete ===\n");
  fprintf(stderr, "Total time: %.2f seconds\n", total_duration.count() / 1000.0);
  fprintf(stderr, "  - Vocab building: %lld ms\n", vocab_duration.count());
  fprintf(stderr, "  - EM training: %lld ms\n", em_duration.count());
  fprintf(stderr, "Final vocabulary size: %zu\n", model->size);
  fprintf(stderr, "Corpus lines processed: %zu\n", num_corpus_lines);
  fprintf(stderr, "Processing rate: %.1f lines/sec\n", 
          num_corpus_lines * 1000.0 / total_duration.count());

  // Cleanup
  free_corpus_lines(corpus_lines, num_corpus_lines);
  vocab_builder_free(builder);  // Use vocab_builder_free instead of free_vocab
  free_unigram_model(model);

  return 0;
}