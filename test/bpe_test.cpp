// test case for BPE trainer
// Compilation: g++ -o run bpe_test.cpp ../shred/csrc/bpe/bpe.cpp ../shred/csrc/bpe/histogram.cpp ../shred/csrc/bpe/hash.cpp ../shred/csrc/bpe/heap.cpp
// Usage: -> ./run

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <unistd.h>
#include "../shred/csrc/bpe/bpe.h"
#include "../shred/csrc/bpe/hash.h"
#include "../shred/csrc/bpe/heap.h"
#include "../shred/csrc/bpe/histogram.h"

// Test utilities
#define TEST_ASSERT(condition, message) \
  do { \
    if (!(condition)) { \
      fprintf(stderr, "[FAIL] %s: %s\n", __func__, message); \
      return 0; \
    } \
  } while(0)

#define TEST_PASS(test_name) \
  do { \
    printf("[PASS] %s\n", test_name); \
    return 1; \
  } while(0)

// Test data creation
static int create_test_corpus(const char* filename) {
  FILE* fp = fopen(filename, "w");
  if (!fp) return 0;
  
  // Simple test corpus with repeated patterns
  fprintf(fp, "the quick brown fox jumps over the lazy dog\n");
  fprintf(fp, "the brown fox is quick and the dog is lazy\n");
  fprintf(fp, "quick brown foxes jump over lazy dogs\n");
  fprintf(fp, "the the the quick quick brown brown fox fox\n");
  fprintf(fp, "jumping foxes are quick brown animals\n");
  fprintf(fp, "lazy dogs sleep under the brown tree\n");
  fprintf(fp, "the quick fox and the lazy dog are friends\n");
  fprintf(fp, "brown and quick describe the fox perfectly\n");
  fprintf(fp, "the lazy dog watches the quick brown fox\n");
  fprintf(fp, "quick movements by the brown fox surprise the dog\n");
  
  // Add some repeated character sequences to test merging
  for (int i = 0; i < 20; i++) {
    fprintf(fp, "hello world hello world programming programming\n");
    fprintf(fp, "testing testing the the quick quick brown brown\n");
    fprintf(fp, "algorithm algorithm implementation implementation\n");
  }
  
  fclose(fp);
  return 1;
}

// Test 1: Basic trainer creation and destruction
static int test_trainer_creation() {
  BPEConfig config = {
    .target_vocab_size = 300,
    .unk_id = -1,
    .character_coverage = 0.995,
    .min_pair_freq = 2
  };
  
  Trainer* trainer = create_trainer(&config);
  TEST_ASSERT(trainer != NULL, "Trainer creation failed");
  TEST_ASSERT(trainer->config.target_vocab_size == 300, "Target vocab size mismatch");
  TEST_ASSERT(trainer->config.character_coverage == 0.995, "Character coverage mismatch");
  TEST_ASSERT(trainer->config.min_pair_freq == 2, "Min pair frequency mismatch");
  TEST_ASSERT(trainer->num_merges == 0, "Initial merge count should be 0");
  
  bpe_trainer_destroy(trainer);
  TEST_PASS("test_trainer_creation");
}

// Test 2: Config validation and defaults
static int test_config_defaults() {
  BPEConfig config = {
    .target_vocab_size = 400,
    .unk_id = -1,
    .character_coverage = 0.0, // Invalid, should default to 0.995
    .min_pair_freq = 0         // Invalid, should default to MIN_PAIR_FREQ
  };
  
  Trainer* trainer = create_trainer(&config);
  TEST_ASSERT(trainer != NULL, "Trainer creation failed");
  TEST_ASSERT(trainer->config.character_coverage == 0.995, "Character coverage should default to 0.995");
  TEST_ASSERT(trainer->config.min_pair_freq == MIN_PAIR_FREQ, "Min pair freq should default to MIN_PAIR_FREQ");
  
  bpe_trainer_destroy(trainer);
  TEST_PASS("test_config_defaults");
}

// Test 3: Corpus loading
static int test_corpus_loading() {
  const char* test_file = "test_corpus.txt";
  TEST_ASSERT(create_test_corpus(test_file), "Failed to create test corpus");
  
  BPEConfig config = {
    .target_vocab_size = 350,
    .unk_id = -1,
    .character_coverage = 0.95,
    .min_pair_freq = 3
  };
  
  Trainer* trainer = create_trainer(&config);
  int result = bpe_load_corpus(trainer, test_file);
  
  TEST_ASSERT(result == 0, "Corpus loading failed");
  TEST_ASSERT(trainer->corpus.vocab_size > 0, "No words loaded from corpus");
  TEST_ASSERT(trainer->corpus.words != NULL, "Words array not allocated");
  TEST_ASSERT(trainer->corpus.word_counts != NULL, "Word counts array not allocated");
  
  printf("[DEBUG] Loaded %zu unique words from test corpus\n", trainer->corpus.vocab_size);
  
  // Verify some words were loaded correctly
  int found_words = 0;
  for (size_t i = 0; i < trainer->corpus.vocab_size && i < 10; i++) {
    if (trainer->corpus.words[i] != NULL && trainer->corpus.word_counts[i] > 0) {
      found_words++;
    }
  }
  TEST_ASSERT(found_words > 0, "No valid words found in loaded corpus");
  
  bpe_trainer_destroy(trainer);
  unlink(test_file); // Clean up test file
  TEST_PASS("test_corpus_loading");
}

// Test 4: Bigram counting
static int test_bigram_counting() {
  const char* test_file = "test_bigrams.txt";
  TEST_ASSERT(create_test_corpus(test_file), "Failed to create test corpus");
  
  BPEConfig config = {
    .target_vocab_size = 400,
    .unk_id = -1,
    .character_coverage = 0.99,
    .min_pair_freq = 2
  };
  
  Trainer* trainer = create_trainer(&config);
  TEST_ASSERT(bpe_load_corpus(trainer, test_file) == 0, "Corpus loading failed");
  
  // Test bigram counting
  bpe_count_bigrams(trainer);
  
  // Verify heap has entries
  TEST_ASSERT(!heap_empty(&trainer->heap), "Heap should not be empty after counting bigrams");
  
  size_t initial_heap_size = trainer->heap.size;
  printf("[DEBUG] Found %zu bigrams above frequency threshold\n", initial_heap_size);
  TEST_ASSERT(initial_heap_size > 0, "No bigrams found above threshold");
  
  // Verify heap ordering (max heap property)
  if (trainer->heap.size >= 2) {
    HeapEntry top = trainer->heap.data[0];
    HeapEntry second = trainer->heap.data[1];
    TEST_ASSERT(top.freq >= second.freq, "Heap ordering violated");
  }
  
  bpe_trainer_destroy(trainer);
  unlink(test_file);
  TEST_PASS("test_bigram_counting");
}

// Test 5: Single merge operation
static int test_single_merge() {
  const char* test_file = "test_merge.txt";
  TEST_ASSERT(create_test_corpus(test_file), "Failed to create test corpus");
  
  BPEConfig config = {
    .target_vocab_size = 400,
    .unk_id = -1,
    .character_coverage = 0.99,
    .min_pair_freq = 2
  };
  
  Trainer* trainer = create_trainer(&config);
  TEST_ASSERT(bpe_load_corpus(trainer, test_file) == 0, "Corpus loading failed");
  
  bpe_init(trainer);
  
  size_t initial_merges = trainer->num_merges;
  int merged = bpe_merge_batch(trainer, 1);
  
  TEST_ASSERT(merged == 1, "Expected exactly one merge");
  TEST_ASSERT(trainer->num_merges == initial_merges + 1, "Merge count not incremented");
  
  // Verify merge operation was recorded
  if (trainer->num_merges > 0) {
    PairKey merge_op = trainer->merge_ops[0];
    TEST_ASSERT(merge_op.first >= 0 && merge_op.first < 1000, "Invalid first token in merge");
    TEST_ASSERT(merge_op.second >= 0 && merge_op.second < 1000, "Invalid second token in merge");
    printf("[DEBUG] First merge: (%d, %d)\n", merge_op.first, merge_op.second);
  }
  
  bpe_trainer_destroy(trainer);
  unlink(test_file);
  TEST_PASS("test_single_merge");
}

// Test 6: Full training cycle
static int test_full_training() {
  const char* test_file = "test_training.txt";
  TEST_ASSERT(create_test_corpus(test_file), "Failed to create test corpus");
  
  BPEConfig config = {
    .target_vocab_size = 300, // Small target for faster testing
    .unk_id = -1,
    .character_coverage = 0.99,
    .min_pair_freq = 2
  };
  
  Trainer* trainer = create_trainer(&config);
  TEST_ASSERT(bpe_load_corpus(trainer, test_file) == 0, "Corpus loading failed");
  
  int total_merges = bpe_train(trainer);
  
  TEST_ASSERT(total_merges > 0, "No merges performed during training");
  TEST_ASSERT(trainer->num_merges == (size_t)total_merges, "Merge count mismatch");
  
  size_t expected_merges = config.target_vocab_size - INITIAL_VOCAB_SIZE;
  printf("[DEBUG] Performed %d merges, expected up to %zu\n", total_merges, expected_merges);
  
  // Training might stop early if heap is exhausted
  TEST_ASSERT((size_t)total_merges <= expected_merges, "Too many merges performed");
  
  bpe_trainer_destroy(trainer);
  unlink(test_file);
  TEST_PASS("test_full_training");
}

// Test 7: Model saving and file creation
static int test_model_saving() {
  const char* test_file = "test_save.txt";
  const char* model_file = "test_model.bin";
  const char* vocab_file = "test_vocab.txt";
  
  TEST_ASSERT(create_test_corpus(test_file), "Failed to create test corpus");
  
  BPEConfig config = {
    .target_vocab_size = 280,
    .unk_id = -1,
    .character_coverage = 0.99,
    .min_pair_freq = 2
  };
  
  Trainer* trainer = create_trainer(&config);
  TEST_ASSERT(bpe_load_corpus(trainer, test_file) == 0, "Corpus loading failed");
  
  int total_merges = bpe_train(trainer);
  TEST_ASSERT(total_merges > 0, "Training failed");
  
  // Test model saving
  bpe_save(trainer, model_file, vocab_file);
  
  // Verify files were created
  FILE* mf = fopen(model_file, "rb");
  TEST_ASSERT(mf != NULL, "Model file not created");
  
  // Check model file has correct structure (3 int32_t per merge)
  fseek(mf, 0, SEEK_END);
  long model_size = ftell(mf);
  fclose(mf);
  
  size_t expected_size = trainer->num_merges * 3 * sizeof(int32_t);
  TEST_ASSERT((size_t)model_size == expected_size, "Model file size incorrect");
  
  // Verify vocab file
  FILE* vf = fopen(vocab_file, "r");
  TEST_ASSERT(vf != NULL, "Vocab file not created");
  
  // Count lines in vocab file
  int line_count = 0;
  char line[1000];
  while (fgets(line, sizeof(line), vf)) {
    line_count++;
  }
  fclose(vf);
  
  size_t expected_vocab_size = INITIAL_VOCAB_SIZE + trainer->num_merges;
  TEST_ASSERT((size_t)line_count == expected_vocab_size, "Vocab file line count incorrect");
  
  printf("[DEBUG] Saved model with %zu merges and %zu vocab entries\n", 
         trainer->num_merges, expected_vocab_size);
  
  bpe_trainer_destroy(trainer);
  
  // Cleanup
  unlink(test_file);
  unlink(model_file);
  unlink(vocab_file);
  
  TEST_PASS("test_model_saving");
}

// Test 8: Error handling
static int test_error_handling() {
  // Test NULL config
  Trainer* trainer = create_trainer(NULL);
  // This should exit the program, so we can't test it directly
  // Instead, test with invalid file
  
  BPEConfig config = {
    .target_vocab_size = 300,
    .unk_id = -1,
    .character_coverage = 0.99,
    .min_pair_freq = 2
  };
  
  trainer = create_trainer(&config);
  
  // Test loading non-existent file
  int result = bpe_load_corpus(trainer, "non_existent_file.txt");
  TEST_ASSERT(result == -1, "Should fail to load non-existent file");
  
  bpe_trainer_destroy(trainer);
  TEST_PASS("test_error_handling");
}

// Test runner
typedef struct {
  const char* name;
  int (*func)(void);
} TestCase;

static TestCase tests[] = {
  {"Trainer Creation", test_trainer_creation},
  {"Config Defaults", test_config_defaults},
  {"Corpus Loading", test_corpus_loading},
  {"Bigram Counting", test_bigram_counting},
  {"Single Merge", test_single_merge},
  {"Full Training", test_full_training},
  {"Model Saving", test_model_saving},
  {"Error Handling", test_error_handling}
};

int main() {
  printf("=== BPE Trainer Test Suite ===\n\n");
  
  int total_tests = sizeof(tests) / sizeof(TestCase);
  int passed = 0;
  int failed = 0;
  
  for (int i = 0; i < total_tests; i++) {
    printf("Running test %d/%d: %s\n", i + 1, total_tests, tests[i].name);
    
    if (tests[i].func()) {
      passed++;
    } else {
      failed++;
      printf("[FAIL] Test failed: %s\n", tests[i].name);
    }
    printf("\n");
  }
  
  printf("=== Test Results ===\n");
  printf("Total tests: %d\n", total_tests);
  printf("Passed: %d\n", passed);
  printf("Failed: %d\n", failed);
  printf("Success rate: %.1f%%\n", 100.0 * passed / total_tests);
  
  if (failed == 0) {
    printf("\n All tests passed!\n");
    return 0;
  } else {
    printf("\n Some tests failed.\n");
    return 1;
  }
}