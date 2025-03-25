/*
  train_test.cpp
  * Test the training mechanism of the BPE tokenizer.
  * It reads a training file, applies dynamic_train_bpe(),
  * and reports metrics such as time elapsed, peak memory usage, 
  * number of merges, and final vocabulary size.
  * Compile as: g++ -o train_test train_test.cpp src/csrc/main.cpp src/csrc/base.cpp src/csrc/cache.cpp -lpthread
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
#include "main.h"
#include "cache.h"

// Reads the entire content of a file into a null-terminated string.
char* read_file(const char* filename) {
  FILE* file = fopen(filename, "r");
  if (!file) {
    fprintf(stderr, "Error: Unable to open file %s\n", filename);
    exit(EXIT_FAILURE);
  }
  fseek(file, 0, SEEK_END);
  long length = ftell(file);
  fseek(file, 0, SEEK_SET);
  char* buffer = (char*)malloc(length + 1);
  if (!buffer) {
    fprintf(stderr, "Error: Memory allocation failed\n");
    fclose(file);
    exit(EXIT_FAILURE);
  }
  fread(buffer, 1, length, file);
  buffer[length] = '\0';
  fclose(file);
  return buffer;
}

int main() {
  const char* train_file = "train.txt"; // path to your training file
  char* train_text = read_file(train_file);

  // Initialize tokenizer
  Shred tokenizer;
  init_shred(&tokenizer);

  struct rusage usage_start, usage_end;
  getrusage(RUSAGE_SELF, &usage_start);
  time_t start_time = time(NULL);

  // Train the tokenizer (using dynamic training as an example)
  dynamic_train_bpe(&tokenizer, train_text, 1256, 5000);

  time_t end_time = time(NULL);
  getrusage(RUSAGE_SELF, &usage_end);

  double elapsed_time = difftime(end_time, start_time);
  long mem_usage = usage_end.ru_maxrss; // peak memory usage (in KB)

  // Report training metrics
  printf("==== Training Report ====\n");
  printf("Training File: %s\n", train_file);
  printf("Elapsed Time: %.2f seconds\n", elapsed_time);
  printf("Peak Memory Usage: %ld KB\n", mem_usage);
  printf("Number of Merges: %d\n", tokenizer.base.merge_count);
  int final_vocab_size = tokenizer.base.vocab_size + tokenizer.base.merge_count + tokenizer.base.special_token_count;
  printf("Final Vocab Size: %d\n", final_vocab_size);
  printf("=========================\n");

  free(train_text);
  free_tokenizer(&(tokenizer.base));
  return 0;
}
