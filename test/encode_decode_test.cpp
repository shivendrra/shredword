/*
  encode_decode_test.cpp
  * Test the encoding and decoding mechanisms of the BPE tokenizer.
  * It reads a test file, trains (or loads) the tokenizer model,
  * encodes the text, decodes it back, and prints metrics such as:
  *   - Encoding and decoding times
  *   - Compression ratio (original bytes per token)
  *   - Peak memory usage
  *   - A check that decoded text matches the original.
  * Compile as: g++ -o encode_decode_test encode_decode_test.cpp src/csrc/main.cpp src/csrc/base.cpp src/csrc/cache.cpp -lpthread
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
  const char* test_file = "test.txt"; // path to your test file
  char* test_text = read_file(test_file);

  // Initialize tokenizer
  Shred tokenizer;
  init_shred(&tokenizer);
  
  // For testing, train the model on the test text (or load a pre-trained model)
  dynamic_train_bpe(&tokenizer, test_text, 1256, 5000);

  struct rusage usage_start, usage_end;
  getrusage(RUSAGE_SELF, &usage_start);
  time_t start_time = time(NULL);

  // Encoding test data
  int encoded_size = 0;
  int* encoded_ids = encode(&tokenizer, test_text, &encoded_size);

  time_t mid_time = time(NULL);

  // Decoding back to text
  char* decoded_text = decode(&tokenizer, encoded_ids, encoded_size);

  time_t end_time = time(NULL);
  getrusage(RUSAGE_SELF, &usage_end);

  double encoding_time = difftime(mid_time, start_time);
  double decoding_time = difftime(end_time, mid_time);
  double total_time = difftime(end_time, start_time);
  long mem_usage = usage_end.ru_maxrss; // peak memory usage (in KB)

  // Calculate compression ratio: original size (bytes) / token count
  int original_size = strlen(test_text);
  double compression_ratio = (double)original_size / encoded_size;

  // Report encoding/decoding metrics
  printf("==== Encoding/Decoding Report ====\n");
  printf("Test File: %s\n", test_file);
  printf("Original Size: %d bytes\n", original_size);
  printf("Encoded Token Count: %d\n", encoded_size);
  printf("Compression Ratio (bytes per token): %.2f\n", compression_ratio);
  printf("Encoding Time: %.2f seconds\n", encoding_time);
  printf("Decoding Time: %.2f seconds\n", decoding_time);
  printf("Total Time: %.2f seconds\n", total_time);
  printf("Peak Memory Usage: %ld KB\n", mem_usage);
  
  if (strcmp(test_text, decoded_text) == 0)
    printf("Decoded text matches original.\n");
  else
    printf("Decoded text does NOT match original.\n");
  printf("==================================\n");

  free(test_text);
  free(encoded_ids);
  free(decoded_text);
  free_tokenizer(&(tokenizer.base));
  return 0;
}