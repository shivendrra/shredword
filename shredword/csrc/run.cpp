#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "main.h"

// read the entire content of a file into a string
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
  // paths to input files
  const char* train_file = "test_data/captions.txt";
  const char* test_file = "test_data/new.txt";

  // reading training and test data
  printf("Reading training data from %s...\n", train_file);
  char* train_text = read_file(train_file);
  printf("Training data imported.\n");

  printf("Reading test data from %s...\n", test_file);
  char* test_text = read_file(test_file);
  printf("Test data imported.\n");

  // tokenizer initialization
  Shred tokenizer;
  init_shred(&tokenizer);

  // training tokenizer
  int target_vocab_size = 356;
  bool verbose = true;
  printf("Training tokenizer...\n");
  train(&tokenizer, train_text, target_vocab_size, verbose);
  printf("Training complete.\n");

  // encoding test data
  printf("Encoding test data...\n");
  int encoded_size;
  int* encoded_ids = encode(&tokenizer, test_text, &encoded_size);
  printf("Encoded IDs: ");
  for (int i = 0; i < encoded_size && i < 20; i++) { // printing only the first 20 IDs
    printf("%d ", encoded_ids[i]);
  }
  printf("... (truncated)\n\n");

  // decoding the encoded data
  printf("Decoding back to text...\n");
  char* decoded_text = decode(&tokenizer, encoded_ids, encoded_size);
  printf("Decoded text (first 200 chars): %.200s\n\n", decoded_text);

  // verify if original and decoded texts are identical
  if (strcmp(test_text, decoded_text) == 0) {
    printf("Decoded text matches the original test text.\n");
  } else {
    printf("Decoded text does NOT match the original test text.\n");
  }

  // cleanup
  free(train_text);
  free(test_text);
  free(encoded_ids);
  free(decoded_text);
  free_tokenizer(&(tokenizer.base));

  return 0;
}
