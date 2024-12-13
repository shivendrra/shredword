/*
  @run.cpp
  - main run file for testing the bpe tokenization logic
  - compile as: g++ -o run run.cpp main.cpp base.cpp
    - run: ./run
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
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
  const char* train_file = "captions.txt";
  const char* test_file = "new.txt";
  const char* model_file = "tokenizer.model";

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

  // loading or train the tokenizer
  if (_access(model_file, 0) == 0) { // check if the model file exists
    printf("Loading tokenizer model from %s...\n", model_file);
    load_model(&tokenizer, model_file);
    printf("Tokenizer model loaded.\n");
  } else {
    printf("Training tokenizer...\n");
    train(&tokenizer, train_text, 280, false);
    printf("Training complete.\n");

    printf("Saving tokenizer model to %s...\n", model_file);
    save_model(&tokenizer, model_file);
    printf("Tokenizer model saved.\n");
  }

  // encoding test data
  printf("Encoding test data...\n");
  int encoded_size;
  int* encoded_ids = encode(&tokenizer, test_text, &encoded_size);
  printf("Encoded IDs (%d tokens): ", encoded_size);
  for (int i = 0; i < encoded_size; i++) {
    printf("%d ", encoded_ids[i]);
  }
  printf("\n\n");

  // decoding the encoded data
  printf("Decoding back to text...\n");
  char* decoded_text = decode(&tokenizer, encoded_ids, encoded_size);
  printf("Decoded text (%lu characters):\n%s\n\n", strlen(decoded_text), decoded_text);

  // verify original and decoded texts
  if (strcmp(test_text, decoded_text) == 0) {
    printf("Decoded text matches the original test text.\n");
  } else {
    printf("Decoded text does NOT match the original test text.\n");
    // locating the first difference
    for (size_t i = 0; i < strlen(test_text); i++) {
      if (test_text[i] != decoded_text[i]) {
        printf("Mismatch at character %lu: Original '%c', Decoded '%c'\n", i, test_text[i], decoded_text[i]);
        break;
      }
    }
  }

  // cleanup
  free(train_text);
  free(test_text);
  free(encoded_ids);
  free(decoded_text);
  free_tokenizer(&(tokenizer.base));

  return 0;
}