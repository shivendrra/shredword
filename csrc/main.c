/*
  main run file for base.c & base.h tokenizer
  run: gcc main.c base.c -o main .\main
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "base.h"

// Function to read text from a file
char *read_file(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (file == NULL) {
    fprintf(stderr, "Error: Could not open file %s\n", filename);
    return NULL;
  }

  fseek(file, 0, SEEK_END);
  long length = ftell(file);
  fseek(file, 0, SEEK_SET);

  char *text = (char *)malloc(length + 1);
  if (text == NULL) {
    fprintf(stderr, "Memory allocation failed for file content\n");
    fclose(file);
    return NULL;
  }

  fread(text, 1, length, file);
  text[length] = '\0'; // Null-terminate the string
  fclose(file);
  return text;
}

int main() {
  const char *file_path = "../new.txt"; // Change to your file path
  char *text = read_file(file_path);
  if (text == NULL) {
    fprintf(stderr, "Reading file failed!\n");
    return 1; // Exit if file read failed
  }
  printf("File read successfully\n");

  init_vocab();
  size_t target_vocab_size = 500;
  train(text, target_vocab_size, true);

  // Print the vocabulary (int -> str)
  printf("Vocabulary:\n");
  for (size_t i = 0; i < target_vocab_size; i++) {
    const char *token = get_token_from_vocab(i);
    if (token) { // Check for NULL token
      printf("%zu -> %s\n", i, token);
      free((void *)token); // Free the rendered token after use
    } else {
      printf("%zu -> NULL\n", i); // Token was NULL
    }
  }

  const char *sample_text = text;
  Token *encoded_ids = NULL;  // Change to Token type
  size_t encoded_length;

  encode(sample_text, &encoded_ids, &encoded_length);
  
  char *decoded_text = decode(encoded_ids, encoded_length);
  
  size_t original_length = strlen(sample_text);
  size_t decoded_length = strlen(decoded_text);
  
  float compression_ratio = (float)original_length / (float)decoded_length;

  printf("Length of encoded tokens: %zu\n", encoded_length);
  printf("Length of decoded text: %zu\n", decoded_length);
  printf("Compression ratio (original/decoded): %.2f\n", compression_ratio);

  // Clean up
  free(encoded_ids);
  free(decoded_text);
  free_vocab();
  free(text); // Free the text read from the file

  return 0;
}