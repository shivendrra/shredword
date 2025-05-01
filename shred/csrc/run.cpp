/*
  @run.cpp
  * main run file for testing the BPE trie-based tokenizer and vocab training
  * compile as: g++ -o run run.cpp heap.cpp threads.cpp base.cpp train.cpp -std=c++11
    - run: ./run
*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <time.h>
#include "base.h"
#include "train.h"
#include "heap.h"

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
  const char* train_file = "train.txt";
  const char* vocab_file = "vocab.txt";

  printf("Training vocabulary from %s...\n", train_file);
  time_t start_time = time(NULL);
  // train_vocab_bpe(train_file, vocab_file, 300);
  train_bpe_fast(train_file, vocab_file, 300, 3);
  time_t end_time = time(NULL);
  printf("Vocabulary training complete in %.2lf seconds.\n", difftime(end_time, start_time));

  printf("Printing vocabulary from %s...\n", vocab_file);
  TrieNode* vocab = create_node();
  // load_vocab(vocab, vocab_file);
  // print_trie(vocab);
  free_trie(vocab);

  return 0;
}