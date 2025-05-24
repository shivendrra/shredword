#include <stdio.h>
#include <stdlib.h>
#include "inc/hash.h"
#include "inc/heap.h"
#include "trainer/bpe.h"

Trainer* create_trainer(const BPEConfig* config) {
  if (config == NULL) {
    fprintf(stderr, "[ERROR]\t Config pointer is NULL\n");
    exit(EXIT_FAILURE);
  }
  Trainer* trainer = (Trainer*)malloc(sizeof(Trainer));
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t Couldn't allocate Memory to Trainer\n");
    exit(EXIT_FAILURE);
  }
  trainer->config = *config;
  // defaulting character coverage value
  if (trainer->config.character_coverage <= 0.0 || trainer->config.character_coverage >= 1.0) {
    trainer->config.character_coverage = 0.995;
  }
  // defaulting min pair freq value
  if (trainer->config.min_pair_freq == 0) {
    trainer->config.min_pair_freq = 100;
  }
  trainer->num_merges = 0;
  trainer->merge_ops = (PairKey*)malloc(sizeof(PairKey) * trainer->config.target_vocab_size);
  heap_init(&trainer->heap, MIN_HEAP_SIZE);   // initialized heap
  printf("[INFO]\t BPE trainer initialized. Heap initialized successfully.\n");
  return trainer;
}

void bpe_trainer_destroy(Trainer* trainer) {
  if (!trainer) {
    fprintf(stderr, "[ERROR]\t No Trainer pointer found to destroy!\n");
    exit(EXIT_FAILURE);
  }
  // freeing corpus arrays (if loaded)
  free(trainer->corpus.words);
  free(trainer->corpus.word_counts);
  heap_free(&trainer->heap);  // free heap storage
  free(trainer);    // finally, free the trainer struct itself
}

/**
  @brief Reads a text file line by line and counts the frequency of each unique token,
        where tokens are separated by tabs, carriage returns, or newlines.
  * This approach uses strtok to split each line in-place, which is memory efficient
  * because it does not require allocating additional memory for each token.
  * By modifying the original line buffer, we avoid unnecessary string copies,
  * making it well-suited for processing large files or lines with many tokens.
  @param input_path [in] Path to the input file to be processed.
  @return void
*/
int bpe_load_corpus(Trainer* trainer, char* input_path) {
  if (!trainer || !input_path) {
    fprintf(stderr, "[ERROR]\t NULL trainer & input path pointers\n");
    exit(EXIT_FAILURE);
  }
  StrMap freq_map;
  strmap_init(&freq_map, INITIAL_STR_BUFFER);
  FILE* fp = fopen(input_path, "r");
  if (!fp) {
    fprintf(stderr, "[ERROR]\t Couldn't open the file\n");
    exit(EXIT_FAILURE);
  }
  char line[INITIAL_STR_BUFFER];
  while (fgets(line, sizeof(line), fp)) {
    char* tok = strtok(line, "\t\r\n");
    while (tok) {
      strmap_increment(&freq_map, tok);
      tok = strtok(NULL, "\t\r\n");
    }
  }
  fclose(fp);

  // building character histogram
  StrMap char_map;
  strmap_init(&char_map, INITIAL_VOCAB_SIZE);
  // strmap_iter(&char_map, );

}