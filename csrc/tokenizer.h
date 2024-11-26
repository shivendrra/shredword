#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stddef.h>  // For size_t
#include <stdint.h>  // For uint8_t
#include <stdbool.h> // For bool type

// Define constants
#define VOCAB_SIZE 256

// Structure for storing merge pairs
typedef struct {
  int first;
  int second;
} Pair;

// Global variables (merges and vocab)
extern Pair *merges;
extern uint8_t **vocab;
extern size_t vocab_size;
extern size_t num_merges;

void init_vocab();
void free_vocab();
void build_vocab(Pair *merges, size_t num_merges);
void replace_control_characters(char *str);
char *render_token(const uint8_t *token);
void merge(int *ids, size_t *length, Pair pair, int new_token);
void get_stats(int *ids, size_t length, int **counts, size_t *count_size);
void encode(const char *text, int **ids, size_t *length);
char *decode(const int *ids, size_t length);
void train(const char *text, size_t vocab_size, bool verbose);
void save_model(const char *filename);
void load_model(const char *filename);

#endif // TOKENIZER_H