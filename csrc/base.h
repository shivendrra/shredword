#ifndef __BASE__H__
#define __BASE__H__

#include <stdbool.h>
#include <stdint.h>
#include <stddef.h>

#define BASE_VOCAB_SIZE 256
#define TARGET_VOCAB_SIZE 0

// structure to store a pair of tokens
typedef struct Pair {
  int first;
  int second;
} Pair;

typedef struct {
  int first;
  int second;
  size_t count;
} PairCount;

typedef uint8_t Token;  // default; this will change dynamically

extern Pair *merges;
extern Token **vocab;
extern size_t vocab_size;
extern size_t num_merges;

void init_vocab();
void free_vocab();
void build_vocab(Pair *merges, size_t num_merges);
void replace_control_characters(char *str);
char *render_token(const Token *token);
void get_stats(const Token *ids, size_t length, PairCount *stats, size_t *stats_size, size_t capacity);
void merge(const Token *ids, size_t length, Token *merged_ids, size_t *merged_length, Pair pair, Token new_token);
void encode(const char *text, int **ids, size_t *length);
char *decode(const int *ids, size_t length);
void train(const char *text, size_t target_vocab_size, bool verbose);
const char *get_token_from_vocab(size_t index);
void select_token_type(size_t vocab_size);

#endif  // !__BASE__H__