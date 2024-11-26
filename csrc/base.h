#ifndef __BASE__H__
#define __BASE__H__

#include <stdbool.h>  // for bool type
#include <stdint.h>   // for uint16_t and other fixed-width integers
#include <stddef.h>   // for size_t

#define BASE_VOCAB_SIZE 256  // default base vocabulary size

// store a pair of tokens
typedef struct Pair {
  int first;
  int second;
} Pair;

// store a pair with a frequency count
typedef struct PairCount {
  int first;
  int second;
  size_t count;  // occurrence count of the pair
} PairCount;

typedef uint16_t Token; // token type is int16

// global variables
extern Token **vocab;           // 2D array representing the vocabulary
extern size_t vocab_size;       // current vocabulary size
extern Pair *merges;            // array of token pairs for merging
extern size_t num_merges;       // number of merges performed

void init_vocab();
void free_vocab();
void build_vocab(Pair *merges, size_t num_merges);
void replace_control_characters(char *str);
char *render_token(const Token *token);
void get_stats(const Token *ids, size_t length, PairCount **stats, size_t *stats_size, size_t *capacity);
void merge(const Token *ids, size_t length, Token *merged_ids, size_t *merged_length, Pair pair, Token new_token);
void encode(const char *text, Token **ids, size_t *length);
char *decode(const Token *ids, size_t length);
void train(const char *text, size_t target_vocab_size, bool verbose);
const char *get_token_from_vocab(size_t index);

#endif  // !__BASE__H__