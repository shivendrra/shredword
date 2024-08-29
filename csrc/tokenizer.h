#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdlib.h>
#include <stdint.h>

typedef struct {
  char *data;
  size_t size;
} Token;

typedef struct {
  Token *tokens;
  size_t num_tokens;
} TokenList;

void initialize_tokenizer(const char *vocab_file, const char *merge_file);
TokenList tokenize(const char *input);
void free_token_list(TokenList *token_list);

#endif // TOKENIZER_H