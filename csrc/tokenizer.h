#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
  size_t index;
  size_t rank;
} Pair;

Pair* byte_pair_merge(const unsigned char* piece, size_t piece_len, size_t** ranks, size_t ranks_size, size_t* merged_size);
size_t* byte_pair_encode(const unsigned char* piece, size_t piece_len, size_t** ranks, size_t ranks_size, size_t* encoded_size);
unsigned char** byte_pair_split(const unsigned char* piece, size_t piece_len, size_t** ranks, size_t ranks_size, size_t* split_size);
void free_encoded_result(size_t* result);
void free_split_result(unsigned char** result, size_t size);

#endif