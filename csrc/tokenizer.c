#include "tokenizer.h"

size_t get_rank(size_t** ranks, size_t ranks_size, const unsigned char* piece, size_t start, size_t length) {
  for (size_t i = 0; i < ranks_size; i++) {
    if (memcmp(ranks[i], piece + start, length) == 0) {
      return i;
    }
  }
  return SIZE_MAX; // Rank::MAX equivalent in Rust
}

// Helper function to merge byte pairs based on rank
Pair* byte_pair_merge(const unsigned char* piece, size_t piece_len, size_t** ranks, size_t ranks_size, size_t* merged_size) {
  Pair* parts = (Pair*)malloc((piece_len + 2) * sizeof(Pair));
  if (!parts) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  size_t min_rank = SIZE_MAX;
  size_t min_rank_index = SIZE_MAX;

  // Populate parts array
  for (size_t i = 0; i < piece_len - 1; i++) {
    size_t rank = get_rank(ranks, ranks_size, piece, i, 2);
    parts[i].index = i;
    parts[i].rank = rank;
    if (rank < min_rank) {
      min_rank = rank;
      min_rank_index = i;
    }
  }

  parts[piece_len - 1].index = piece_len - 1;
  parts[piece_len - 1].rank = SIZE_MAX;
  parts[piece_len].index = piece_len;
  parts[piece_len].rank = SIZE_MAX;

  size_t parts_len = piece_len + 1;

  // Merge pairs based on minimum rank
  while (min_rank != SIZE_MAX) {
    size_t i = min_rank_index;

    if (i > 0) {
      parts[i - 1].rank = get_rank(ranks, ranks_size, piece, parts[i - 1].index, parts[i + 1].index - parts[i - 1].index);
    }

    parts[i].rank = get_rank(ranks, ranks_size, piece, parts[i].index, parts[i + 1].index - parts[i].index);
    for (size_t j = i + 1; j < parts_len - 1; j++) {
      parts[j] = parts[j + 1];
    }
    parts_len--;

    min_rank = SIZE_MAX;
    min_rank_index = SIZE_MAX;
    for (size_t k = 0; k < parts_len - 1; k++) {
      if (parts[k].rank < min_rank) {
        min_rank = parts[k].rank;
        min_rank_index = k;
      }
    }
  }

  *merged_size = parts_len;
  return parts;
}

// Function to encode a piece of text using byte pair encoding
size_t* byte_pair_encode(const unsigned char* piece, size_t piece_len, size_t** ranks, size_t ranks_size, size_t* encoded_size) {
  size_t merged_size;
  Pair* merged = byte_pair_merge(piece, piece_len, ranks, ranks_size, &merged_size);

  size_t* result = (size_t*)malloc((merged_size - 1) * sizeof(size_t));
  if (!result) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (size_t i = 0; i < merged_size - 1; i++) {
    size_t rank = get_rank(ranks, ranks_size, piece, merged[i].index, merged[i + 1].index - merged[i].index);
    result[i] = rank;
  }

  free(merged);
  *encoded_size = merged_size - 1;
  return result;
}

// Function to split a piece of text into sub-pieces using byte pair encoding
unsigned char** byte_pair_split(const unsigned char* piece, size_t piece_len, size_t** ranks, size_t ranks_size, size_t* split_size) {
  size_t merged_size;
  Pair* merged = byte_pair_merge(piece, piece_len, ranks, ranks_size, &merged_size);

  unsigned char** result = (unsigned char**)malloc((merged_size - 1) * sizeof(unsigned char*));
  if (!result) {
    fprintf(stderr, "Memory allocation failed\n");
    exit(1);
  }

  for (size_t i = 0; i < merged_size - 1; i++) {
    size_t len = merged[i + 1].index - merged[i].index;
    result[i] = (unsigned char*)malloc(len * sizeof(unsigned char));
    memcpy(result[i], piece + merged[i].index, len);
  }

  free(merged);
  *split_size = merged_size - 1;
  return result;
}

// Function to free the memory allocated for encoded results
void free_encoded_result(size_t* result) {
  if (result) free(result);
}

// Function to free the memory allocated for split results
void free_split_result(unsigned char** result, size_t size) {
  if (result) {
    for (size_t i = 0; i < size; i++) {
      if (result[i]) free(result[i]);
    }
    free(result);
  }
}