#include "tokenizer.h"
#include <stdlib.h>
#include <string.h>
#include <limits.h>

void init_bpe_encoder(BpeEncoder *encoder, BpeEntry *entries, size_t num_entries) {
  encoder->entries = entries;
  encoder->num_entries = num_entries;
}

void free_bpe_encoder(BpeEncoder *encoder) {
  free(encoder->entries);
}

Rank get_rank(const uint8_t *seq, size_t len, const BpeEncoder *encoder) {
  for (size_t i = 0; i < encoder->num_entries; i++) {
    if (encoder->entries[i].length == len && memcmp(encoder->entries[i].data, seq, len) == 0) {
      return encoder->entries[i].rank;
    }
  }
  return UINT_MAX;
}

size_t *byte_pair_merge(const uint8_t *piece, size_t len, const BpeEncoder *encoder, size_t *out_len) {
  size_t *parts = (size_t *)malloc((len + 2) * sizeof(size_t));
  if (!parts) return NULL;
  for (size_t i = 0; i < len - 1; i++) {
    parts[i] = get_rank(&piece[i], 2, encoder);
  }
  parts[len - 1] = UINT_MAX;
  *out_len = len;

  while (1) {
    size_t min_rank = UINT_MAX;
    size_t min_idx = UINT_MAX;

    for (size_t i = 0; i < *out_len - 1; i++) {
      if (parts[i] < min_rank) {
        min_rank = parts[i];
        min_idx = i;
      }
    }

    if (min_rank == UINT_MAX) break;

    for (size_t i = min_idx + 1; i < *out_len - 1; i++) {
      parts[i] = parts[i + 1];
    }
    (*out_len)--;
  }
  return parts;
}

Rank *byte_pair_encode(const uint8_t *piece, size_t len, const BpeEncoder *encoder, size_t *out_len) {
  size_t merged_len;
  size_t *merged_parts = byte_pair_merge(piece, len, encoder, &merged_len);
  
  if (!merged_parts) return NULL;
  Rank *encoded = (Rank *)malloc((merged_len - 1) * sizeof(Rank));
  if (!encoded) {
    free(merged_parts);
    return NULL;
  }

  for (size_t i = 0; i < merged_len - 1; i++) {
    encoded[i] = get_rank(&piece[merged_parts[i]], 2, encoder);
  }

  *out_len = merged_len - 1;
  free(merged_parts);
  return encoded;
}

uint8_t **byte_pair_split(const uint8_t *piece, size_t len, const BpeEncoder *encoder, size_t *out_len) {
  size_t merged_len;
  size_t *merged_parts = byte_pair_merge(piece, len, encoder, &merged_len);
  
  if (!merged_parts) return NULL;
  uint8_t **split = (uint8_t **)malloc((merged_len - 1) * sizeof(uint8_t *));
  if (!split) {
    free(merged_parts);
    return NULL;
  }

  for (size_t i = 0; i < merged_len - 1; i++) {
    size_t start = merged_parts[i];
    size_t end = (i + 1 < merged_len) ? merged_parts[i + 1] : len;

    split[i] = (uint8_t *)malloc((end - start + 1) * sizeof(uint8_t));
    memcpy(split[i], &piece[start], end - start);
    split[i][end - start] = '\0';
  }

  *out_len = merged_len - 1;
  free(merged_parts);
  return split;
}

void free_encoded_output(Rank *encoded) {
  free(encoded);
}

void free_split_output(uint8_t **split, size_t num_pieces) {
  for (size_t i = 0; i < num_pieces; i++) {
    free(split[i]);
  }
  free(split);
}