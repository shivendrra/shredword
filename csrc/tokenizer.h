#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <stdint.h>
#include <stddef.h>

typedef uint32_t Rank;
typedef struct BpeEntry {
  uint8_t *data;  // Byte sequence (e.g., "ab")
  size_t length;  // Length of the byte sequence
  Rank rank;      // Rank value associated with the byte sequence
} BpeEntry;

typedef struct BpeEncoder {
  BpeEntry *entries; // Array of BPE entries
  size_t num_entries; // Number of entries in the array
} BpeEncoder;

void init_bpe_encoder(BpeEncoder *encoder, BpeEntry *entries, size_t num_entries);
void free_bpe_encoder(BpeEncoder *encoder);
Rank *byte_pair_encode(const uint8_t *piece, size_t len, const BpeEncoder *encoder, size_t *out_len);
uint8_t **byte_pair_split(const uint8_t *piece, size_t len, const BpeEncoder *encoder, size_t *out_len);
void free_encoded_output(Rank *encoded);
void free_split_output(uint8_t **split, size_t num_pieces);


#endif // TOKENIZER_H