#include "base.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdbool.h>

// global variables
Pair *merges = NULL;
Token **vocab = NULL;
size_t vocab_size = BASE_VOCAB_SIZE;
size_t num_merges = 0;

// dynamically selects the token type based on vocabulary size
void select_token_type(size_t vocab_size) {
  if (vocab_size <= 255) {
    typedef uint8_t Token;
  } else if (vocab_size <= 65535) {
    typedef uint16_t Token;
  } else {
    typedef uint32_t Token;
  }
}

// initializes basic utf-8 characters
void init_vocab() {
  printf("[DEBUG] Initializing vocabulary...\n");
  vocab = (uint8_t**)malloc(BASE_VOCAB_SIZE * sizeof(uint8_t*));
  if (vocab == NULL) {
    fprintf(stderr, "Memory allocation failed for vocab\n");
    exit(1);
  }
  for (size_t i = 0; i < BASE_VOCAB_SIZE; i++) {
    vocab[i] = (uint8_t*)malloc(2);
    if (vocab[i] == NULL) {
      fprintf(stderr, "Memory allocation failed for vocab[%zu]\n", i);
      exit(1);
    }
    vocab[i][0] = (uint8_t)i;
    vocab[i][1] = '\0';
  }
  printf("[DEBUG] Vocabulary initialized with %zu tokens.\n", BASE_VOCAB_SIZE);
}

// frees the vocab from the memory
void free_vocab() {
  printf("[DEBUG] Freeing vocabulary...\n");
  if (vocab == NULL) return;
  for (size_t i = 0; i < vocab_size; i++) {
    if (vocab[i] != NULL) free(vocab[i]);
  }
  free(vocab);
  vocab = NULL;
  if (merges != NULL) free(merges);
  merges = NULL;
}

// builds the new vocab using the new pairs
// builds the new vocab using the new pairs
void build_vocab(Pair *merges, size_t num_merges) {
  printf("[DEBUG] Building vocabulary with %zu merges...\n", num_merges);
  
  for (size_t i = 0; i < num_merges; i++) {
    int idx = BASE_VOCAB_SIZE + i;
    int p0 = merges[i].first;
    int p1 = merges[i].second;

    printf("[DEBUG] Before reallocating vocab: current size = %zu\n", vocab_size);
    
    uint8_t **new_vocab = realloc(vocab, (vocab_size + 1) * sizeof(uint8_t *));
    if (new_vocab == NULL) {
      fprintf(stderr, "[ERROR] Memory allocation failed for new vocab\n");
      free_vocab();
      exit(1);
    }
    vocab = new_vocab;

    printf("[DEBUG] After reallocating vocab: new size = %zu\n", vocab_size + 1);
    
    // Debug before merging tokens
    printf("[DEBUG] Merging tokens: '%s' + '%s' (indices: %d and %d)\n", vocab[p0], vocab[p1], p0, p1);
    
    // Attempt to allocate memory for the new merged token
    vocab[idx] = (uint8_t *)malloc(strlen((char *)vocab[p0]) + strlen((char *)vocab[p1]) + 1);
    if (vocab[idx] == NULL) {
      fprintf(stderr, "[ERROR] Memory allocation failed for merged token at index: %d\n", idx);
      free_vocab();
      exit(1);
    }

    sprintf((char *)vocab[idx], "%s%s", vocab[p0], vocab[p1]);
    vocab_size++;
    
    printf("[DEBUG] Created new token: '%s' at index: %d\n", vocab[idx], idx);
  }

  // Log final vocabulary after building
  printf("[DEBUG] Final vocabulary size: %zu\n", vocab_size);
  for (size_t i = 0; i < vocab_size; i++) {
    printf("[DEBUG] Vocabulary[%zu]: %s\n", i, vocab[i]);
  }
}


// replaces control characters like \n, \r, etc with unicode escape sequences
void replace_control_characters(char *str) {
  printf("[DEBUG] Replacing control characters in string...\n");
  size_t len = strlen(str);
  for (size_t i = 0; i < len; i++) {
    if (iscntrl(str[i])) {
      printf("\\u%04x", str[i]);
    } else {
      putchar(str[i]);
    }
  }
}

// renders the character form of token from merges
char *render_token(const uint8_t *token) {
  printf("[DEBUG] Rendering token: %s\n", token);
  size_t len = strlen((char *)token);
  char *rendered = (char *)malloc(len + 1);
  if (rendered == NULL) {
    fprintf(stderr, "Failed to allocate memory for rendering token\n");
    exit(1);
  }
  strcpy(rendered, (char *)token);
  replace_control_characters(rendered);
  return rendered;
}

// computes the most frequent pairs
void get_stats(const Token *ids, size_t length, PairCount *stats, size_t *stats_size, size_t capacity) {
  printf("[DEBUG] Getting stats for %zu tokens...\n", length);
  for (size_t i = 0; i < length - 1; ++i) {
    int first = ids[i];
    int second = ids[i + 1];

    // Update pair count or add a new pair.
    int found = 0;
    for (size_t j = 0; j < *stats_size; ++j) {
      if (stats[j].first == first && stats[j].second == second) {
        stats[j].count++;
        found = 1;
        break;
      }
    }
    if (!found && *stats_size < capacity) {
      stats[*stats_size].first = first;
      stats[*stats_size].second = second;
      stats[*stats_size].count = 1;
      (*stats_size)++;
      printf("[DEBUG] Found new pair: (%d, %d)\n", first, second);
    }
  }
}

// merges the most frequent pairs into a new token
void merge(const Token *ids, size_t length, Token *merged_ids, size_t *merged_length, Pair pair, Token new_token) {
  printf("[DEBUG] Merging pair: (%d, %d) into token %d\n", pair.first, pair.second, new_token);
  size_t j = 0;
  for (size_t i = 0; i < length; ++i) {
    if (i < length - 1 && ids[i] == pair.first && ids[i + 1] == pair.second) {
      merged_ids[j++] = new_token;
      i++;  // skip the next element since it's merged
    } else {
      merged_ids[j++] = ids[i];
    }
  }
  *merged_length = j;
}

// encodes the strings to tokens
void encode(const char *text, int **ids, size_t *length) {
  printf("[DEBUG] Encoding text: %s\n", text);
  size_t text_len = strlen(text);
  *ids = (int *)malloc(text_len * sizeof(int));
  if (*ids == NULL) {
    fprintf(stderr, "Memory allocation failed for encoded IDs\n");
    exit(1);
  }

  // convert text to UTF-8 byte ids
  for (size_t i = 0; i < text_len; i++) {
    (*ids)[i] = (uint8_t)text[i];
  }
  *length = text_len;

  // perform merges based on the `merges` dictionary
  size_t merged_length;
  int *temp_ids = (int *)malloc(text_len * sizeof(int));

  while (*length >= 2) {
    Pair best_pair = { -1, -1 };
    int best_idx = INT_MAX;

    for (size_t i = 0; i < *length - 1; i++) {
      Pair p = { (*ids)[i], (*ids)[i + 1] };
      for (size_t j = 0; j < num_merges; j++) {
        if (merges[j].first == p.first && merges[j].second == p.second) {
          if (j < best_idx) {
            best_pair = p;
            best_idx = j;
          }
        }
      }
    }

    if (best_pair.first == -1) break;  // no more merges possible
    merge((Token *)*ids, *length, (Token *)temp_ids, &merged_length, best_pair, BASE_VOCAB_SIZE + best_idx);
    *length = merged_length;
    memcpy(*ids, temp_ids, merged_length * sizeof(int));
    printf("[DEBUG] Merged to length: %zu\n", *length);
  }

  free(temp_ids);
}

// decodes the tokens back to string
char *decode(const int *ids, size_t length) {
  printf("[DEBUG] Decoding %zu tokens...\n", length);
  size_t buffer_size = length * 2 + 1;  // assume max UTF-8 byte size is 2
  char *decoded = (char *)malloc(buffer_size);
  if (decoded == NULL) {
    fprintf(stderr, "Memory allocation failed for decoded text\n");
    exit(1);
  }

  size_t offset = 0;
  for (size_t i = 0; i < length; i++) {
    size_t token_len = strlen((char *)vocab[ids[i]]);
    memcpy(decoded + offset, vocab[ids[i]], token_len);
    offset += token_len;
  }
  decoded[offset] = '\0';

  printf("[DEBUG] Decoded string: %s\n", decoded);
  return decoded;
}

// function to train the tokenizer
void train(const char *text, size_t target_vocab_size, bool verbose) {
  printf("[DEBUG] Starting training with target vocab size: %zu\n", target_vocab_size);
  if (target_vocab_size < BASE_VOCAB_SIZE) {
    fprintf(stderr, "Error: Target vocab size must be at least %d.\n", BASE_VOCAB_SIZE);
    return;
  }

  init_vocab();

  // Example merges and stats initialization (to be replaced with actual implementation)
  num_merges = target_vocab_size - BASE_VOCAB_SIZE;
  merges = (Pair *)malloc(num_merges * sizeof(Pair));
  if (merges == NULL) {
    fprintf(stderr, "Memory allocation failed for merges\n");
    free_vocab();
    return;
  }

  // Dummy loop to populate merges (this should be your actual implementation)
  for (size_t i = 0; i < num_merges; i++) {
    merges[i].first = BASE_VOCAB_SIZE + i;  // Example token IDs for merging
    merges[i].second = BASE_VOCAB_SIZE + i + 1;  // Example token IDs for merging
    printf("[DEBUG] Created merge pair: (%d, %d)\n", merges[i].first, merges[i].second);
  }

  // Build the vocabulary using the specified merges.
  build_vocab(merges, num_merges);

  // Log the final vocabulary
  for (size_t i = 0; i < vocab_size; i++) {
    printf("[DEBUG] Vocabulary[%zu]: %s\n", i, vocab[i]);
  }

  // Encoding and decoding for demonstration
  int *ids = NULL;
  size_t length = 0;
  encode(text, &ids, &length);
  char *decoded_text = decode(ids, length);

  // Cleanup
  free(decoded_text);
  free(ids);
  free_vocab();
}

const char *get_token_from_vocab(size_t index) {
  if (index < vocab_size) {
    return vocab[index]; // Return the token at the given index
  }
  return NULL; // Handle error case
}
