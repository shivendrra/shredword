#include "base.h"
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <stdbool.h>
#include <assert.h>

Token **vocab = NULL;
size_t vocab_size = 0;
Pair *merges = NULL;
size_t num_merges = 0;

// Function to initialize vocabulary
void init_vocab() {
  vocab_size = BASE_VOCAB_SIZE;
  vocab = (Token **)malloc(vocab_size * sizeof(Token *));
  if (!vocab) {
    fprintf(stderr, "Memory allocation for vocab failed\n");
    exit(EXIT_FAILURE);
  }

  for (size_t i = 0; i < vocab_size; i++) {
    vocab[i] = (Token *)malloc(sizeof(Token));
    if (!vocab[i]) {
      fprintf(stderr, "Memory allocation for vocab[%zu] failed\n", i);
      exit(EXIT_FAILURE);
    }
    vocab[i][0] = (Token)i; // Each token is initialized to its index
  }
}

// Function to free allocated vocabulary
void free_vocab() {
  if (vocab) {
    for (size_t i = 0; i < vocab_size; i++) {
      free(vocab[i]);
    }
    free(vocab);
    vocab = NULL;
  }
}

// Function to build the vocabulary from merges
void build_vocab(Pair *merges, size_t num_merges) {
  // Create a hashmap-like structure to track token pairs
  for (size_t i = 0; i < num_merges; i++) {
    int first = merges[i].first;
    int second = merges[i].second;

    // Create a new token that is the combination of the two tokens
    if (first < vocab_size && second < vocab_size) {
      vocab[vocab_size] = (Token *)malloc(sizeof(Token) * 2);
      if (!vocab[vocab_size]) {
        fprintf(stderr, "[ERROR] Memory allocation for merged vocab[%zu] failed\n", vocab_size);
        exit(EXIT_FAILURE);
      }
      vocab[vocab_size][0] = (Token)first;
      vocab[vocab_size][1] = (Token)second;
      vocab_size++;
    } else {
      fprintf(stderr, "Merge indices out of bounds: first: %d, second: %d\n", first, second);
    }
  }
}

// Function to replace control characters
void replace_control_characters(char *str) {
  while (*str) {
    if (*str < 32 && *str != 9) { // Replacing control characters except tab
      *str = ' ';
    }
    str++;
  }
}

// Function to render a token as a string
char *render_token(const Token *token) {
  char *str = (char *)malloc(2);
  if (!str) {
    fprintf(stderr, "Memory allocation for rendered token failed\n");
    exit(EXIT_FAILURE);
  }
  str[0] = (char)(*token);
  str[1] = '\0';
  return str;
}

// Function to get statistics of token pairs
void get_stats(const Token *ids, size_t length, PairCount **stats, size_t *stats_size, size_t *capacity) {
  // Initialize if stats is NULL
  if (*stats == NULL) {
    *capacity = 10; // Starting capacity
    *stats = malloc(*capacity * sizeof(PairCount));
    if (*stats == NULL) {
      perror("Failed to allocate memory for stats");
      exit(EXIT_FAILURE);
    }
    memset(*stats, 0, *capacity * sizeof(PairCount)); // Clear previous stats
  }
  
  // Count pairs
  for (size_t i = 0; i < length - 1; i++) {
    PairCount *pair_count = NULL;

    // Check if the pair exists
    for (size_t j = 0; j < *stats_size; j++) {
      if ((*stats)[j].first == ids[i] && (*stats)[j].second == ids[i + 1]) {
        pair_count = &(*stats)[j];
        break;
      }
    }

    if (pair_count == NULL) {
      if (*stats_size < *capacity) {
        (*stats)[*stats_size].first = ids[i];
        (*stats)[*stats_size].second = ids[i + 1];
        (*stats)[*stats_size].count = 1;
        (*stats_size)++;
      } else {
        // Increase capacity
        *capacity *= 2; // Double the capacity
        PairCount *new_stats = realloc(*stats, *capacity * sizeof(PairCount));
        if (new_stats == NULL) {
          perror("Failed to reallocate memory for stats");
          exit(EXIT_FAILURE);
        }
        *stats = new_stats; // Update pointer to new memory location
        memset(*stats + *stats_size, 0, (*capacity - *stats_size) * sizeof(PairCount)); // Clear new space

        // Add new pair after resizing
        (*stats)[*stats_size].first = ids[i];
        (*stats)[*stats_size].second = ids[i + 1];
        (*stats)[*stats_size].count = 1;
        (*stats_size)++;
      }
    } else {
      pair_count->count++;
    }
  }
}

// Function to merge token pairs
void merge(const Token *ids, size_t length, Token *merged_ids, size_t *merged_length, Pair pair, Token new_token) {
  size_t j = 0;
  for (size_t i = 0; i < length - 1; i++) {
    if (ids[i] == pair.first && ids[i + 1] == pair.second) {
      merged_ids[j++] = new_token;
      i++; // Skip the next token as it has been merged
    } else {
      merged_ids[j++] = ids[i];
    }
  }
  if (length > 0) {
    merged_ids[j++] = ids[length - 1]; // Add the last token
  }
  *merged_length = j;
}

// Function to encode text into tokens
void encode(const char *text, Token **ids, size_t *length) {
  *length = strlen(text);
  
  *ids = (Token *)malloc((*length) * sizeof(Token));
  if (!*ids) {
    fprintf(stderr, "Memory allocation for ids failed\n");
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < *length; i++) {
    (*ids)[i] = (Token)(unsigned char)text[i]; // Convert to unsigned char to fit Token type
  }
}

// Function to decode tokens back into text
char *decode(const Token *ids, size_t length) {
  char *decoded_text = (char *)malloc(length + 1);
  if (!decoded_text) {
    fprintf(stderr, "Memory allocation for decoded text failed\n");
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < length; i++) {
    decoded_text[i] = (char)ids[i];
  }
  decoded_text[length] = '\0'; // Null-terminate the string
  return decoded_text;
}

// Function to train the tokenizer
// Update the train function
void train(const char *text, size_t target_vocab_size, bool verbose) {
  assert(target_vocab_size >= BASE_VOCAB_SIZE);
  num_merges = target_vocab_size - BASE_VOCAB_SIZE;
  printf("[DEBUG] Training tokenizer, target_vocab_size: %zu, num_merges: %zu\n", target_vocab_size, num_merges);
  
  // Initialize the vocabulary
  init_vocab();
  
  // Encode the input text
  Token *ids;
  size_t length;
  encode(text, &ids, &length);

  // Correctly allocate stats
  size_t capacity = (length / 2);
  PairCount *stats = (PairCount *)malloc(capacity * sizeof(PairCount)); // Allocate for max possible pairs
  if (!stats) {
    fprintf(stderr, "Memory allocation for stats failed\n");
    exit(EXIT_FAILURE);
  }
  size_t stats_size = 0; // Initialize stats_size to 0

  for (size_t i = 0; i < num_merges; i++) {
    // Call get_stats with correct pointer types
    get_stats(ids, length, &stats, &stats_size, &capacity); // Pass &stats to get_stats

    if (stats_size == 0) {
      break; // No more pairs to merge
    }

    // Find the pair with the highest frequency
    Pair max_pair = {0, 0};
    size_t max_count = 0;

    for (size_t j = 0; j < stats_size; j++) {
      if (stats[j].count > max_count) {
        max_count = stats[j].count;
        max_pair.first = stats[j].first;
        max_pair.second = stats[j].second;
      }
    }

    // Create a new token for the merged pair
    Token new_token = BASE_VOCAB_SIZE + i;

    // Merge the pairs
    Token *merged_ids = (Token *)malloc((length) * sizeof(Token)); // Allocate for merged tokens
    if (!merged_ids) {
      fprintf(stderr, "Memory allocation for merged_ids failed\n");
      exit(EXIT_FAILURE);
    }
    size_t merged_length;
    merge(ids, length, merged_ids, &merged_length, max_pair, new_token);

    // Update length for next iteration
    length = merged_length;
    free(ids); // Free previous ids
    ids = merged_ids; // Update to the merged tokens

    printf("[DEBUG] Current Vocabulary after merging:\n");
    for (size_t i = 0; i < length; i++) {
      const char *token_str = render_token(&ids[i]); // Assuming ids[i] is a Token
      if (token_str) {
        if (strlen(token_str) > 0) {
          printf("%zu -> %s\n", i, token_str);
        } else {
          printf("%zu -> EMPTY (token has no visible representation)\n", i);
        }
        free((void *)token_str); // Free the rendered token after use
      } else {
        printf("%zu -> NULL (not rendered)\n", i);
      }
    }
  }

  free(ids);
  free(stats);
  // free_vocab();
}

const char *get_token_from_vocab(size_t index) {
  if (index < vocab_size) {
    return render_token(&vocab[index][0]);
  }
  return NULL;
}