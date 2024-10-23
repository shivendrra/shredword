#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <stdint.h>
#include <stdbool.h>

#define VOCAB_SIZE 256  // Define the size of the initial vocabulary

Pair *merges = NULL;
uint8_t **vocab = NULL;
size_t vocab_size = VOCAB_SIZE;
size_t num_merges = 0;

void init_vocab() {
  vocab = (uint8_t **)malloc(VOCAB_SIZE * sizeof(uint8_t *));
  if (vocab == NULL) {
    fprintf(stderr, "Memory allocation failed for vocab\n");
    exit(1);
  }
  for (size_t i = 0; i < VOCAB_SIZE; i++) {
    vocab[i] = (uint8_t *)malloc(2);
    if (vocab[i] == NULL) {
      fprintf(stderr, "Memory allocation failed for vocab[%zu]\n", i);
      exit(1);
    }
    vocab[i][0] = (uint8_t)i;
    vocab[i][1] = '\0';
  }
}

void free_vocab() {
  if (vocab == NULL) return;
  for (size_t i = 0; i < vocab_size; i++) {
    if (vocab[i] != NULL) free(vocab[i]);
  }
  free(vocab);
  vocab = NULL;
  if (merges != NULL) free(merges);
  merges = NULL;
}

void build_vocab(Pair *merges, size_t num_merges) {
  for (size_t i = 0; i < num_merges; i++) {
    int idx = VOCAB_SIZE + i;
    int p0 = merges[i].first;
    int p1 = merges[i].second;

    vocab[idx] = (uint8_t *)malloc(strlen((char *)vocab[p0]) + strlen((char *)vocab[p1]) + 1);
    if (vocab[idx] == NULL) {
      fprintf(stderr, "Memory allocation failed for merged token\n");
      exit(1);
    }
    sprintf((char *)vocab[idx], "%s%s", vocab[p0], vocab[p1]);
  }
}

void replace_control_characters(char *str) {
  size_t len = strlen(str);
  for (size_t i = 0; i < len; i++) {
    if (iscntrl(str[i])) {
      printf("\\u%04x", str[i]);
    } else {
      putchar(str[i]);
    }
  }
}

char *render_token(const uint8_t *token) {
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

void merge(int *ids, size_t *length, Pair pair, int new_token) {
  if (pair.first >= VOCAB_SIZE || pair.second >= VOCAB_SIZE) {
    fprintf(stderr, "Error: Trying to merge out-of-bounds tokens %d and %d\n", pair.first, pair.second);
    return;  // Prevent merging out-of-bounds
  }

  int *new_ids = (int *)malloc(*length * sizeof(int));
  if (new_ids == NULL) {
    fprintf(stderr, "Failed to allocate memory for merged IDs\n");
    exit(1);
  }

  size_t j = 0;
  for (size_t i = 0; i < *length; i++) {
    if (i + 1 < *length && ids[i] == pair.first && ids[i + 1] == pair.second) {
      new_ids[j++] = new_token;
      i++;  // Skip the next token as it has been merged
    } else {
      new_ids[j++] = ids[i];
    }
  }

  *length = j;  // Update the length to the new merged length
  memcpy(ids, new_ids, j * sizeof(int));  // Copy merged tokens back to ids
  free(new_ids);
}


void get_stats(int *ids, size_t length, int **counts, size_t *count_size) {
  *counts = (int *)calloc(VOCAB_SIZE * VOCAB_SIZE, sizeof(int));  // Ensure sufficient allocation
  if (*counts == NULL) {
    fprintf(stderr, "Failed to allocate memory for counts\n");
    return; // Ensure to return if memory allocation fails
  }
  *count_size = VOCAB_SIZE * VOCAB_SIZE;

  for (size_t i = 0; i < length - 1; i++) {
    if (ids[i] >= VOCAB_SIZE || ids[i + 1] >= VOCAB_SIZE) {
      fprintf(stderr, "ID out of bounds: %d, %d\n", ids[i], ids[i + 1]);
      continue; // Prevent access violations
    }
    (*counts)[ids[i] * VOCAB_SIZE + ids[i + 1]]++;
  }
}

void encode(const char *text, int **ids, size_t *length) {
  *length = strlen(text);
  *ids = (int *)malloc(*length * sizeof(int));
  if (*ids == NULL) {
    fprintf(stderr, "Failed to allocate memory for encoding\n");
    exit(1);
  }

  for (size_t i = 0; i < *length; i++) {
    (*ids)[i] = (int)text[i];
  }
}

char *decode(const int *ids, size_t length) {
  char *text = (char *)malloc(length + 1);
  if (text == NULL) {
    fprintf(stderr, "Failed to allocate memory for decoding\n");
    exit(1);
  }

  for (size_t i = 0; i < length; i++) {
    text[i] = (char)ids[i];
  }
  text[length] = '\0';
  return text;
}

void train(const char *text, size_t vocab_size, bool verbose) {
  size_t length;
  int *ids;
  encode(text, &ids, &length);
  
  if (ids == NULL) {
    perror("Failed to allocate memory for ids");
    return;
  }

  printf("Encoding complete. Length: %zu\n", length);  // Debugging output

  for (size_t i = 0; i < vocab_size - VOCAB_SIZE; i++) {
    int *counts;
    size_t count_size;

    get_stats(ids, length, &counts, &count_size);
    
    if (counts == NULL) {  // Check if counts were allocated properly
      perror("Failed to allocate memory for counts");
      free(ids);  // Free ids before return
      return;
    }

    int max_count = 0;
    Pair best_pair = {0, 0};

    // Find the best pair to merge
    for (size_t j = 0; j < count_size; j++) {
      if (counts[j] > max_count) {
        max_count = counts[j];
        best_pair.first = j / VOCAB_SIZE;
        best_pair.second = j % VOCAB_SIZE;
      }
    }
    printf("Merging: %d and %d\n", best_pair.first, best_pair.second);  // Debugging output

    merge(ids, &length, best_pair, VOCAB_SIZE + i);
    if (verbose) {
      printf("Merged %d and %d into %d\n", best_pair.first, best_pair.second, VOCAB_SIZE + i);
    } 
    free(counts);  // Free counts after use
  }
  free(ids);  // Free ids after the training loop
}

void save_model(const char *filename) {
  FILE *file = fopen(filename, "w");
  if (!file) {
    perror("Failed to open file for saving");
    return;
  }
  fprintf(file, "minibpe v1\n");
  for (size_t i = 0; i < num_merges; i++) {
    fprintf(file, "%d %d\n", merges[i].first, merges[i].second);
  }
  fclose(file);
}

void load_model(const char *filename) {
  FILE *file = fopen(filename, "r");
  if (!file) {
    perror("Failed to open file for loading");
    return;
  }
  
  merges = (Pair *)malloc(100 * sizeof(Pair));
  if (!merges) {
    perror("Failed to allocate memory for merges");
    fclose(file);
    return;
  }

  fscanf(file, "minibpe v1\n");
  while (!feof(file)) {
    int first, second;
    if (fscanf(file, "%d %d\n", &first, &second) != 2) {
      break;
    }
    merges[num_merges++] = (Pair){first, second};

    if (num_merges % 100 == 0) {
      merges = realloc(merges, (num_merges + 100) * sizeof(Pair));
      if (!merges) {
        perror("Failed to reallocate memory for merges");
        fclose(file);
        return;
      }
    }
  }
  fclose(file);
}