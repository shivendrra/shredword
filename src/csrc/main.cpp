#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "main.h"
#include "base.h"
#include "cache.h"

#define MIN_PAIR_FREQUENCY 5000

void init_shred(Shred* tokenizer) {
  init_tokenizer(&(tokenizer->base));
  initialize_threads();
  initialize_caches();
}

// helper function: compute simple hash string for int array
static void hash_ids(const int* ids, int ids_size, char* out_key, int out_key_size) {
  unsigned long h = 5381;
  for (int i = 0; i < ids_size; i++) {
    h = ((h << 5) + h) + ids[i]; /* h * 33 + ids[i] */
  }
  snprintf(out_key, out_key_size, "%lu", h);
}

char* decode(Shred* tokenizer, const int* ids, int ids_size) {
  if (!tokenizer || !ids || ids_size <= 0) {
    fprintf(stderr, "Error: Invalid arguments to decode.\n");
    return NULL;
  }

  initialize_token_cache(tokenizer);

  // dividing work among threads
  pthread_t threads[MAX_THREADS];
  ThreadArgs thread_args[MAX_THREADS];

  int chunk_size = ids_size / MAX_THREADS;
  for (int i = 0; i < MAX_THREADS; i++) {
    thread_args[i].tokenizer = tokenizer;
    thread_args[i].ids = ids;
    thread_args[i].start = i * chunk_size;
    thread_args[i].end = (i == MAX_THREADS - 1) ? ids_size : (i + 1) * chunk_size;
    thread_args[i].output_str = NULL;
    thread_args[i].output_int = NULL;
    thread_args[i].output_size = (size_t*)malloc(sizeof(size_t));

    if (pthread_create(&threads[i], NULL, decode_worker, &thread_args[i]) != 0) {
      fprintf(stderr, "Error: Failed to create thread %d.\n", i);
      exit(EXIT_FAILURE);
    }
  }

  // waiting for threads to finish and calculate total size
  size_t total_size = 0;
  for (int i = 0; i < MAX_THREADS; i++) {
    pthread_join(threads[i], NULL);
    total_size += *thread_args[i].output_size;
  }

  // merging results
  char* output = (char*)malloc(total_size + 1);
  if (!output) {
    fprintf(stderr, "Error: Memory allocation for output failed.\n");
    exit(EXIT_FAILURE);
  }

  char* current_pos = output;
  for (int i = 0; i < MAX_THREADS; i++) {
    memcpy(current_pos, thread_args[i].output_str, *thread_args[i].output_size);
    current_pos += *thread_args[i].output_size;
    free(thread_args[i].output_str);
    free(thread_args[i].output_size);
  }
  *current_pos = '\0';

  return output;
}

char* decode_with_cache(Shred* tokenizer, const int* ids, int ids_size) {
  if (!tokenizer || !ids || ids_size <= 0) {
    fprintf(stderr, "Error: Invalid arguments to decode.\n");
    return NULL;
  }
  initialize_token_cache(tokenizer);

  // --- LRU Caching for Decoding ---
  char key_buf[1024] = {0};
  char temp[16];

  for (int i = 0; i < ids_size; i++) {
    snprintf(temp, sizeof(temp), "%d,", ids[i]);
    strncat(key_buf, temp, sizeof(key_buf) - strlen(key_buf) - 1);
  }

  size_t cached_size = 0;
  char* cached_output = (char*) lru_cache_get(g_decode_cache, key_buf, &cached_size);
  if (cached_output) {
    return cached_output;
  }

  // --- If Not Cached, Perform Decoding ---
  pthread_t threads[MAX_THREADS];
  ThreadArgs thread_args[MAX_THREADS];
  int chunk_size = ids_size / MAX_THREADS;

  for (int i = 0; i < MAX_THREADS; i++) {
    thread_args[i].tokenizer = tokenizer;
    thread_args[i].ids = ids;
    thread_args[i].start = i * chunk_size;
    thread_args[i].end = (i == MAX_THREADS - 1) ? ids_size : (i + 1) * chunk_size;
    thread_args[i].output_size = (size_t*) malloc(sizeof(size_t));

    if (pthread_create(&threads[i], NULL, decode_worker, &thread_args[i]) != 0) {
      fprintf(stderr, "Error: Failed to create decoding thread %d.\n", i);
      exit(EXIT_FAILURE);
    }
  }

  // --- Collecting Results from Threads ---
  size_t total_size = 0;
  for (int i = 0; i < MAX_THREADS; i++) {
    pthread_join(threads[i], NULL);
    total_size += *thread_args[i].output_size;
  }

  char* output = (char*) malloc(total_size + 1);
  if (!output) {
    fprintf(stderr, "Error: Memory allocation for output failed.\n");
    exit(EXIT_FAILURE);
  }

  char* current_pos = output;
  for (int i = 0; i < MAX_THREADS; i++) {
    memcpy(current_pos, thread_args[i].output_str, *thread_args[i].output_size);
    current_pos += *thread_args[i].output_size;
    free(thread_args[i].output_str);
    free(thread_args[i].output_size);
  }
  *current_pos = '\0';

  // --- Store in Cache ---
  lru_cache_put(g_decode_cache, key_buf, output, total_size + 1);

  return output;
}

int* encode(Shred* tokenizer, const char* text, int* output_size) {
  if (!tokenizer || !text || output_size == NULL) {
    fprintf(stderr, "Error: Invalid arguments to encode.\n");
    return NULL;
  }

  size_t text_len = strlen(text);

  // Initialize IDs with input character values
  int* ids = (int*)malloc(text_len * sizeof(int));
  if (!ids) {
    fprintf(stderr, "Error: Memory allocation for ids failed.\n");
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < text_len; i++) {
    ids[i] = (unsigned char)text[i];
  }

  // Divide the input into chunks for parallel processing
  pthread_t threads[MAX_THREADS];
  ThreadArgs thread_args[MAX_THREADS];

  int chunk_size = text_len / MAX_THREADS;
  for (int i = 0; i < MAX_THREADS; i++) {
    thread_args[i].tokenizer = tokenizer;
    thread_args[i].ids = ids;
    thread_args[i].start = i * chunk_size;
    thread_args[i].end = (i == MAX_THREADS - 1) ? text_len : (i + 1) * chunk_size;
    thread_args[i].output_int = NULL;
    thread_args[i].output_str = NULL;
    thread_args[i].output_size = (size_t*)malloc(sizeof(size_t));

    if (pthread_create(&threads[i], NULL, encode_worker, &thread_args[i]) != 0) {
      fprintf(stderr, "Error: Failed to create thread %d.\n", i);
      exit(EXIT_FAILURE);
    }
  }

  // Wait for threads to finish and calculate total size
  size_t total_size = 0;
  for (int i = 0; i < MAX_THREADS; i++) {
    pthread_join(threads[i], NULL);
    total_size += *thread_args[i].output_size;
  }

  // Merge results
  int* output = (int*)malloc(total_size * sizeof(int));
  if (!output) {
    fprintf(stderr, "Error: Memory allocation for output failed.\n");
    exit(EXIT_FAILURE);
  }

  size_t offset = 0;
  for (int i = 0; i < MAX_THREADS; i++) {
    memcpy(output + offset, thread_args[i].output_int, *thread_args[i].output_size * sizeof(int));
    offset += *thread_args[i].output_size;
    free(thread_args[i].output_int);
    free(thread_args[i].output_size);
  }

  free(ids);
  *output_size = total_size;
  return output;
}

int* encode_with_cache(Shred* tokenizer, const char* text, int* output_size) {
  if (!tokenizer || !text || output_size == NULL) {
    fprintf(stderr, "Error: Invalid arguments to encode.\n");
    return NULL;
  }

  // --- LRU Caching for Encoding ---
  size_t text_key_len = strlen(text) + 1;
  char* text_key = strdup(text);
  size_t cached_size = 0;
  int* cached_output = (int*) lru_cache_get(g_encode_cache, text_key, &cached_size);
  if (cached_output) {
    *output_size = cached_size / sizeof(int);
    free(text_key);
    return cached_output;
  }

  // --- If Not Cached, Perform Encoding ---
  size_t text_len = strlen(text);
  int* ids = (int*) malloc(text_len * sizeof(int));
  if (!ids) {
    fprintf(stderr, "Error: Memory allocation for ids failed.\n");
    free(text_key);
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < text_len; i++) {
    ids[i] = (unsigned char) text[i];
  }

  pthread_t threads[MAX_THREADS];
  ThreadArgs thread_args[MAX_THREADS];
  int chunk_size = text_len / MAX_THREADS;

  for (int i = 0; i < MAX_THREADS; i++) {
    thread_args[i].tokenizer = tokenizer;
    thread_args[i].ids = ids;
    thread_args[i].start = i * chunk_size;
    thread_args[i].end = (i == MAX_THREADS - 1) ? text_len : (i + 1) * chunk_size;
    thread_args[i].output_int = NULL;
    thread_args[i].output_size = (size_t*) malloc(sizeof(size_t));

    if (pthread_create(&threads[i], NULL, encode_worker, &thread_args[i]) != 0) {
      fprintf(stderr, "Error: Failed to create encoding thread %d.\n", i);
      free(text_key);
      exit(EXIT_FAILURE);
    }
  }

  // --- Collecting Results from Threads ---
  size_t total_size = 0;
  for (int i = 0; i < MAX_THREADS; i++) {
    pthread_join(threads[i], NULL);
    total_size += *thread_args[i].output_size;
  }

  int* output = (int*) malloc(total_size * sizeof(int));
  if (!output) {
    fprintf(stderr, "Error: Memory allocation for output failed.\n");
    free(text_key);
    exit(EXIT_FAILURE);
  }

  size_t offset = 0;
  for (int i = 0; i < MAX_THREADS; i++) {
    memcpy(output + offset, thread_args[i].output_int, *thread_args[i].output_size * sizeof(int));
    offset += *thread_args[i].output_size;
    free(thread_args[i].output_int);
    free(thread_args[i].output_size);
  }

  free(ids);
  *output_size = total_size;

  // --- Store in Cache ---
  lru_cache_put(g_encode_cache, text_key, output, total_size * sizeof(int));
  free(text_key);

  return output;
}

void save_model(const Shred* tokenizer, const char* file_path) {
  if (!tokenizer || !file_path) {
    fprintf(stderr, "Error: Invalid arguments passed to save_model.\n");
    return;
  }
  save_tokenizer(&(tokenizer->base), file_path);
}

void load_model(Shred* tokenizer, const char* model_file) {
  if (!tokenizer || !model_file) {
    fprintf(stderr, "Error: Invalid arguments passed to load_model.\n");
    return;
  }
  load_tokenizer(&(tokenizer->base), model_file);
}

// function to export vocabulary as a serialized string
char* export_merges(const Shred* tokenizer) {
  if (!tokenizer) {
    printf("Error: tokenizer pointer is null.\n");
    return NULL;
  }

  size_t buffer_size = MAX_MERGES * 32; // rough estimate for output buffer memory
  char* output = (char*)malloc(buffer_size);
  if (!output) {
    fprintf(stderr, "Error: Unable to allocate memory for output.\n");
    return NULL;
  }

  int offset = 0;
  memset(output, 0, buffer_size);
  for (int i = 0; i < tokenizer->base.merge_count; i++) {
    Pair pair = tokenizer->base.merges[i].pair;
    int index = tokenizer->base.merges[i].idx;

    if (pair.idx1 >= 0 && pair.idx2 >= 0) { // only include valid pairs
      offset += snprintf(output + offset, buffer_size - offset, "(%d, %d) %d\n", pair.idx1, pair.idx2, index);
      if (offset >= buffer_size) {
        fprintf(stderr, "Error: Output buffer overflow.\n");
        free(output);
        return NULL;
      }
    }
  }

  return output;
}

void free_string(char* string) {
  free(string);
}