#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "main.h"
#include "base.h"
#include "cache.h"

void init_shred(Shred* tokenizer) {
  init_tokenizer(&(tokenizer->base));
  initialize_threads();
  initialize_caches();
}

void consistency_check(Shred* tokenizer, int n_merges) {
  // consistency check
  // matches all the merged pairs with the verbose outputs pair by pair & logs the errors
  printf("\nPerforming consistency check for merges...\n");
  int mismatch_count = 0;
  for (int i = 0; i < n_merges; i++) {
    if (tokenizer->base.merges[i].pair.idx1 != tokenizer->base.merges[i].pair.idx1 ||
      tokenizer->base.merges[i].pair.idx2 != tokenizer->base.merges[i].pair.idx2) {
      printf("Mismatch at merge %d: Expected (%d, %d), Found (%d, %d)\n",
             i + 1,
             tokenizer->base.merges[i].pair.idx1, tokenizer->base.merges[i].pair.idx2,
             tokenizer->base.merges[i].pair.idx1, tokenizer->base.merges[i].pair.idx2);
      mismatch_count++;
    }
  }

  if (mismatch_count == 0) {
    printf("All merges are consistent between the training logic and tokenizer state.\n");
  } else {
    printf("Consistency check failed: %d mismatches found in the merges.\n", mismatch_count);
  }
}

// helper function: compute simple hash string for int array
static void hash_ids(const int* ids, int ids_size, char* out_key, int out_key_size) {
  unsigned long h = 5381;
  for (int i = 0; i < ids_size; i++) {
    h = ((h << 5) + h) + ids[i]; /* h * 33 + ids[i] */
  }
  snprintf(out_key, out_key_size, "%lu", h);
}

void train_with_cache(Shred* tokenizer, const char* text, int vocab_size) {
  assert(vocab_size >= VOCAB_SIZE);
  int n_merges = vocab_size - VOCAB_SIZE;
  size_t text_len = strlen(text);
  unsigned char* text_bytes = (unsigned char*) malloc((text_len + 1) * sizeof(unsigned char));
  if (!text_bytes) {
    fprintf(stderr, "Error: Memory allocation for text_bytes failed.\n");
    exit(EXIT_FAILURE);
  }
  memcpy(text_bytes, text, text_len);
  text_bytes[text_len] = '\0';
  int* ids = (int*) malloc(text_len * sizeof(int));
  if (!ids) {
    fprintf(stderr, "Error: Memory allocation for ids failed.\n");
    free(text_bytes);
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < text_len; i++) {
    ids[i] = text_bytes[i];
  }
  VocabEntry vocab[VOCAB_SIZE + MAX_MERGES];
  memcpy(vocab, tokenizer->base.vocab, VOCAB_SIZE * sizeof(VocabEntry));

  for (int i = 0; i < n_merges; i++) {
    // --- Parallel frequency counting using train_worker ---
    int num_threads = MAX_THREADS;
    pthread_t threads[MAX_THREADS];
    TrainThreadArgs targs[MAX_THREADS];
    PairStat* partial_stats[MAX_THREADS];
    int local_counts[MAX_THREADS] = {0};
    int chunk_size = text_len / num_threads;
    for (int t = 0; t < num_threads; t++) {
      targs[t].ids = ids;
      targs[t].start = t * chunk_size;
      targs[t].end = (t == num_threads - 1) ? text_len : (t + 1) * chunk_size;
      partial_stats[t] = (PairStat*) malloc(INITIAL_CACHE_SIZE * sizeof(PairStat));
      if (!partial_stats[t]) {
        fprintf(stderr, "Error: Memory allocation for partial_stats failed for thread %d.\n", t);
        exit(EXIT_FAILURE);
      }
      targs[t].local_stats = partial_stats[t];
      targs[t].local_count = &local_counts[t];
      *targs[t].local_count = 0;
      if (pthread_create(&threads[t], NULL, train_worker, &targs[t]) != 0) {
        fprintf(stderr, "Error: Failed to create training thread %d.\n", t);
        exit(EXIT_FAILURE);
      }
    }
    for (int t = 0; t < num_threads; t++) {
      pthread_join(threads[t], NULL);
    }
    PairStat global_stats[INITIAL_CACHE_SIZE];
    int global_count = 0;
    merge_train_stats(global_stats, &global_count, partial_stats, local_counts, num_threads);
    for (int t = 0; t < num_threads; t++) {
      free(partial_stats[t]);
    }
    // --- Select best pair from merged stats ---
    int max_occurrences = 0;
    Pair max_pair = {0, 0};
    for (int j = 0; j < global_count; j++) {
      if (global_stats[j].freq > max_occurrences) {
        max_occurrences = global_stats[j].freq;
        max_pair.idx1 = global_stats[j].idx1;
        max_pair.idx2 = global_stats[j].idx2;
      }
    }
    if (max_occurrences == 0) {
      printf("Stopping early at merge %d: No more pairs to merge.\n", i + 1);
      break;
    }
    int new_idx = VOCAB_SIZE + i;
    // --- Use hashmap-based caching for training merge ---
    char merge_key[128];
    char ids_hash[32];
    hash_ids(ids, text_len, ids_hash, sizeof(ids_hash));
    snprintf(merge_key, sizeof(merge_key), "M:%s:%d:%d:%d", ids_hash, max_pair.idx1, max_pair.idx2, new_idx);
    size_t cached_size = 0;
    int* cached_ids = (int*) lru_cache_get(g_train_cache, merge_key, &cached_size);
    if (cached_ids) {
      free(ids);
      ids = cached_ids;
      text_len = cached_size / sizeof(int);
    } else {
      ids = merge(ids, text_len, max_pair, new_idx, &text_len);
      lru_cache_put(g_train_cache, merge_key, ids, text_len * sizeof(int));
    }
    tokenizer->base.merges[i].pair = max_pair;
    tokenizer->base.merges[i].idx = new_idx;
    size_t len1 = strlen(vocab[max_pair.idx1].value);
    size_t len2 = strlen(vocab[max_pair.idx2].value);
    vocab[new_idx].value = (char*) malloc(len1 + len2 + 1);
    if (!vocab[new_idx].value) {
      fprintf(stderr, "Error: Memory allocation for vocab[%d].value failed.\n", new_idx);
      exit(EXIT_FAILURE);
    }
    snprintf(vocab[new_idx].value, len1 + len2 + 1, "%s%s", vocab[max_pair.idx1].value, vocab[max_pair.idx2].value);
    vocab[new_idx].idx = new_idx;
    printf("\tMerge %d/%d: (%d, %d) -> %d (%s) had %d occurrences\n", i + 1, n_merges, max_pair.idx1, max_pair.idx2, new_idx, vocab[new_idx].value, max_occurrences);
    fflush(stdout);
  }
  tokenizer->base.merge_count = n_merges;
  memcpy(tokenizer->base.vocab, vocab, (VOCAB_SIZE + n_merges) * sizeof(VocabEntry));
  free(text_bytes);
  free(ids);
  consistency_check(tokenizer, n_merges);
}

void train_with_incremental_cache(Shred* tokenizer, const char* text, int vocab_size) {
  assert(vocab_size >= VOCAB_SIZE);
  int n_merges = vocab_size - VOCAB_SIZE;
  size_t text_len = strlen(text);
  
  unsigned char* text_bytes = (unsigned char*) malloc((text_len + 1) * sizeof(unsigned char));
  if (!text_bytes) {
    fprintf(stderr, "Error: Memory allocation for text_bytes failed.\n");
    exit(EXIT_FAILURE);
  }
  memcpy(text_bytes, text, text_len);
  text_bytes[text_len] = '\0';
  
  int* ids = (int*) malloc(text_len * sizeof(int));
  if (!ids) {
    fprintf(stderr, "Error: Memory allocation for ids failed.\n");
    free(text_bytes);
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < text_len; i++) {
    ids[i] = text_bytes[i];
  }
  
  VocabEntry vocab[VOCAB_SIZE + MAX_MERGES];
  memcpy(vocab, tokenizer->base.vocab, VOCAB_SIZE * sizeof(VocabEntry));
  
  // Build initial frequency cache for all adjacent pairs.
  for (int i = 0; i < text_len - 1; i++) {
    char key[64];
    snprintf(key, sizeof(key), "P:%d,%d", ids[i], ids[i + 1]);
    size_t size;
    int* freq = (int*) lru_cache_get(g_train_cache, key, &size);
    int new_freq = (freq ? *freq : 0) + 1;
    if (freq) free(freq);
    lru_cache_put(g_train_cache, key, &new_freq, sizeof(int));
  }
  
  // Incremental merging loop.
  for (int i = 0; i < n_merges; i++) {
    // Find the most frequent pair in the training cache.
    int max_occurrences = 0;
    Pair max_pair = {0, 0};
    for (int j = 0; j < g_train_cache->capacity; j++) {
      if (g_train_cache->entries[j].key != NULL) {
        int freq = *((int*) g_train_cache->entries[j].value);
        if (freq > max_occurrences) {
          max_occurrences = freq;
          int a, b;
          sscanf(g_train_cache->entries[j].key, "P:%d,%d", &a, &b);
          max_pair.idx1 = a;
          max_pair.idx2 = b;
        }
      }
    }
    if (max_occurrences == 0) {
      printf("Stopping early at merge %d: No more pairs to merge.\n", i + 1);
      break;
    }
    
    int new_idx = VOCAB_SIZE + i;
    int* merge_positions;
    int num_positions;
    // Perform merge and obtain positions.
    int* new_ids = merge_with_positions(ids, text_len, max_pair, new_idx, &text_len, &merge_positions, &num_positions);
    free(ids);
    ids = new_ids;
    
    // For each merge position, update the cache for neighboring pairs.
    for (int k = 0; k < num_positions; k++) {
      update_frequency_cache_for_merge(ids, text_len, merge_positions[k], new_idx);
    }
    free(merge_positions);
    // Clear the merged pair from the cache.
    clear_merged_pair_in_cache(max_pair);
    
    // Update tokenizer merges and build new vocabulary entry.
    tokenizer->base.merges[i].pair = max_pair;
    tokenizer->base.merges[i].idx = new_idx;
    size_t len1 = strlen(vocab[max_pair.idx1].value);
    size_t len2 = strlen(vocab[max_pair.idx2].value);
    vocab[new_idx].value = (char*) malloc(len1 + len2 + 1);
    if (!vocab[new_idx].value) {
      fprintf(stderr, "Error: Memory allocation for vocab[%d].value failed.\n", new_idx);
      exit(EXIT_FAILURE);
    }
    snprintf(vocab[new_idx].value, len1 + len2 + 1, "%s%s", vocab[max_pair.idx1].value, vocab[max_pair.idx2].value);
    vocab[new_idx].idx = new_idx;
    
    printf("\tMerge %d/%d: (%d, %d) -> %d (%s) had %d occurrences\n",
           i + 1, n_merges, max_pair.idx1, max_pair.idx2, new_idx, vocab[new_idx].value, max_occurrences);
    fflush(stdout);
  }
  
  tokenizer->base.merge_count = n_merges;
  memcpy(tokenizer->base.vocab, vocab, (VOCAB_SIZE + n_merges) * sizeof(VocabEntry));
  
  free(text_bytes);
  free(ids);
  consistency_check(tokenizer, n_merges);
}

void train(Shred* tokenizer, const char* text, int vocab_size) {
  assert(vocab_size >= VOCAB_SIZE);
  int n_merges = vocab_size - VOCAB_SIZE;
  size_t text_len = strlen(text);
  unsigned char* text_bytes = (unsigned char*)malloc((text_len + 1) * sizeof(unsigned char));
  if (!text_bytes) {
    fprintf(stderr, "Error: Memory allocation for text_bytes failed.\n");
    exit(EXIT_FAILURE);
  }
  memcpy(text_bytes, text, text_len);
  text_bytes[text_len] = '\0';

  int* ids = (int*)malloc(text_len * sizeof(int));
  if (!ids) {
    fprintf(stderr, "Error: Memory allocation for ids failed.\n");
    free(text_bytes);
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < text_len; i++) {
    ids[i] = text_bytes[i];
  }
  VocabEntry vocab[VOCAB_SIZE + MAX_MERGES];
  memcpy(vocab, tokenizer->base.vocab, VOCAB_SIZE * sizeof(VocabEntry));

  for (int i = 0; i < n_merges; i++) {
    // parallel frequency counting using train_worker
    int num_threads = MAX_THREADS;
    pthread_t threads[MAX_THREADS];
    TrainThreadArgs targs[MAX_THREADS];
    PairStat* partial_stats[MAX_THREADS];
    int local_counts[MAX_THREADS] = {0};
    int chunk_size = text_len / num_threads;
    for (int t = 0; t < num_threads; t++) {
      targs[t].ids = ids;
      targs[t].start = t * chunk_size;
      targs[t].end = (t == num_threads - 1) ? text_len : (t + 1) * chunk_size;
      partial_stats[t] = (PairStat*)malloc(INITIAL_CACHE_SIZE * sizeof(PairStat));
      if (!partial_stats[t]) {
        fprintf(stderr, "Error: Memory allocation for partial_stats failed for thread %d.\n", t);
        exit(EXIT_FAILURE);
      }
      targs[t].local_stats = partial_stats[t];
      targs[t].local_count = &local_counts[t];
      *targs[t].local_count = 0;
      if (pthread_create(&threads[t], NULL, train_worker, &targs[t]) != 0) {
        fprintf(stderr, "Error: Failed to create training thread %d.\n", t);
        exit(EXIT_FAILURE);
      }
    }
    // waiting for all training threads to complete
    for (int t = 0; t < num_threads; t++) {
      pthread_join(threads[t], NULL);
    }
    // merging partial frequency counts from each thread into a global stats table
    PairStat global_stats[INITIAL_CACHE_SIZE];
    int global_count = 0;
    merge_train_stats(global_stats, &global_count, partial_stats, local_counts, num_threads);
    // free per-thread partial stats
    for (int t = 0; t < num_threads; t++) {
      free(partial_stats[t]);
    }
    // selected the pair with the maximum frequency from the merged stats
    int max_occurrences = 0;
    Pair max_pair = {0, 0};
    for (int j = 0; j < global_count; j++) {
      if (global_stats[j].freq > max_occurrences) {
        max_occurrences = global_stats[j].freq;
        max_pair.idx1 = global_stats[j].idx1;
        max_pair.idx2 = global_stats[j].idx2;
      }
    }
    if (max_occurrences == 0) {
      printf("Stopping early at merge %d: No more pairs to merge.\n", i + 1);
      break;
    }
    int new_idx = VOCAB_SIZE + i;
    ids = merge(ids, text_len, max_pair, new_idx, &text_len);

    // update the merges in the tokenizer state
    tokenizer->base.merges[i].pair = max_pair;
    tokenizer->base.merges[i].idx = new_idx;

    // creating the new vocabulary entry by concatenating the merged tokens
    size_t len1 = strlen(vocab[max_pair.idx1].value);
    size_t len2 = strlen(vocab[max_pair.idx2].value);
    vocab[new_idx].value = (char*)malloc(len1 + len2 + 1);
    if (!vocab[new_idx].value) {
      fprintf(stderr, "Error: Memory allocation for vocab[%d].value failed.\n", new_idx);
      exit(EXIT_FAILURE);
    }
    snprintf(vocab[new_idx].value, len1 + len2 + 1, "%s%s", vocab[max_pair.idx1].value, vocab[max_pair.idx2].value);
    vocab[new_idx].idx = new_idx;

    // verbose logging
    printf("\tMerge %d/%d: (%d, %d) -> %d (%s) had %d occurrences\n", i + 1, n_merges, max_pair.idx1, max_pair.idx2, new_idx, vocab[new_idx].value, max_occurrences);
    fflush(stdout);
  }
  // final updates: update merge count and vocab in tokenizer
  tokenizer->base.merge_count = n_merges;
  memcpy(tokenizer->base.vocab, vocab, (VOCAB_SIZE + n_merges) * sizeof(VocabEntry));
  free(text_bytes);
  free(ids);

  consistency_check(tokenizer, n_merges);
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

// function to export the regex pattern, if any, to the python code interface
char* export_pattern(const Shred* tokenizer) {
  if (!tokenizer) {
    fprintf(stderr, "Error: Tokenizer pointer is null.\n");
    return strdup(""); // return empty string to indicate no pattern
  }
  if (strlen(tokenizer->base.pattern) == 0) {
    return strdup(""); // return empty string if no pattern exists
  }
  return strdup(tokenizer->base.pattern); // return a copy of the pattern
}

// function to export the ``special_tokens`` if any, to the python code interface
char* export_special_tokens(const Shred* tokenizer) {
  if (!tokenizer) {
    fprintf(stderr, "Error: Tokenizer pointer is null.\n");
    return strdup(""); // return empty string to indicate no special tokens.
  }

  if (tokenizer->base.special_token_count == 0) {
    return strdup(""); // return empty string if no special tokens exist
  }
  const int buffer_size = MAX_SPECIAL_TOKENS * MAX_LINE_LENGTH;
  char* output = (char*)malloc(buffer_size);
  if (!output) {
    fprintf(stderr, "Error: Memory allocation failed for output.\n");
    return strdup(""); // return empty string in case of allocation failure
  }

  output[0] = '\0'; // initialize output as an empty string.
  int offset = 0;
  for (int i = 0; i < tokenizer->base.special_token_count; i++) {
    offset += snprintf(output + offset, buffer_size - offset, "%s %d\n", tokenizer->base.special_tokens[i], tokenizer->base.special_token_indices[i]);
    if (offset >= buffer_size) {
      fprintf(stderr, "Error: Output buffer overflow.\n");
      free(output);
      return strdup(""); // return empty string in case of overflow.
    }
  }
  return output; // return serialized special tokens.
}

void set_pattern(Shred* tokenizer, const char* new_pattern) {
  if (!tokenizer || !new_pattern) {
    fprintf(stderr, "Error: Invalid arguments passed to set_pattern.\n");
    return;
  }
  strncpy(tokenizer->base.pattern, new_pattern, MAX_LINE_LENGTH - 1);
  tokenizer->base.pattern[MAX_LINE_LENGTH - 1] = '\0'; // ensure null termination
  printf("Pattern updated successfully.\n");
}

void set_special_tokens(Shred* tokenizer, const char* token_data) {
  if (!tokenizer || !token_data) {
    fprintf(stderr, "Error: Invalid arguments passed to set_special_tokens.\n");
    return;
  }
  
  tokenizer->base.special_token_count = 0; // reset existing tokens

  const char* line = token_data;
  while (*line) {
    char token[MAX_LINE_LENGTH];
    int index;
    int items_read = sscanf(line, "%s %d", token, &index);
    if (items_read == 2 && tokenizer->base.special_token_count < MAX_SPECIAL_TOKENS) {
      strncpy(tokenizer->base.special_tokens[tokenizer->base.special_token_count], token, MAX_LINE_LENGTH - 1);
      tokenizer->base.special_tokens[tokenizer->base.special_token_count][MAX_LINE_LENGTH - 1] = '\0';
      tokenizer->base.special_token_indices[tokenizer->base.special_token_count] = index;
      tokenizer->base.special_token_count++;
    } else if (items_read != 2) {
      fprintf(stderr, "Error: Invalid token data format or too many tokens.\n");
      break;
    }
    // move to the next line
    line = strchr(line, '\n');
    if (line) line++; // skip the newline character
    else break;
  }
  printf("Special tokens updated successfully.\n");
}

void free_string(char* string) {
  free(string);
}