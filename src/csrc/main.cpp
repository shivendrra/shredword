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

/* 
  New optimized training function:
   - Uses multi-threaded frequency counting (as before)
   - Then builds a priority queue (binary heap) for fast merge selection
   - Filters out low-frequency pairs (frequency < MIN_PAIR_FREQUENCY)
   - Uses the original merge() function for compatibility
*/
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
    int stats[MAX_MERGES][3];
    memset(stats, 0, sizeof(stats));
    get_stats(ids, text_len, stats);

    int max_occurrences = 0, max_ids = -1;
    Pair max_pair = {0, 0};
    for (int j = 0; j < MAX_MERGES && stats[j][2] > 0; j++) {
      if (stats[j][2] > max_occurrences) {
        max_occurrences = stats[j][2];
        max_pair.idx1 = stats[j][0];
        max_pair.idx2 = stats[j][1];
        max_ids = j;
      }
    }
    if (max_ids == -1 || max_occurrences == 0) {
      printf("Stopping early at merge %d: No more pairs to merge.\n", i + 1);
      break;
    }

    int new_idx = VOCAB_SIZE + i;
    ids = merge(ids, text_len, max_pair, new_idx, &text_len);

    // directly updating tokenizer->base.merges unlike the previous implementation where i used a buffer variable
    // & fucked up the whole loigc & took 3 weeks to fix it (peak skill issue)
    tokenizer->base.merges[i].pair = max_pair;
    tokenizer->base.merges[i].idx = new_idx;

    size_t len1 = strlen(vocab[max_pair.idx1].value);
    size_t len2 = strlen(vocab[max_pair.idx2].value);
    vocab[new_idx].value = (char*)malloc(len1 + len2 + 1);
    if (!vocab[new_idx].value) {
      fprintf(stderr, "Error: Memory allocation for vocab[%d].value failed.\n", new_idx);
      exit(EXIT_FAILURE);
    }
    snprintf(vocab[new_idx].value, len1 + len2 + 1, "%s%s", vocab[max_pair.idx1].value, vocab[max_pair.idx2].value);
    vocab[new_idx].idx = new_idx;

    // verbose logging is not optional
    printf("\tMerge %d/%d: (%d, %d) -> %d (%s) had %d occurrences\n", i + 1, n_merges, max_pair.idx1, max_pair.idx2, new_idx, vocab[new_idx].value, max_occurrences);
    fflush(stdout);
  }
  // removed tqdm, it was slowing the process down
  // final updates
  tokenizer->base.merge_count = n_merges;
  memcpy(tokenizer->base.vocab, vocab, (VOCAB_SIZE + n_merges) * sizeof(VocabEntry));
  free(text_bytes);
  free(ids);
}

void dynamic_train_bpe(Shred* tokenizer, const char* text, int vocab_size, int min_freq) {
  // Ensure that vocab_size is large enough.
  assert(vocab_size >= VOCAB_SIZE);
  int n_merges = vocab_size - VOCAB_SIZE;
  int merge_count = 0;
  size_t text_len = strlen(text);

  // Create initial ids array from the text.
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
  free(text_bytes);

  // Copy initial vocabulary from the tokenizer.
  VocabEntry vocab[VOCAB_SIZE + MAX_MERGES];
  memcpy(vocab, tokenizer->base.vocab, VOCAB_SIZE * sizeof(VocabEntry));

  // Main merging loop.
  while (merge_count < n_merges) {
    // --- Step 1: Compute frequency statistics ---
    int num_threads = MAX_THREADS;
    pthread_t threads[MAX_THREADS];
    TrainThreadArgs targs[MAX_THREADS];
    PairStat* partial_stats_arr[MAX_THREADS];
    int local_counts[MAX_THREADS] = {0};
    int current_ids_len = text_len;
    int chunk_size = current_ids_len / num_threads;
    for (int t = 0; t < num_threads; t++) {
      targs[t].ids = ids;
      targs[t].start = t * chunk_size;
      targs[t].end = (t == num_threads - 1) ? current_ids_len : (t + 1) * chunk_size;
      partial_stats_arr[t] = (PairStat*)malloc(INITIAL_CACHE_SIZE * sizeof(PairStat));
      if (!partial_stats_arr[t]) {
        fprintf(stderr, "Error: Memory allocation for partial_stats failed for thread %d.\n", t);
        exit(EXIT_FAILURE);
      }
      targs[t].local_stats = partial_stats_arr[t];
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
    merge_train_stats(global_stats, &global_count, partial_stats_arr, local_counts, num_threads);
    for (int t = 0; t < num_threads; t++) {
      free(partial_stats_arr[t]);
    }

    // --- Step 2: Build priority queue from stats ---
    PriorityQueue* pq = pq_create(1024);
    for (int j = 0; j < global_count; j++) {
      // Only consider pairs with frequency >= min_freq (if specified)
      if (global_stats[j].freq >= (min_freq > 0 ? min_freq : 1)) {
        TokenPair tp;
        tp.idx1 = global_stats[j].idx1;
        tp.idx2 = global_stats[j].idx2;
        tp.frequency = global_stats[j].freq;
        pq_push(pq, tp);
      }
    }
    if (pq_empty(pq)) {
      pq_free(pq);
      break;
    }

    // --- Step 3: Extract a batch of top pairs (up to 15) ---
    int batch_size = ((n_merges - merge_count) < 15) ? (n_merges - merge_count) : 15;
    TokenPair batch_merges[15];
    int actual_batch = 0;
    for (int i = 0; i < batch_size && !pq_empty(pq); i++) {
      batch_merges[i] = pq_pop(pq);
      actual_batch++;
    }
    pq_free(pq);

    // --- Step 4: Apply each merge in the batch ---
    for (int i = 0; i < actual_batch; i++) {
      TokenPair best = batch_merges[i];
      int new_idx = VOCAB_SIZE + merge_count;  // New token index
      int* old_ids = ids;
      ids = merge(ids, text_len, (Pair){best.idx1, best.idx2}, new_idx, &text_len);
      free(old_ids);
      
      // Update vocabulary: create a new token by concatenating the two tokens.
      size_t len1 = strlen(vocab[best.idx1].value);
      size_t len2 = strlen(vocab[best.idx2].value);
      vocab[new_idx].value = (char*)malloc(len1 + len2 + 1);
      if (!vocab[new_idx].value) {
        fprintf(stderr, "Error: Memory allocation for vocab[%d].value failed.\n", new_idx);
        exit(EXIT_FAILURE);
      }
      snprintf(vocab[new_idx].value, len1 + len2 + 1, "%s%s", vocab[best.idx1].value, vocab[best.idx2].value);
      vocab[new_idx].idx = new_idx;
      
      tokenizer->base.merges[merge_count].pair.idx1 = best.idx1;
      tokenizer->base.merges[merge_count].pair.idx2 = best.idx2;
      tokenizer->base.merges[merge_count].idx = new_idx;
      
      printf("Merge %d: (%d, %d) -> %d [%s] had %d occurrences\n",
             merge_count + 1, best.idx1, best.idx2, new_idx, vocab[new_idx].value, best.frequency);
      merge_count++;
      if (merge_count >= n_merges)
        break;
    }
    // Continue loop with updated ids (and updated text_len).
  }
  
  // --- Finalize ---
  tokenizer->base.merge_count = merge_count;
  memcpy(tokenizer->base.vocab, vocab, (VOCAB_SIZE + merge_count) * sizeof(VocabEntry));
  free(ids);
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