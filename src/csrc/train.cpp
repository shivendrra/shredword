#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include "train.h"
#include "cache.h"
#include "base.h"

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