#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "main.h"
#include "base.h"
#include "tqdm.h"

void init_shred(Shred* tokenizer) {
 init_tokenizer(&(tokenizer->base));
}

void train(Shred* tokenizer, const char* text, int vocab_size, bool verbose) {
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
  Pair merges[MAX_MERGES];
  VocabEntry vocab[VOCAB_SIZE + MAX_MERGES];
  memcpy(vocab, tokenizer->base.vocab, VOCAB_SIZE * sizeof(VocabEntry));
  
  // tqdm progress bar initialization
  tqdm bar;
  init_tqdm(&bar, "Training BPE tokenizer: ", false, "merges", true, n_merges, 1);

  for (int i = 0; i < n_merges; i++) {
    update_tqdm(&bar, 1, i == n_merges - 1);

    // debug statements ---------
    // printf("Starting merge iteration %d/%d...\n", i + 1, n_merges);
    // fflush(stdout); // force output to appear immediately
    // ----------
    
    int stats[MAX_MERGES][3];
    get_stats(ids, text_len, stats);
    
    int max_occurrences = 0, max_ids = -1;
    Pair max_pair;
    for(int j = 0; j < MAX_MERGES && stats[j][2] > 0; j++) {
      if(stats[j][2] > max_occurrences) {
        max_occurrences = stats[j][2];
        max_pair.idx1 = stats[j][0];
        max_pair.idx2 = stats[j][1];
        max_ids = j;
      }
    }
    if (max_ids == -1) {
      break;    // no more merging pairs left
    }
    int new_idx = VOCAB_SIZE + i;
    ids = merge(ids, text_len, max_pair, new_idx, &text_len);
    merges[i] = max_pair;
    vocab[new_idx].idx = new_idx;

    size_t len1 = strlen(vocab[max_pair.idx1].value);
    size_t len2 = strlen(vocab[max_pair.idx2].value);
    vocab[new_idx].value = (char*)malloc(len1 + len2 + 1);
    strcpy(vocab[new_idx].value, vocab[max_pair.idx1].value);
    strcpy(vocab[new_idx].value, vocab[max_pair.idx2].value);

    if (verbose == true) {
      printf("Merge %d/%d: (%d, %d) -> %d (%s) had %d occurrences\n",
             i + 1, n_merges, max_pair.idx1, max_pair.idx2, new_idx,
             vocab[new_idx].value, max_occurrences);
    }
  }
  close_tqdm(&bar);
  tokenizer->base.merge_count = n_merges;
  memcpy(tokenizer->base.merges, merges, n_merges * sizeof(Pair));
  memcpy(tokenizer->base.vocab, vocab, (n_merges + VOCAB_SIZE) * sizeof(VocabEntry));
  free(text_bytes);
  free(ids);
}

char* decode(Shred* tokenizer, const int* ids, int ids_size) {
  size_t output_size = 0;
  for (int i = 0; i < ids_size; i++) {
    output_size += strlen(tokenizer->base.vocab[ids[i]].value);
  }
  char* output = (char*)malloc(output_size + 1);
  if (!output) {
  fprintf(stderr, "Error: Memory allocation for output failed.\n");
  exit(EXIT_FAILURE);
  }
  output[0] = '\0';
  for(int i = 0; i < ids_size; i++) {
    strcat(output, tokenizer->base.vocab[ids[i]].value);
  }
  return output;
}

int* encode(Shred* tokenizer, const char* text, int* output_size) {
  size_t text_len = strlen(text);
  unsigned char* text_bytes = (unsigned char*)malloc(text_len + 1);
  memcpy(text_bytes, text, text_len);
  text_bytes[text_len] = '\0';

  int* ids = (int*)malloc(text_len * sizeof(int));
  if (!ids) {
  fprintf(stderr, "Error: Memory allocation for ids failed.\n");
  exit(EXIT_FAILURE);
  }
  for (int i = 0; i < text_len; i++) {
    ids[i] = text_bytes[i];
  }

  while (text_len >= 2) {
    int stats[MAX_MERGES][3];
    get_stats(ids, text_len, stats);
    
    int min_merge_idx = -1;
    Pair min_pair;
    int min_value = INT_MAX;
    for (int i = 0; i < MAX_MERGES && stats[i][2] > 0; i++) {
      Pair pair = {stats[i][0], stats[i][1]};
      int merge_idx = tokenizer->base.merges[i].idx;
      if (merge_idx < min_value) {
        min_value = merge_idx;
        min_pair = pair;
        min_merge_idx = merge_idx;
      }
    }
    if(min_merge_idx == -1) {
      break;
    }
    ids = merge(ids, text_len, min_pair, min_merge_idx, &text_len);
  }
  *output_size = text_len;
  free(text_bytes);
  return ids;
}

void save_model(const Shred* tokenizer, const char* file_path) {
  save_tokenizer(&(tokenizer->base), file_path);
}

void load_model(Shred* tokenizer, const char* model_file) {
  load_tokenizer(&(tokenizer->base), model_file);
}