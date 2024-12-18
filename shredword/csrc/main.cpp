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
  tqdm bar;
  init_tqdm(&bar, "Training BPE tokenizer: ", false, "merges", true, n_merges, 1);

  for (int i = 0; i < n_merges; i++) {
    update_tqdm(&bar, 1, i == n_merges - 1);
    fflush(stdout);
    int stats[MAX_MERGES][3];
    get_stats(ids, text_len, stats);
    int max_occurrences = 0, max_ids = -1;
    Pair max_pair;
    for (int j = 0; j < MAX_MERGES && stats[j][2] > 0; j++) {
      if (stats[j][2] > max_occurrences) {
        max_occurrences = stats[j][2];
        max_pair.idx1 = stats[j][0];
        max_pair.idx2 = stats[j][1];
        max_ids = j;
      }
    }
    if (max_ids == -1 || max_occurrences == 0) {
      break;
    }
    int new_idx = VOCAB_SIZE + i;
    ids = merge(ids, text_len, max_pair, new_idx, &text_len);
    merges[i] = max_pair;
    size_t len1 = strlen(vocab[max_pair.idx1].value);
    size_t len2 = strlen(vocab[max_pair.idx2].value);
    vocab[new_idx].value = (char*)malloc(len1 + len2 + 1);
    if (!vocab[new_idx].value) {
      fprintf(stderr, "Error: Memory allocation for vocab[%d].value failed.\n", new_idx);
      exit(EXIT_FAILURE);
    }
    strcpy(vocab[new_idx].value, vocab[max_pair.idx1].value);
    strcat(vocab[new_idx].value, vocab[max_pair.idx2].value);
    vocab[new_idx].idx = new_idx;
    if (verbose) {
      printf("Merge %d/%d: (%d, %d) -> %d (%s) had %d occurrences\n", i + 1, n_merges, max_pair.idx1, max_pair.idx2, new_idx, vocab[new_idx].value, max_occurrences);
    }
  }
  close_tqdm(&bar);
  tokenizer->base.merge_count = n_merges;
  memcpy(tokenizer->base.merges, merges, n_merges * sizeof(Pair));
  memcpy(tokenizer->base.vocab, vocab, (VOCAB_SIZE + n_merges) * sizeof(VocabEntry));
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
  for (int i = 0; i < ids_size; i++) {
    strcat(output, tokenizer->base.vocab[ids[i]].value);
  }
  return output;
}

int* encode(Shred* tokenizer, const char* text, int* output_size) {
  size_t text_len = strlen(text);
  // initialize ids with input character values
  int* ids = (int*)malloc(text_len * sizeof(int));
  if (!ids) {
    fprintf(stderr, "Error: Memory allocation for ids failed.\n");
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < text_len; i++) {
    ids[i] = (unsigned char)text[i];
  }
  size_t ids_len = text_len;
  // iteratively apply merge rules
  for (int i = 0; i < tokenizer->base.merge_count; i++) {
    MergeEntry merge = tokenizer->base.merges[i];   // access MergeEntry
    int idx1 = (merge.idx >> 16) & 0xFFFF;          // extract first part
    int idx2 = merge.idx & 0xFFFF;                  // extract second part

    size_t new_ids_len = 0;
    int* new_ids = (int*)malloc(ids_len * sizeof(int));
    if (!new_ids) {
      fprintf(stderr, "Error: Memory allocation for new_ids failed.\n");
      free(ids);
      exit(EXIT_FAILURE);
    }

    for (size_t j = 0; j < ids_len; j++) {
      if (j < ids_len - 1 && ids[j] == idx1 && ids[j + 1] == idx2) {
        // applying merge
        new_ids[new_ids_len++] = VOCAB_SIZE + i;
        j++; // skip the next id as it's part of the merge
      } else {
        new_ids[new_ids_len++] = ids[j];
      }
    }

    free(ids);
    ids = new_ids;
    ids_len = new_ids_len;
  }
  *output_size = ids_len;
  return ids;
}

void save_model(const Shred* tokenizer, const char* file_path) {
  save_tokenizer(&(tokenizer->base), file_path);
}

void load_model(Shred* tokenizer, const char* model_file) {
  load_tokenizer(&(tokenizer->base), model_file);
}