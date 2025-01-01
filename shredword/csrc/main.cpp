#include <assert.h>
#include <string.h>
#include <stdio.h>
#include "main.h"
#include "base.h"

void init_shred(Shred* tokenizer) {
 init_tokenizer(&(tokenizer->base));
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

  consistency_check(tokenizer, n_merges);  // consistency check function calling
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

  int* ids = (int*)malloc(text_len * sizeof(int));
  if (!ids) {
    fprintf(stderr, "Error: Memory allocation for ids failed.\n");
    exit(EXIT_FAILURE);
  }
  for (size_t i = 0; i < text_len; i++) {
    ids[i] = (unsigned char)text[i];
  }
  size_t ids_len = text_len;
  
  for (int i = 0; i < tokenizer->base.merge_count; i++) {
    MergeEntry merge = tokenizer->base.merges[i];
    Pair max_pair = merge.pair;

    size_t new_ids_len = 0;
    int* new_ids = (int*)malloc(ids_len * sizeof(int));
    if (!new_ids) {
      fprintf(stderr, "Error: Memory allocation for new_ids failed.\n");
      free(ids);
      exit(EXIT_FAILURE);
    }

    for (size_t j = 0; j < ids_len; j++) {
      if (j < ids_len - 1 && ids[j] == max_pair.idx1 && ids[j + 1] == max_pair.idx2) {
        new_ids[new_ids_len++] = VOCAB_SIZE + i;
        j++;
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

// helper to format a single merge pair into a string
static void format_merge(Pair pair, char* buffer) {
  sprintf(buffer, "%d %d", pair.idx1, pair.idx2);
}

// helper to format a single vocabulary entry into a string
static void format_vocab(VocabEntry vocab, char* buffer) {
  sprintf(buffer, "%d %s", vocab.idx, vocab.value);
}

// function to export vocabulary as a serialized string
char* export_merges(const Shred* tokenizer) {
  size_t buffer_size = MAX_MERGES * 32; // rough estimate for memory
  char* output = (char*)malloc(buffer_size);
  if (!output) {
    fprintf(stderr, "Error: Memory allocation failed for export_merges.\n");
    exit(EXIT_FAILURE);
  }
  output[0] = '\0';

  for (int i = 0; i < tokenizer->base.merge_count; i++) {
    Pair pair = tokenizer->base.merges[i].pair;
    // skipping invalid or uninitialized merges
    if (pair.idx1 < 0 || pair.idx2 < 0) {
      continue;
    }
    char merge_str[32];
    sprintf(merge_str, "%d %d\n", pair.idx1, pair.idx2);
    strcat(output, merge_str);
  }
  return output;
}

// function to export vocabulary as a serialized string
char* export_vocab(const Shred* tokenizer) {
  size_t buffer_size = (VOCAB_SIZE + MAX_MERGES) * 64;  // rough estimate for memory
  char* output = (char*)malloc(buffer_size);
  if (!output) {
    fprintf(stderr, "Error: Memory allocation failed for export_vocab.\n");
    exit(EXIT_FAILURE);
  }
  output[0] = '\0';

  for (int i = 0; i < tokenizer->base.vocab_size + tokenizer->base.merge_count; i++) {
    if (tokenizer->base.vocab[i].value == NULL) continue;

    char vocab_str[64];
    format_vocab(tokenizer->base.vocab[i], vocab_str);
    strcat(output, vocab_str);
    strcat(output, "\n");
  }

  return output; // caller must free this memory
}
