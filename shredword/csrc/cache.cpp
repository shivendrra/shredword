#include "cache.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>

void initialize_token_cache(const Shred* tokenizer) {
  for (int i = 0; i < VOCAB_SIZE + MAX_MERGES; i++) {
    if (tokenizer->base.vocab[i].value) {
      token_length_cache[i] = strlen(tokenizer->base.vocab[i].value);
    } else {
      token_length_cache[i] = 0;
    }
  }
}

// worker function for decoding in parallel
void* decode_worker(void* args) {
  ThreadArgs* thread_args = (ThreadArgs*)args;
  const Shred* tokenizer = thread_args->tokenizer;
  const int* ids = thread_args->ids;
  int start = thread_args->start;
  int end = thread_args->end;

  size_t local_size = 0;

  // calculating local size
  for (int i = start; i < end; i++) {
    local_size += token_length_cache[ids[i]];
  }

  char* local_output = (char*)malloc(local_size + 1);
  if (!local_output) {
    fprintf(stderr, "Error: Memory allocation failed for local output.\n");
    exit(EXIT_FAILURE);
  }

  char* current_pos = local_output;
  for (int i = start; i < end; i++) {
    const char* token = tokenizer->base.vocab[ids[i]].value;
    size_t token_len = token_length_cache[ids[i]];
    memcpy(current_pos, token, token_len);
    current_pos += token_len;
  }
  *current_pos = '\0';

  thread_args->output_str = local_output;
  thread_args->output_int = NULL;
  *thread_args->output_size = local_size;

  return NULL;
}

// worker function for encoding in parallel
void* encode_worker(void* args) {
  ThreadArgs* thread_args = (ThreadArgs*)args;
  const Shred* tokenizer = thread_args->tokenizer;
  const int* ids = thread_args->ids;
  int start = thread_args->start;
  int end = thread_args->end;

  size_t local_size = end - start;
  int* local_output = (int*)malloc(local_size * sizeof(int));
  if (!local_output) {
    fprintf(stderr, "Error: Memory allocation for local_output failed.\n");
    exit(EXIT_FAILURE);
  }

  size_t new_output_size = 0;
  for (size_t i = start; i < end; i++) {
    int current_id = ids[i];
    if (i < end - 1) {
      for (int j = 0; j < tokenizer->base.merge_count; j++) {
        Pair max_pair = tokenizer->base.merges[j].pair;
        if (current_id == max_pair.idx1 && ids[i + 1] == max_pair.idx2) {
          local_output[new_output_size++] = VOCAB_SIZE + j;
          i++; // skip the next ID as it is part of the merge
          current_id = -1; // mark as merged
          break;
        }
      }
    }
    if (current_id != -1) {
      local_output[new_output_size++] = current_id;
    }
  }

  thread_args->output_int = local_output;
  thread_args->output_str = NULL;
  *thread_args->output_size = new_output_size;

  return NULL;
}