#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <ctype.h>
#include "base.h"

void init_tokenizer(BaseTokenizer* tokenizer) {
  tokenizer->merge_count = 0;
  tokenizer->vocab_size = VOCAB_SIZE;
  tokenizer->special_token_count = 0;
  for (int i = 0; i < VOCAB_SIZE; i++) {
    tokenizer->vocab[i].idx = i;
    tokenizer->vocab[i].value = (char*)malloc(2);
    tokenizer->vocab[i].value[0] = (char)i;
    tokenizer->vocab[i].value[1] = '\0';
  }
}

void build_vocab(BaseTokenizer* tokenizer) {
  for (int i = 0; i < tokenizer->merge_count; i++) {
    int idx = VOCAB_SIZE + i;
    Pair pair = tokenizer->merges[i].pair;

    size_t len1 = strlen(tokenizer->vocab[pair.idx1].value);
    size_t len2 = strlen(tokenizer->vocab[pair.idx2].value);

    tokenizer->vocab[idx].value = (char*)malloc(len1 + len2 + 1);
    strcpy(tokenizer->vocab[pair.idx1].value, tokenizer->vocab[pair.idx1].value);
    strcpy(tokenizer->vocab[pair.idx2].value, tokenizer->vocab[pair.idx2].value);
    tokenizer->vocab[idx].idx = idx;
  }
  for (int i = 0; i < tokenizer->special_token_count; i++) {
    int idx = VOCAB_SIZE + tokenizer->merge_count + i;
    tokenizer->vocab[idx].value = strdup(tokenizer->special_tokens[i]);
    tokenizer->vocab[idx].idx = idx;
  }
}

void get_stats(const int* ids, int ids_size, int stats[MAX_MERGES][3]) {
  for (int i = 0; i < MAX_MERGES; i++) {
    stats[i][0] = -1; // first token in the pair
    stats[i][1] = -1; // second token in the pair
    stats[i][2] = 0;  // frequency
  }

  for (int i = 0; i < ids_size - 1; i++) {
    int idx1 = ids[i];
    int idx2 = ids[i + 1];
    int found = 0;
    // checking if the pair already exists in stats
    for (int j = 0; j < MAX_MERGES; j++) {
      if (stats[j][0] == idx1 && stats[j][1] == idx2) {
        stats[j][2]++;
        found = 1;
        break;
      }
    }
    // if not found, add the new pair
    if (!found) {
      for (int j = 0; j < MAX_MERGES; j++) {
        if (stats[j][0] == -1) { // empty slot
          stats[j][0] = idx1;
          stats[j][1] = idx2;
          stats[j][2] = 1;
          break;
        }
      }
    }
  }
}

int* merge(const int* ids, int ids_size, Pair pair, int idx, size_t* new_size) {
  int* new_ids = (int*)malloc(ids_size * sizeof(int));
  int new_idx = 0;

  for (int i = 0; i < ids_size; i++) {
    if (i < ids_size - 1 && ids[i] == pair.idx1 && ids[i + 1] == pair.idx2) {
      new_ids[new_idx++] = idx;  // merging the pair into a single token
      i++;  // skipping the second token in the pair
    } else {
      new_ids[new_idx++] = ids[i];  // copy the token as is
    }
  }
  *new_size = new_idx;  // updateing the size of the new array
  new_ids = (int*)realloc(new_ids, new_idx * sizeof(int));  // resizing the array to fit the new size
  return new_ids;
}


void save_tokenizer(const BaseTokenizer* tokenizer, const char* file_prefix) {
  char model_file[MAX_LINE_LENGTH];
  snprintf(model_file, MAX_LINE_LENGTH, "%s.model", file_prefix);

  FILE* model_fp = fopen(model_file, "w");
  fprintf(model_fp, "BaseTokenizerword v1\n%s\n%d\n", tokenizer->pattern, tokenizer->special_token_count);
  for (int i = 0; i < tokenizer->special_token_count; i++) {
    fprintf(model_fp, "%s %d\n", tokenizer->special_tokens[i], tokenizer->special_token_indices[i]);
  }
  for (int i = 0; i < tokenizer->merge_count; i++) {
    fprintf(model_fp, "%d %d\n", tokenizer->merges[i].pair.idx1, tokenizer->merges[i].pair.idx2);
  }
  fclose(model_fp);

  char vocab_file[MAX_LINE_LENGTH];
  snprintf(vocab_file, MAX_LINE_LENGTH, "%s.vocab", file_prefix);

  FILE* vocab_fp = fopen(vocab_file, "w");
  for (int i = 0; i < tokenizer->vocab_size + tokenizer->merge_count + tokenizer->special_token_count; i++) {
    char rendered[MAX_LINE_LENGTH];
    render_token(tokenizer->vocab[i].value, rendered);
    fprintf(vocab_fp, "[%s] %d\n", rendered, tokenizer->vocab[i].idx);
  }
  fclose(vocab_fp);
}

void load_tokenizer(BaseTokenizer* tokenizer, const char* model_file) {
  FILE* fp = fopen(model_file, "r");
  char line[MAX_LINE_LENGTH];
  fgets(line, MAX_LINE_LENGTH, fp); // version
  fgets(tokenizer->pattern, MAX_LINE_LENGTH, fp); // pattern

  int num_special;
  fscanf(fp, "%d\n", &num_special);
  tokenizer->special_token_count = num_special;
  for (int i = 0; i < num_special; i++) {
    fscanf(fp, "%s %d\n", tokenizer->special_tokens[i], &tokenizer->special_token_indices[i]);
  }
  while (fscanf(fp, "%d %d\n", &tokenizer->merges[tokenizer->merge_count].pair.idx1,
                &tokenizer->merges[tokenizer->merge_count].pair.idx2) == 2) {
    tokenizer->merge_count++;
  }
  fclose(fp);
}

void replace_control_characters(const char* input, char* output) {
  size_t out_idx = 0;
  for (size_t i = 0; input[i] != '\0'; i++) {
    if (iscntrl(input[i])) {
      out_idx += sprintf(output + out_idx, "\\u%04x", (unsigned char)input[i]);
    } else {
      output[out_idx++] = input[i];
    }
  }
  output[out_idx] = '\0';
}

void render_token(const char* token, char* output) {
  replace_control_characters(token, output);
}

void free_tokenizer(BaseTokenizer* tokenizer) {
  for (int i = 0; i < tokenizer->vocab_size + tokenizer->merge_count + tokenizer->special_token_count; i++) {
    free(tokenizer->vocab[i].value);
  }
}