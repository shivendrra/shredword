/*
  base.h
  * Base class for basic BPE functions & logics.
  * Contains functions for tokenizer initialization, normalization, and helper routines.
  * To be compiled with base.cpp containing the main logic (no regex, no caching)
*/

#ifndef __BASE__H__
#define __BASE__H__

#define VOCAB_SIZE 256
#define MAX_LINE_LENGTH 2048
#define MAX_SPECIAL_TOKENS 100
#define MAX_MERGES 10000

typedef struct {
  int idx1, idx2;
} Pair;

typedef struct {
  int idx;
  char* value;
} VocabEntry;

typedef struct {
  Pair pair;
  int idx;
} MergeEntry;

typedef struct {
  VocabEntry vocab[VOCAB_SIZE + MAX_MERGES];
  MergeEntry merges[MAX_MERGES];
  int merge_count, vocab_size, special_token_indices[MAX_SPECIAL_TOKENS], special_token_count;
  char special_tokens[MAX_SPECIAL_TOKENS][MAX_LINE_LENGTH], pattern[MAX_LINE_LENGTH];
} BaseTokenizer;

extern "C" {
  void init_tokenizer(BaseTokenizer* tokenizer);
  // normalize input text to NFKC form and replace spaces with "▁"
  char* normalize_text(const char* input);

  void get_stats(const int* ids, int ids_size, int stats[MAX_MERGES][3]);
  int* merge(const int* ids, int ids_size, Pair pair, int idx, size_t* new_size);
  void render_token(const char* token, char* output);
  void replace_control_characters(const char* input, char* output);

  void save_tokenizer(const BaseTokenizer* tokenizer, const char* file_path);
  void load_tokenizer(BaseTokenizer* tokenizer, const char* model_file);

  void free_tokenizer(BaseTokenizer* tokenizer);
}

#endif