/*
  main.h
  * main ``ShredWord`` class codes in this file
  * wrapper over the ``base.cpp`` & it's functions, handling the training, encoding & decoding taks
  * to be compiled with ``base.cpp`` containing the main logic (no regex, no caching)
  * compile it as:
    *- '.so': g++ -shared -fPIC -o libtoken.so main.cpp base.cpp cache.cpp / for linux
    *- '.dll': g++ -shared -o libtoken.dll main.cpp base.cpp cache.cpp / for windows
*/

#ifndef __MAIN__H__
#define __MAIN__H__

#include "base.h"
#include <stdbool.h>

typedef struct {
  BaseTokenizer base;
} Shred;

extern "C" {
  void init_shred(Shred* tokenizer);
  void train(Shred* tokenizer, const char* text, int vocab_size);
  void dynamic_train_bpe(Shred* tokenizer, const char* text, int vocab_size, int min_freq);

  int* encode(Shred* tokenizer, const char* text, int* output_size);
  int* encode_with_cache(Shred* tokenizer, const char* text, int* output_size);
  char* decode(Shred* tokenizer, const int* ids, int ids_size);
  char* decode_with_cache(Shred* tokenizer, const int* ids, int ids_size);

  void save_model(const Shred* tokenizer, const char* file_path);
  void load_model(Shred* tokenizer, const char* model_file);

  char* export_merges(const Shred* tokenizer);
  void free_string(char* string);
}

#endif