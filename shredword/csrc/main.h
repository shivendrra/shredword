/*
  main.h
  - main ``ShredWord`` class codes in this file
  - wrapper over the ``base.cpp`` & it's functions, handling the training, encoding & decoding taks
  - to be compiled with ``base.cpp`` containing the main logic (no regex, no caching)
  - compile it as:
    -- '.so': g++ -shared -fPIC -o libtoken.so main.cpp base.cpp / for linux
    -- '.dll': g++ -shared -o libtoken.dll main.cpp base.cpp / for windows
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
  void train(Shred* tokenizer, const char* text, int vocab_size, bool verbose);
  char* decode(Shred* tokenizer, const int* ids, int ids_size);
  int* encode(Shred* tokenizer, const char* text, int* output_size);
}

#endif