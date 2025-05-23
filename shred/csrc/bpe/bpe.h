/** 
  @brief bpe.h header file for the bpr trainer code logic.
  * each new vocab is determined & merged based on traditional bpe merging
      with help of hashing & heaps for faster merges.
  * main entry point file code for BPE-trainer related codebase.
  * compile it as:
    *- '.so': g++ -shared -fPIC -o libtrainer.so bpe.cpp hash.cpp heap.cpp
    *- '.dll': g++ -shared -o libtrainer.dll bpe.cpp hash.cpp heap.cpp
*/

#ifndef __BPE__H__
#define __BPE__H__

#include <stdint.h>
#include "heap.h"
#include "hash.h"

#define  MIN_HEAP_SIZE  4096
#define  INITIAL_VOCAB_SIZE  256  // UTF-8 base chars from 0 -> 255

#endif  //!__BPE__H__