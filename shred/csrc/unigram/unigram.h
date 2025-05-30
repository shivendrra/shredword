/** 
  @brief uingram.h header file for the unigram trainer code logic.
  * each new vocab is determined & merged based EM using unigram approach
      with help of hashing & heaps for faster merges.
  * main entry point file code for Unigram-trainer related codebase.
  * compile it as:
    *- '.so': g++ -shared -fPIC -o libtrainer.so unigram/unigram.cpp inc/hash.cpp inc/heap.cpp
    *- '.dll': g++ -shared -o libtrainer.dll uingram/unigram.cpp inc/hash.cpp inc/heap.cpp
*/

#ifndef __UNIGRAM__H__
#define __UNIGRAM__H__

#include <stdint.h>

#endif  //!__UNIGRAM__H__