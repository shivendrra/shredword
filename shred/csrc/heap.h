/**
 @file heap.h
 @brief A simple max‑heap implementation over string keys and integer frequencies.

 * This heap is used in the BPE merge process to always pop the
 * highest‑frequency symbol pair. Keys are C‑strings (heap owns them),
 * and must be freed after use. 
*/

#ifndef __HEAP__H__
#define __HEAP__H__

#include <stdint.h>
#include <stddef.h>
#include "inc/chash.h"

typedef struct HeapEntry {
  PairKey key;
  uint64_t freq;
  uint32_t version;
} HeapEntry; // An entry in the heap

typedef struct MaxHeap {
  HeapEntry* data;  // array of heap entries
  size_t size;   // current no of elements
  size_t cap;    // allocation capacity MaxHeap
} MaxHeap;  // A simple max-heap over HeapEntry

extern "C" {
  // heap related functions
  void heap_init(MaxHeap* h, size_t capacity);
  void heap_push(MaxHeap* h, PairKey key, uint64_t freq, uint32_t version);
  HeapEntry heap_pop(MaxHeap* h); // removes & returns top
  bool is_version_valid(const PairKey &key, uint32_t version);
  int heap_empty(MaxHeap* h);
  void heap_free(MaxHeap* h);
}

#endif  //!__HEAP__H__