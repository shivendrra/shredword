/**
 @file heap.h
 @brief A simple max‑heap implementation over string keys and integer frequencies.

 * This heap is used in the BPE merge process to always pop the
 * highest‑frequency symbol pair. Keys are C‑strings (heap owns them),
 * and must be freed after use. 
*/

#ifndef __HEAP__H__
#define __HEAP__H__

typedef struct PairKey {
  char* a; // first symbol
  char* b; // second symbol
} PairKey;

typedef struct HeapEntry {
  char* key;  // single character or maybe merged
  int freq;
} HeapEntry; // An entry in the heap

typedef struct MaxHeap {
  HeapEntry* data;  // array of heap entries
  int size;   // current no of elements
  int cap;    // allocation capacity MaxHeap
} MaxHeap;  // A simple max-heap over HeapEntry

extern "C" {
  // heap related functions
  void heap_init(MaxHeap* h, int capacity);
  void heap_push(MaxHeap* h, char* key, int freq);
  HeapEntry heap_pop(MaxHeap* h); // removes & returns top
  int heap_empty(MaxHeap* h);
  void heap_free(MaxHeap* h);
}

#endif  //!__HEAP__H__