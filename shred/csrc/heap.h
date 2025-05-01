#ifndef __HEAP__H__
#define __HEAP__H__

typedef struct {
  char* a; // first symbol
  char* b; // second symbol
} PairKey;

typedef struct {
  char* key;  // single character or maybe merged
  int freq;
} HeapEntry; // An entry in the heap

typedef struct {
  HeapEntry* data;
  int size;
  int cap;
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