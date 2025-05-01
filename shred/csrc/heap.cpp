#include <stdio.h>
#include <assert.h>
#include "heap.h"

// swap two heap entries
static void he_swap(HeapEntry* x, HeapEntry* y) {
  if (x == NULL || y == NULL) {
    fprintf(stderr, "Error: HeapEntry pointers are Null.\n");
    exit(EXIT_FAILURE);
  }
  HeapEntry t = *x; *x = *y; *y = t;
}

void heap_init(MaxHeap* h, int capacity) {
  if (h == NULL) {
    fprintf(stderr, "Error: Heap pointer is Null.\n");
    exit(EXIT_FAILURE);
  }
  h->data = (HeapEntry*)malloc(sizeof(HeapEntry) * capacity);
  h->size = 0; h->cap = capacity;
}

void heap_push(MaxHeap* h, char* key, int freq) {
  if (h == NULL) {
    fprintf(stderr, "Error: Heap pointer is Null.\n");
    exit(EXIT_FAILURE);
  }
  // grow if needed
  if (h->size == h->cap) {
    h->cap *= 2;
    h->data = (HeapEntry*)realloc(h->data, sizeof(HeapEntry) * h->cap);
  }
  // insert at end and sift up
  int idx = h->size++;
  h->data[idx].key = key;
  h->data[idx].freq = freq;
  while (idx > 0) {
    int p = (idx - 1) >> 1;
    if (h->data[p].freq >= h->data[idx].freq) break;
    he_swap(&h->data[p], &h->data[idx]);
    idx = p;
  }
}

HeapEntry heap_pop(MaxHeap* h) {
  if (h == NULL) {
    fprintf(stderr, "Error: Heap pointer is Null.\n");
    exit(EXIT_FAILURE);
  }
  assert(h->size > 0);
  HeapEntry top = h->data[0];
  h->data[0] = h->data[--h->size];
  // sift down
  int idx = 0;
  while (1) {
    int l = (idx<<1) + 1, r = l+1, best = idx;
    if (l < h->size && h->data[l].freq > h->data[best].freq) best = l;
    if (r < h->size && h->data[r].freq > h->data[best].freq) best = r;
    if (best == idx) break;
    he_swap(&h->data[idx], &h->data[best]);
    idx = best;
  }
  return top;
}

int heap_empty(MaxHeap* h) {
  if (h == NULL) {
    fprintf(stderr, "Error: Heap pointer is Null.\n");
    exit(EXIT_FAILURE);
  }
  return h->size == 0;
}

void heap_free(MaxHeap* h) {
  if (h == NULL) {
    fprintf(stderr, "Error: Heap pointer is Null, can't free the Memory.\n");
    exit(EXIT_FAILURE);
  }
  free(h->data);
  h->data = NULL; h->size = h->cap = 0;
}