#include <stdio.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include "heap.h"
#include "../bpe/bpe.h"
#include "hash.h"

/**
  @brief Swap two heap entries in-place.
  * @param x  Pointer to the first entry.
  * @param y  Pointer to the second entry.
 */
static void he_swap(HeapEntry* x, HeapEntry* y) {
  assert(x && y);
  HeapEntry tmp = *x;
  *x = *y;
  *y = tmp;
}

/**
  @brief Initialize a max‑heap.
  @param h Pointer to MaxHeap struct to initialize.
  @param capacity Initial capacity (number of entries) to reserve.
*/
void heap_init(MaxHeap* h, size_t capacity) {
  if (h == NULL) {
    fprintf(stderr, "Error: Heap pointer is NULL.\n");
    exit(EXIT_FAILURE);
  }
  if (capacity == 0) {
    fprintf(stderr, "Error: Heap capacity must be > 0.\n");
    exit(EXIT_FAILURE);
  }

  h->data = (HeapEntry *)malloc(sizeof(HeapEntry) * capacity);
  if (!h->data) {
    fprintf(stderr, "Memory allocation failed for heap data!\n");
    exit(EXIT_FAILURE);
  }
  h->size = 0;
  h->cap = capacity;
}

/**
  @brief Push a key/frequency pair onto the heap.
          Grows the underlying array if needed.
  @param h Pointer to the heap.
  @param key Merged key that needs to be updated.
  @param freq Integer frequency used for ordering (max‑heap).
  @param freq Version for lazy invalidation.
 */
void heap_push(MaxHeap* h, PairKey key, uint64_t freq, uint32_t version) {
  if (h == NULL) {
    fprintf(stderr, "Error: Heap pointer is NULL.\n");
    exit(EXIT_FAILURE);
  }
  // grow if needed
  if (h->size == h->cap) {
    size_t new_cap = h->cap * 2;
    HeapEntry* new_data = (HeapEntry*)realloc(h->data, sizeof(HeapEntry) * new_cap);
    if (!new_data) {
      fprintf(stderr, "Memory reallocation failed!\n");
      exit(EXIT_FAILURE);
    }
    h->data = new_data;
    h->cap = new_cap;
  }  
  // insert at end and sift up
  size_t idx = h->size++;
  h->data[idx].key = key;
  h->data[idx].freq = freq;
  h->data[idx].version = version;
  while (idx > 0) {
    size_t p = (idx - 1) >> 1;
    if (h->data[p].freq >= h->data[idx].freq) break;
    he_swap(&h->data[p], &h->data[idx]);
    idx = p;
  }
}

/**
  @brief Pop the top (highest-frequency) entry from the heap.
          The returned HeapEntry.key must be freed by the caller.
  @param h Pointer to the heap.
  @return The popped HeapEntry.
 */
HeapEntry heap_pop(MaxHeap* h) {
  if (h == NULL) {
    fprintf(stderr, "Error: Heap pointer is Null.\n");
    exit(EXIT_FAILURE);
  }
  if (h->size == 0) {
    fprintf(stderr, "Error: Cannot pop from empty heap.\n");
    exit(EXIT_FAILURE);
  }
  HeapEntry top = h->data[0];
  h->data[0] = h->data[--h->size];

  size_t idx = 0;
  while (true) {
    size_t left = (idx << 1) + 1, right = left + 1, best = idx;
    if (left < h->size && h->data[left].freq > h->data[best].freq)
      best = left;
    if (right < h->size && h->data[right].freq > h->data[best].freq)
      best = right;
    if (best == idx)
      break;
    he_swap(&h->data[idx], &h->data[best]);
    idx = best;
  }

  return top;
}

/**
  @brief Check if the heap is empty.
  @param h Pointer to the heap.
  @return Non-zero if empty, zero otherwise.
*/
int heap_empty(MaxHeap* h) {
  if (h == NULL) {
    fprintf(stderr, "Error: Heap pointer is Null.\n");
    exit(EXIT_FAILURE);
  }
  return h->size == 0;
}

/**
  @brief Free all resources held by the heap (but not the heap struct itself).
  @param h  Pointer to the heap.
*/
void heap_free(MaxHeap* h) {
  if (h == NULL) {
    fprintf(stderr, "Error: Heap pointer is Null, can't free the Memory.\n");
    exit(EXIT_FAILURE);
  }
  free(h->data);
  h->data = NULL;
  h->size = h->cap = 0;
}