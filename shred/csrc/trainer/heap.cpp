#include <stdio.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include "heap.h"
#include "bpe.h"
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
 @brief Check whether a popped heap entry is still fresh.
        Compares its version against the bigram-info table.
*/
bool is_version_valid(const PairKey &key, uint32_t version) {
  // if the current version in the map matches, it’s valid
  return bpe_get_current_version(key) == version;
}

/**
  @brief Initialize a max‑heap.
  @param h Pointer to MaxHeap struct to initialize.
  @param capacity Initial capacity (number of entries) to reserve.
*/

void heap_init(MaxHeap* h, size_t capacity) {
  if (h == NULL) {
    fprintf(stderr, "Error: Heap pointer is Null.\n");
    exit(EXIT_FAILURE);
  }
  h->data = (HeapEntry*)malloc(sizeof(HeapEntry) * capacity);
  if (!h) {
    fprintf(stderr, "Pointer allocation failed!\n");
    exit(EXIT_FAILURE);
  }
  h->size = 0; h->cap = capacity;
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
    fprintf(stderr, "Error: Heap pointer is Null.\n");
    exit(EXIT_FAILURE);
  }
  // grow if needed
  if (h->size == h->cap) {
    h->cap *= 2;
    h->data = (HeapEntry*)realloc(h->data, sizeof(HeapEntry) * h->cap);
    if (!h->data) {
      fprintf(stderr, "Pointer allocation failed!\n");
      exit(EXIT_FAILURE);
    }
  }
  // insert at end and sift up
  int idx = h->size++;
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
  assert(h->size > 0);
  HeapEntry top;

  do {
    // Remove the current root
    top = h->data[0];
    h->data[0] = h->data[--h->size];

    // Sift down to restore max‑heap property
    size_t idx = 0;
    while (true) {
      size_t left = (idx << 1) + 1, right = left + 1, best = idx;
      if (left < h->size && h->data[left].freq > h->data[best].freq) best = left;
      if (right < h->size && h->data[right].freq > h->data[best].freq) best = right;
      if (best == idx) break;
      he_swap(&h->data[idx], &h->data[best]);
      idx = best;
    }

    // If this entry’s version no longer matches the bigram’s current version,
    // it’s stale—pop again.
  } while (!is_version_valid(top.key, top.version));
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
  h->data = NULL; h->size = h->cap = 0;
}