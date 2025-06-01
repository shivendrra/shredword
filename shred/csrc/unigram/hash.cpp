#include <stdlib.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include "../trie.h"
#include "hash.h"

// faster hash functions
static inline uint32_t xxhash32(const void* data, size_t len, uint32_t seed) {
  const uint8_t* p = (const uint8_t*)data;
  const uint8_t* end = p + len;
  uint32_t h = seed + 374761393U;

  while (p + 4 <= end) {
    uint32_t k = *(const uint32_t*)p;
    k *= 3266489917U;
    k = (k << 15) | (k >> 17);
    k *= 668265263U;
    h ^= k;
    h = (h << 13) | (h >> 19);
    h = h * 5 + 3864292196U;
    p += 4;
  }

  while (p < end) {
    h ^= *p++;
    h *= 16777619U;
  }
  
  h ^= h >> 16;
  h *= 2246822519U;
  h ^= h >> 13;
  h *= 3266489917U;
  h ^= h >> 16;
  
  return h;
}

static inline uint64_t rolling_hash_init(const char* str, size_t len) {
  uint64_t hash = 0;
  const uint64_t prime = 31;
  for (size_t i = 0; i < len; i++) {
    hash = hash * prime + (unsigned char)str[i];
  }
  return hash;
}

static inline uint64_t rolling_hash_update(uint64_t hash, char old_char, char new_char, uint64_t prime_pow) {
  hash -= (unsigned char)old_char * prime_pow;
  hash = hash * 31 + (unsigned char)new_char;
  return hash;
}

// Memory arena functions
MemoryArena* arena_create(size_t size) {
  MemoryArena* arena = (MemoryArena*)malloc(sizeof(MemoryArena));
  arena->memory = (char*)malloc(size);
  arena->size = size;
  arena->used = 0;
  arena->next = NULL;
  return arena;
}

void* arena_alloc(MemoryArena** arena_ptr, size_t size) {
  size = (size + 7) & ~7; // 8-byte alignment
  
  MemoryArena* arena = *arena_ptr;
  if (arena->used + size > arena->size) {
    // Need new arena
    MemoryArena* new_arena = arena_create(arena->size * 2);
    new_arena->next = arena;
    *arena_ptr = new_arena;
    arena = new_arena;
  }
  
  void* ptr = arena->memory + arena->used;
  arena->used += size;
  return ptr;
}

void arena_free_all(MemoryArena* arena) {
  while (arena) {
    MemoryArena* next = arena->next;
    free(arena->memory);
    free(arena);
    arena = next;
  }
}

// String interner functions
StringInterner* interner_create(uint32_t capacity) {
  StringInterner* interner = (StringInterner*)malloc(sizeof(StringInterner));
  interner->capacity = capacity;
  interner->hash_mask = capacity - 1;
  interner->count = 0;
  interner->strings = (InternedString*)calloc(capacity, sizeof(InternedString));
  interner->hash_table = (uint32_t*)malloc(sizeof(uint32_t) * capacity);
  interner->arena = arena_create(1024 * 1024); // 1MB initial
  
  for (uint32_t i = 0; i < capacity; i++) {
    interner->hash_table[i] = UINT32_MAX;
  }
  
  return interner;
}

uint32_t interner_add(StringInterner* interner, const char* str, uint32_t len) {
  uint32_t hash = xxhash32(str, len, 0);
  uint32_t idx = hash & interner->hash_mask;
  
  // Linear probing
  while (interner->hash_table[idx] != UINT32_MAX) {
    uint32_t existing_id = interner->hash_table[idx];
    InternedString* existing = &interner->strings[existing_id];
    
    if (existing->hash == hash && existing->len == len &&
        memcmp(existing->str, str, len) == 0) {
      return existing_id;
    }
    
    idx = (idx + 1) & interner->hash_mask;
  }
  
  // Add new string
  if (interner->count >= interner->capacity * 0.75) {
    fprintf(stderr, "[ERROR] String interner full\n");
    return UINT32_MAX;
  }
  
  uint32_t new_id = interner->count++;
  char* interned_str = (char*)arena_alloc(&interner->arena, len + 1);
  memcpy(interned_str, str, len);
  interned_str[len] = '\0';
  
  interner->strings[new_id] = (InternedString){
    .str = interned_str,
    .len = len,
    .hash = hash,
    .id = new_id
  };
  
  interner->hash_table[idx] = new_id;
  return new_id;
}

const char* interner_get_string(const StringInterner* interner, uint32_t id) {
  if (id >= interner->count) return NULL;
  return interner->strings[id].str;
}

uint32_t interner_get_length(const StringInterner* interner, uint32_t id) {
  if (id >= interner->count) return 0;
  return interner->strings[id].len;
}

void interner_free(StringInterner* interner) {
  arena_free_all(interner->arena);
  free(interner->strings);
  free(interner->hash_table);
  free(interner);
}

// Vocabulary builder functions
VocabBuilder* vocab_builder_create(size_t max_entries, size_t min_frequency) {
  VocabBuilder* builder = (VocabBuilder*)malloc(sizeof(VocabBuilder));
  builder->root = trie_create_node();
  builder->interner = interner_create(max_entries * 4);
  builder->freq_buffer = (uint32_t*)malloc(sizeof(uint32_t) * max_entries);
  builder->id_buffer = (uint32_t*)malloc(sizeof(uint32_t) * max_entries);
  builder->buffer_size = max_entries;
  builder->min_frequency = min_frequency;
  builder->max_entries = max_entries;
  return builder;
}

void vocab_builder_add_line(VocabBuilder* builder, const char* line, size_t max_subword_len) {
  size_t len = strlen(line);
  if (len == 0) return;
  
  // Pre-compute prime powers for rolling hash
  uint64_t prime_powers[32];
  prime_powers[0] = 1;
  for (int i = 1; i < 32 && i <= (int)max_subword_len; i++) {
    prime_powers[i] = prime_powers[i-1] * 31;
  }
  
  // Generate all substrings with rolling hash
  for (size_t start = 0; start < len; start++) {
    size_t max_end = (start + max_subword_len < len) ? start + max_subword_len : len;
    
    for (size_t end = start + 1; end <= max_end; end++) {
      size_t substr_len = end - start;
      trie_insert(builder->root, builder->interner, line + start, substr_len, builder->min_frequency);
    }
  }
}

// Comparison function for qsort (descending frequency order)
static int compare_vocab_entries(const void* a, const void* b) {
  const HashVocabEntry* entry_a = (const HashVocabEntry*)a;
  const HashVocabEntry* entry_b = (const HashVocabEntry*)b;
  
  if (entry_a->frequency > entry_b->frequency) return -1;
  if (entry_a->frequency < entry_b->frequency) return 1;
  return 0;
}

size_t vocab_builder_finalize(VocabBuilder* builder, HashVocabEntry** out_entries) {
  HashVocabEntry* entries = (HashVocabEntry*)malloc(sizeof(HashVocabEntry) * builder->max_entries);
  size_t count = 0;
  
  trie_collect_entries(builder->root, entries, &count, builder->max_entries, builder->interner);
  
  // Sort by frequency (descending) using qsort
  qsort(entries, count, sizeof(HashVocabEntry), compare_vocab_entries);
  // Compute log probabilities
  uint64_t total_freq = 0;
  for (size_t i = 0; i < count; i++) {
    total_freq += entries[i].frequency;
  }
  
  if (total_freq > 0) {
    double log_total = log((double)total_freq);
    for (size_t i = 0; i < count; i++) {
      entries[i].score = log((double)entries[i].frequency) - log_total;
    }
  }
  
  *out_entries = entries;
  return count;
}

void vocab_builder_free(VocabBuilder* builder) {
  if (!builder) return;
  
  trie_free(builder->root);
  interner_free(builder->interner);
  free(builder->freq_buffer);
  free(builder->id_buffer);
  free(builder);
}