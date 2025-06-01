#ifndef __HASH__H__
#define __HASH__H__

#include <stddef.h>
#include <stdint.h>
#include <math.h>

// Forward declaration
typedef struct TrieNode TrieNode;

typedef struct MemoryArena {
  char* memory;
  size_t size;
  size_t used;
  struct MemoryArena* next;
} MemoryArena;

// String interning system
typedef struct InternedString {
  const char* str;
  uint32_t len;
  uint32_t hash;
  uint32_t id;
} InternedString;

typedef struct StringInterner {
  InternedString* strings;
  uint32_t* hash_table;
  uint32_t capacity;
  uint32_t count;
  uint32_t hash_mask;
  MemoryArena* arena;
} StringInterner;

typedef struct HashVocabEntry {
  uint32_t string_id;
  uint64_t frequency;
  double score;
  uint16_t length;
} HashVocabEntry; // Entry for final vocabulary

typedef struct VocabBuilder {
  TrieNode* root;
  StringInterner* interner;
  uint32_t* freq_buffer;
  uint32_t* id_buffer;
  size_t buffer_size;
  size_t min_frequency;
  size_t max_entries;
} VocabBuilder; // Optimized vocabulary builder

extern "C" {
  // Memory arena functions
  MemoryArena* arena_create(size_t size);
  void* arena_alloc(MemoryArena** arena_ptr, size_t size);
  void arena_free_all(MemoryArena* arena);

  // String interner functions
  StringInterner* interner_create(uint32_t capacity);
  void interner_free(StringInterner* interner);
  uint32_t interner_add(StringInterner* interner, const char* str, uint32_t len);
  const char* interner_get_string(const StringInterner* interner, uint32_t id);
  uint32_t interner_get_length(const StringInterner* interner, uint32_t id);

  // Vocabulary builder functions
  VocabBuilder* vocab_builder_create(size_t max_entries, size_t min_frequency);
  void vocab_builder_add_line(VocabBuilder* builder, const char* line, size_t max_subword_len);
  size_t vocab_builder_finalize(VocabBuilder* builder, HashVocabEntry** out_entries);
  void vocab_builder_free(VocabBuilder* builder);
}

#endif  //!__HASH__H__