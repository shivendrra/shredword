#include <ctype.h>
#include <string.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdint.h>
#include "normalize.h"

// UTF-8 sequence for U+2581 (▁)
static const unsigned char u2581[] = {0xE2, 0x96, 0x81};

// Fast check if character is whitespace
static inline int is_whitespace(char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// Fast check for UTF-8 space marker sequence
static inline int is_space_marker(const char* p) {
  return (unsigned char)p[0] == 0xE2 && 
         (unsigned char)p[1] == 0x96 && 
         (unsigned char)p[2] == 0x81;
}

int normalize_line(const char* input, char* output, size_t output_size) {
  if (!input || !output || output_size == 0) {
    return -1;
  }

  const char* in = input;
  char* out = output;
  char* out_end = output + output_size - 1; // Reserve space for null terminator
  int in_space = 1; // Treat start as space

  while (*in && out < out_end) {
    if (is_whitespace(*in)) {
      if (!in_space) {
        // Check if we have space for the 3-byte UTF-8 sequence
        if (out + 3 > out_end) {
          break;
        }
        *out++ = u2581[0];
        *out++ = u2581[1];
        *out++ = u2581[2];
        in_space = 1;
      }
    } else {
      *out++ = tolower((unsigned char)*in);
      in_space = 0;
    }
    ++in;
  }
  // removing trailing space symbol if present
  if (out >= output + 3 && is_space_marker(out - 3)) {
    out -= 3;
  }

  *out = '\0';
  return out - output;
}

// hash function (FNV-1a)
static inline uint32_t hash_fnv1a(const char* str, size_t len) {
  uint32_t hash = 2166136261u;
  for (size_t i = 0; i < len; i++) {
    hash ^= (unsigned char)str[i];
    hash *= 16777619u;
  }
  return hash;
}

// finds next prime number >= n
static size_t next_prime(size_t n) {
  if (n <= 2) return 2;
  if (n % 2 == 0) n++;
    
  while (1) {
    int is_prime = 1;
    for (size_t i = 3; i * i <= n; i += 2) {
      if (n % i == 0) {
        is_prime = 0;
        break;
      }
    }
    if (is_prime) return n;
    n += 2;
  }
}

VocabTable* create_vocab(size_t initial_capacity) {
  if (initial_capacity == 0) {
    initial_capacity = INITIAL_CAPACITY;
  }
  VocabTable* table = (VocabTable*)malloc(sizeof(VocabTable));
  if (!table) {
    fprintf(stderr, "[ERROR]\t Couldn't assign Table pointer\n");
    exit(EXIT_FAILURE);
  }

  // use prime number for better hash distribution
  table->capacity = next_prime(initial_capacity);
  table->entries = (VocabEntry* )calloc(table->capacity, sizeof(VocabEntry));
  if (!table->entries) {
    fprintf(stderr, "[ERROR]\t Couldn't allocate memory to pointer\n");
    exit(EXIT_FAILURE);
  }
    
  table->size = 0;
  table->threshold = (size_t)(table->capacity * LOAD_FACTOR_THRESHOLD);
  return table;
}

void free_vocab(VocabTable* table) {
  if (!table) {
    fprintf(stderr, "[ERROR]\t Table pointer is NULL!\n");
    exit(EXIT_FAILURE);    
  }
  if (table->entries) {
    for (size_t i = 0; i < table->capacity; i++) {
      if (table->entries[i].str) {
        free(table->entries[i].str);
      }
    }
    free(table->entries);
  }
  free(table);
}

// resize hash table when load factor exceeds threshold
static int resize_vocab_table(VocabTable* table) {
  if (!table) {
    fprintf(stderr, "[ERROR]\t Table pointer is NULL!\n");
    exit(EXIT_FAILURE);    
  }
  size_t old_capacity = table->capacity;
  VocabEntry* old_entries = table->entries;

  // double capacity and find next prime
  table->capacity = next_prime(old_capacity * 2);
  table->entries = (VocabEntry* )calloc(table->capacity, sizeof(VocabEntry));
  if (!table->entries) {
    table->entries = old_entries;
    table->capacity = old_capacity;
    return -1;
  }

  table->threshold = (size_t)(table->capacity * LOAD_FACTOR_THRESHOLD);
  size_t old_size = table->size;
  table->size = 0;

    // Rehash all existing entries
  for (size_t i = 0; i < old_capacity; i++) {
    if (old_entries[i].str) {
      uint32_t h = hash_fnv1a(old_entries[i].str, old_entries[i].str_len) % table->capacity;

      // Find empty slot using quadratic probing
      size_t probe = 0;
      while (table->entries[h].str) {
        probe++;
        h = (h + probe * probe) % table->capacity;
      }

      table->entries[h] = old_entries[i]; // Move entry (no string copy needed)
      table->size++;
    }
  }
    
  free(old_entries);
  return 0;
}

static int insert_or_increment(VocabTable* table, const char* s, size_t len) {
  if (!table) {
    fprintf(stderr, "[ERROR]\t Table pointer is NULL!\n");
    exit(EXIT_FAILURE);    
  }
  if (len >= MAX_SUBWORD_LEN || len == 0) {
    return 0;
  }

  // resize when necessary
  if (table->size >= table->threshold) {
    if (resize_vocab_table(table) != 0) {
      return -1; // Resize failed
    }
  }

  uint32_t h = hash_fnv1a(s, len) % table->capacity;
  size_t probe = 0;

  // Quadratic probing for better distribution
  while (table->entries[h].str) {
    if (table->entries[h].str_len == len && 
      memcmp(table->entries[h].str, s, len) == 0) {
      table->entries[h].count++;
      return 0;
    }
    probe++;
    h = (h + probe * probe) % table->capacity;
  }
    
  // Create new entry
  table->entries[h].str = (char* )malloc(len + 1);
  if (!table->entries[h].str) {
    return -1;
  }

  memcpy(table->entries[h].str, s, len);
  table->entries[h].str[len] = '\0';
  table->entries[h].str_len = len;
  table->entries[h].count = 1;
  table->size++;
  return 0;
}

int add_subwords(VocabTable* table, const char* line, size_t max_len) {
  if (!table || !line || max_len == 0) {
    fprintf(stderr, "[ERROR]\t Table pointer is NULL!\n");
    exit(EXIT_FAILURE);    
  }

  const char* start = line;
  while (*start) {
    // skipping space markers efficiently
    if (is_space_marker(start)) {
      start += 3;
      continue;
    }
    // adding all subwords starting at this position
    for (size_t len = 1; len <= max_len && start[len - 1]; len++) {
      if (insert_or_increment(table, start, len) != 0) {
        return -1; // Error occurred
      }
    }
    start++;
  }
  return 0;
}

void dump_vocab(VocabTable* table) {
  if (!table || !table->entries) return;
    
  for (size_t i = 0; i < table->capacity; i++) {
    if (table->entries[i].str) {
      printf("%s %zu\n", table->entries[i].str, table->entries[i].count);
    }
  }
}