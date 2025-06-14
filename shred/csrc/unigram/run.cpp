// compilation & running
// -> g++ -o run test.cpp unigram/normalize.cpp
// -> chcp 65001
// -> run captions.txt

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "normalize.h"

// Buffer size for reading lines (can be adjusted based on expected line lengths)
#define READ_BUFFER_SIZE 8192
#define NORMALIZED_BUFFER_SIZE (READ_BUFFER_SIZE * 4)

static void print_usage(const char *program_name) {
  printf("Usage: %s <corpus_file> [max_subword_length]\n", program_name);
  printf("  corpus_file: Input text file to process\n");
  printf("  max_subword_length: Maximum subword length (default: 10, max: %d)\n", MAX_SUBWORD_LEN - 1);
}

static void print_stats(const VocabTable *vocab) {
  if (!vocab) return;
    
  size_t total_entries = vocab_size(vocab);
  fprintf(stderr, "Processed vocabulary statistics:\n");
  fprintf(stderr, "  Total unique subwords: %zu\n", total_entries);
  fprintf(stderr, "  Hash table capacity: %zu\n", vocab->capacity);
  fprintf(stderr, "  Load factor: %.2f%%\n", 
          total_entries > 0 ? (100.0 * total_entries / vocab->capacity) : 0.0);
}

int main(int argc, char *argv[]) {
  if (argc < 2 || argc > 3) {
    print_usage(argv[0]);
    return 1;
  }

  // Parse command line arguments
  const char *filename = argv[1];
  size_t max_subword_len = 10; // Default value

  if (argc == 3) {
    char *endptr;
    long parsed_len = strtol(argv[2], &endptr, 10);

    if (*endptr != '\0' || parsed_len < 1 || parsed_len >= MAX_SUBWORD_LEN) {
      fprintf(stderr, "Error: Invalid max_subword_length. Must be between 1 and %d\n", 
              MAX_SUBWORD_LEN - 1);
      return 1;
    }
    max_subword_len = (size_t)parsed_len;
  }

  // Open input file
  FILE *fp = fopen(filename, "r");
  if (!fp) {
    fprintf(stderr, "Error opening file '%s': %s\n", filename, strerror(errno));
    return 1;
  }

  // Create vocabulary table with reasonable initial capacity
  VocabTable *vocab = create_vocab(200000);
  if (!vocab) {
    fprintf(stderr, "Error: Failed to create vocabulary table\n");
    fclose(fp);
    return 1;
  }

  // Allocate buffers
  char *line = (char *)malloc(READ_BUFFER_SIZE);
  char *normalized = (char *)malloc(NORMALIZED_BUFFER_SIZE);
    
  if (!line || !normalized) {
    fprintf(stderr, "Error: Failed to allocate memory buffers\n");
    free(line);
    free(normalized);
    free_vocab(vocab);
    fclose(fp);
    return 1;
  }

  // Process file line by line
  size_t line_count = 0;
  size_t error_count = 0;
    
  fprintf(stderr, "Processing corpus file: %s\n", filename);
  fprintf(stderr, "Max subword length: %zu\n", max_subword_len);
    
  while (fgets(line, READ_BUFFER_SIZE, fp)) {
    line_count++;
        
    // Remove trailing newline if present
    size_t len = strlen(line);
    if (len > 0 && line[len - 1] == '\n') {
      line[len - 1] = '\0';
    }
        
    // Skip empty lines
    if (line[0] == '\0') {
      continue;
    }
        
    // Normalize the line
    int norm_result = normalize_line(line, normalized, NORMALIZED_BUFFER_SIZE);
    if (norm_result < 0) {
      fprintf(stderr, "Warning: Failed to normalize line %zu (too long?)\n", line_count);
      error_count++;
      continue;
    }
        
      // Add subwords to vocabulary
    if (add_subwords(vocab, normalized, max_subword_len) != 0) {
      fprintf(stderr, "Warning: Failed to add subwords from line %zu\n", line_count);
      error_count++;
      continue;
    }
        
    // Progress reporting for large files
    if (line_count % 10000 == 0) {
      fprintf(stderr, "Processed %zu lines, vocabulary size: %zu\n", 
                line_count, vocab_size(vocab));
    }
  }

    // Check for file reading errors
  if (ferror(fp)) {
    fprintf(stderr, "Error reading from file: %s\n", strerror(errno));
    free(line);
    free(normalized);
    free_vocab(vocab);
    fclose(fp);
    return 1;
  }

  fprintf(stderr, "Processing complete.\n");
  fprintf(stderr, "Lines processed: %zu (errors: %zu)\n", line_count, error_count);

    // Print statistics
  print_stats(vocab);

    // Output vocabulary to stdout
  fprintf(stderr, "Dumping vocabulary...\n");
  dump_vocab(vocab);

  // Cleanup
  free(line);
  free(normalized);
  free_vocab(vocab);
  fclose(fp);

  return (error_count > 0) ? 2 : 0; // Return 2 if there were processing errors
}