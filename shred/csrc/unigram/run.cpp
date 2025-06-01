#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "core.h"
#include "training.h"

#define MAX_LINES 10000
#define MAX_LINE_LENGTH 512

int main(int argc, char* argv[]) {
    if (argc != 2) {
        printf("Usage: %s <input_text_file>\n", argv[0]);
        return 1;
    }
    
    const char* filename = argv[1];
    FILE* file = fopen(filename, "r");
    if (!file) {
        fprintf(stderr, "Error: Cannot open file '%s'\n", filename);
        return 1;
    }
    
    // Read lines from file
    char** lines = (char**)malloc(sizeof(char*) * MAX_LINES);
    size_t num_lines = 0;
    char buffer[MAX_LINE_LENGTH];
    
    printf("Reading text file...\n");
    while (fgets(buffer, sizeof(buffer), file) && num_lines < MAX_LINES) {
        // Remove newline
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len-1] == '\n') {
            buffer[len-1] = '\0';
            len--;
        }
        
        // Skip empty lines
        if (len == 0) continue;
        
        // Allocate and copy line
        lines[num_lines] = (char*)malloc(len + 1);
        strcpy(lines[num_lines], buffer);
        num_lines++;
    }
    fclose(file);
    
    if (num_lines == 0) {
        printf("No valid lines found in file\n");
        for (size_t i = 0; i < num_lines; i++) {
            free(lines[i]);
        }
        free(lines);
        return 1;
    }
    
    printf("Read %zu lines from file\n", num_lines);
    
    // Training parameters
    const size_t max_subword_len = 8;    // Maximum subword length
    const size_t min_frequency = 2;      // Minimum frequency to include in vocab
    const size_t target_vocab_size = 1000; // Target vocabulary size
    
    printf("Creating model from text...\n");
    UnigramModel* model = create_model_from_text(
        (const char**)lines, 
        num_lines, 
        max_subword_len, 
        min_frequency, 
        target_vocab_size
    );
    
    if (!model) {
        fprintf(stderr, "Failed to create model\n");
        // Cleanup lines and exit
        for (size_t i = 0; i < num_lines; i++) {
            free(lines[i]);
        }
        free(lines);
        return 1;
    }
    
    printf("Model created with %zu entries\n", model->size);
    
    // Declare variables here (before any complex control flow)
    const size_t max_em_steps = 10;
    size_t test_samples = (num_lines < 3) ? num_lines : 3;
    
    // Run EM training
    printf("Starting EM training...\n");
    run_em_training(model, (const char* const*)lines, num_lines, max_em_steps);
    
    printf("Training completed!\n");
    
    // Test tokenization on a few sample lines
    printf("\n=== Testing Tokenization ===\n");
    
    for (size_t i = 0; i < test_samples; i++) {
        size_t token_count = 0;
        char** tokens = viterbi_tokenize(model, lines[i], &token_count);
        
        printf("Input:  \"%s\"\n", lines[i]);
        printf("Tokens: ");
        
        if (tokens && token_count > 0) {
            for (size_t j = 0; j < token_count; j++) {
                printf("\"%s\"", tokens[j]);
                if (j < token_count - 1) printf(" ");
                free(tokens[j]);
            }
            free(tokens);
        } else {
            printf("(no tokens)");
        }
        printf("\n\n");
    }
    
    // Test token ID encoding
    printf("=== Testing Token ID Encoding ===\n");
    if (num_lines > 0) {
        size_t id_count = 0;
        int* ids = encode_to_ids(model, lines[0], &id_count);
        
        printf("Input: \"%s\"\n", lines[0]);
        printf("IDs:   ");
        
        if (ids && id_count > 0) {
            for (size_t i = 0; i < id_count; i++) {
                printf("%d", ids[i]);
                if (i < id_count - 1) printf(" ");
            }
            free(ids);
        } else {
            printf("(no IDs)");
        }
        printf("\n");
    }
    
    // Optional: dump model info
    printf("\n=== Model Information ===\n");
    dump_unigram_model(model);
    
    // Cleanup
    free_unigram_model(model);
    
    for (size_t i = 0; i < num_lines; i++) {
        free(lines[i]);
    }
    free(lines);
    
    printf("Test completed successfully!\n");
    return 0;
}