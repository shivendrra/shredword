#ifndef REGEX_H
#define REGEX_H

#include <regex.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

int match_pattern(const char* string, const char* pattern);
char* replace_pattern(const char* string, const char* pattern, const char* replacement);
char** split_pattern(const char* string, const char* pattern, int* out_count);
int validate_format(const char* string, const char* pattern);
void free_split_results(char** results, int count);

#endif