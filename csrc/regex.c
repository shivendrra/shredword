#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <regex.h>

#define MAX_MATCHES 10
// #define REG_EXTENDED 10;

int match_pattern(const char* string, const char* pattern) {
  regex_t regex;
  int result;

  result = regcomp(&regex, pattern, REG_EXTENDED);
  if (result) {
    fprintf(stderr, "Could not compile regex.\n");
    return 0;
  }

  result = regexec(&regex, string, 0, NULL, 0);
  regfree(&regex);

  return !result;
}

char* replace_pattern(const char* string, const char* pattern, const char* replacement) {
  regex_t regex;
  regmatch_t matches[MAX_MATCHES];
  int result;
  size_t len;
  char *new_str, *temp_str;
  int start, end;

  result = regcomp(&regex, pattern, REG_EXTENDED);
  if (result) {
    fprintf(stderr, "Could not compile regex.\n");
    return NULL;
  }

  len = strlen(string) + 1;
  new_str = (char*)malloc(len * sizeof(char));
  strcpy(new_str, string);

  while (regexec(&regex, new_str, MAX_MATCHES, matches, 0) == 0) {
    start = matches[0].rm_so;
    end = matches[0].rm_eo;
    len += strlen(replacement) - (end - start);
    temp_str = (char*)malloc(len * sizeof(char));

    strncpy(temp_str, new_str, start);
    temp_str[start] = '\0';
    strcat(temp_str, replacement);
    strcat(temp_str, new_str + end);

    free(new_str);  // Free the old string
    new_str = temp_str;
  }

  regfree(&regex);  // Free regex memory
  return new_str;   // Return the modified string
}

char** split_pattern(const char* string, const char* pattern, int* out_count) {
  regex_t regex;
  regmatch_t matches[MAX_MATCHES];
  int result, count = 0;
  char **split_result = NULL;
  int start, end, prev_end = 0;

  result = regcomp(&regex, pattern, REG_EXTENDED);
  if (result) {
    fprintf(stderr, "Could not compile regex.\n");
    return NULL;
  }

  split_result = (char**)malloc(sizeof(char*));  // Allocate initial memory for split results
  while (regexec(&regex, string + prev_end, 1, matches, 0) == 0) {
    start = matches[0].rm_so + prev_end;
    end = matches[0].rm_eo + prev_end;

    split_result = (char**)realloc(split_result, sizeof(char*) * (count + 1));
    split_result[count] = (char*)malloc((start - prev_end + 1) * sizeof(char));
    strncpy(split_result[count], string + prev_end, start - prev_end);
    split_result[count][start - prev_end] = '\0';

    count++;
    prev_end = end;
  }

  split_result = (char**)realloc(split_result, sizeof(char*) * (count + 1));
  split_result[count] = strdup(string + prev_end);
  count++;

  *out_count = count;  // Set the output count
  regfree(&regex);     // Free regex memory
  return split_result; // Return the array of split strings
}

int validate_format(const char* string, const char* pattern) {
  return match_pattern(string, pattern);
}

void free_split_results(char** results, int count) {
  for (int i = 0; i < count; i++) {
    free(results[i]);
  }
  free(results);
}