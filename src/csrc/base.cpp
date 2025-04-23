#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <unicode/unorm2.h>
#include <unicode/utypes.h>
#include <unicode/ustring.h>
#include "base.h"

TrieNode *create_node() {
  TrieNode* node = (TrieNode*)malloc(sizeof(TrieNode));
  node->terminal = false;
  for (int i = 0; i < NUM_CHARS; i++) {
    node->children[i] = NULL;
  }
  return node;
}

void trie_insert(TrieNode *root, const char *word) {
  if (root == NULL) {
    root = create_node();
  }
  unsigned char *idx = (unsigned char *)word;
  TrieNode *temp = root;
  size_t length = strlen(word);
  for (int i = 0; i < length; i++) {
    if (temp->children[idx[i]] == NULL) {
      temp->children[idx[i]] = create_node();
    }
    temp = temp->children[idx[i]];
  }
  temp->terminal = true;
}

int longest_prefix(TrieNode* root, const char* text) {
  if (root == NULL) {
    fprintf(stderr, "Error: Invaild Node to check for length.\n");
    exit(EXIT_FAILURE);
  }
  TrieNode* temp = root;
  int max_len = 0, pos = 0;
  while (text[pos] && temp->children[(unsigned char)text[pos]]) {
    temp = temp->children[(unsigned char)text[pos]];
    pos++;
    if (temp->terminal) {
      max_len = pos;
    }
  }
  return max_len;
}

void print_trie_recursively(TrieNode *node, unsigned char *prefix, int length) {
  unsigned char newprefix[length+2];
  memcpy(newprefix, prefix, length);
  newprefix[length+1] = 0;

  if (node->terminal) {
    printf("WORD: %s\n", prefix);
  }

  for (int i = 0; i < NUM_CHARS; i++) {
    if (node->children[i] != NULL) {
      newprefix[length] = i;
      print_trie_recursively(node->children[i], newprefix, length+1);
    }
  }
}

void print_trie(TrieNode *node) {
  if (node == NULL) {
    fprintf(stderr, "Error: Invaild Trie-Node to print.\n");
    exit(EXIT_FAILURE);
  }
  print_trie_recursively(node, NULL, 0);
}

void free_trie(TrieNode *node) {
  if (node == NULL) {
    fprintf(stderr, "Error: Invaild Trie-Node to free from memory.\n");
    exit(EXIT_FAILURE);
  }
  for (int i = 0; i < NUM_CHARS; ++i) {
    free_trie(node->children[i]);
  }
  delete node;
}

/**
  @brief Normalize input UTF-8 text to NFKC form and replace spaces with "▁" (U+2581).

  * This implementation dynamically allocates buffers based on the required output sizes.
  * It ensures correctness for inputs of arbitrary length and avoids fixed-size limitations.

  * @param input_text A null-terminated UTF-8 encoded string.
  * @return A newly allocated UTF-8 normalized string with spaces replaced.
          - The caller is responsible for freeing the returned string.
*/
char* normalize_text(const char* input_text) {
  UErrorCode status = U_ZERO_ERROR;

  const UNormalizer2* normalizer = unorm2_getNFKCInstance(&status);
  if (U_FAILURE(status)) {
    fprintf(stderr, "Failed to get NKFC-instance: %s\n", u_errorName(status));
    return NULL;
  }
 
  // Converting UTF-8 to UTF-16
  int32_t utf16_len = 0;
  u_strFromUTF8(NULL, 0, &utf16_len, input_text, -1, &status);
  if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
    fprintf(stderr, "UTF-8 to UTF-16 sizing failed: %s\n", u_errorName(status));
    return NULL;
  }
  status = U_ZERO_ERROR;

  UChar* utf16 = (UChar*)malloc((utf16_len + 1) * sizeof(UChar));
  u_strFromUTF8(utf16, utf16_len + 1, NULL, input_text, -1, &status);
  if (U_FAILURE(status)) {
    fprintf(stderr, "UTF-8 to UTF-16 conversion failed: %s\n", u_errorName(status));
    free(utf16);
    return NULL;
  }

  // Normalizing in UTF-16
  int32_t norm16_len = unorm2_normalize(normalizer, utf16, utf16_len, NULL, 0, &status);
  if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
    fprintf(stderr, "Normalization sizing failed: %s\n", u_errorName(status));
    free(utf16);
    return NULL;
  }
  status = U_ZERO_ERROR;

  UChar* norm16 = (UChar*)malloc((norm16_len + 1) * sizeof(UChar));
  unorm2_normalize(normalizer, utf16, utf16_len, norm16, norm16_len + 1, &status);
  free(utf16);
  if (U_FAILURE(status)) {
    fprintf(stderr, "Normalization failed: %s\n", u_errorName(status));
    free(norm16);
    return NULL;
  }

  // Converting UTF-16 back to UTF-8
  int32_t norm8_len = 0;
  u_strToUTF8(NULL, 0, &norm8_len, norm16, norm16_len, &status);
  if (status != U_BUFFER_OVERFLOW_ERROR && U_FAILURE(status)) {
    fprintf(stderr, "UTF-16 to UTF-8 sizing failed: %s\n", u_errorName(status));
    free(norm16);
    return NULL;
  }
  status = U_ZERO_ERROR;

  char* norm8 = (char*)malloc(norm8_len + 1);
  u_strToUTF8(norm8, norm8_len + 1, NULL, norm16, norm16_len, &status);
  free(norm16);
  if (U_FAILURE(status)) {
    fprintf(stderr, "UTF-16 to UTF-8 conversion failed: %s\n", u_errorName(status));
    free(norm8);
    return NULL;
  }

  // replacing all the ASCII spaces with U+2581 (UTF-8: E2 96 81)
  const char* rep = "\xE2\x96\x81";
  size_t rep_len = 3, final_len = 0;

  for (int i = 0; i < norm8_len; i++)
    final_len += (norm8[i] == ' ') ? rep_len : 1;

  char* final = (char*)malloc(final_len + 1);
  size_t j = 0;
  for (int i = 0; i < norm8_len; i++) {
    if (norm8[i] == ' ') {
      memcpy(&final[j], rep, rep_len);
      j += rep_len;
    } else {
      final[j++] = norm8[i];
    }
  }
  final[j] = '\0';
  free(norm8);

  return final;
}

// helper function for saving new vocabs in tries, recursively
static void save_vocab_rec(TrieNode* node, FILE* out, unsigned char* buf, int depth) {
  if (node->terminal) {
    buf[depth] = '\0';
    fprintf(out, "%s\n", buf);
  }
  for (int c = 0; c < NUM_CHARS; c++) {
    if (node->children[c]) {
      buf[depth] = (unsigned char)c;
      save_vocab_rec(node->children[c], out, buf, depth + 1);
    }
  }
}

void save_vocab(TrieNode* root, const char* vocab_file) {
  FILE* out = fopen(vocab_file, "w");
  if (!out) { perror("fopen(vocab_file)"); return; }
  unsigned char* buffer = (unsigned char*)malloc(1024);
  save_vocab_rec(root, out, buffer, 0);
  free(buffer);
  fclose(out);
}

TrieNode* load_vocab(const char* vocab_file) {
  TrieNode* root = create_node();
  FILE* in = fopen(vocab_file, "r");
  if (!in) { perror("fopen(vocab_file)"); return root; }
  char line[1024];
  while (fgets(line, sizeof(line), in)) {
    line[strcspn(line, "\r\n")] = '\0';
    if (line[0]) trie_insert(root, line);
  }
  fclose(in);
  return root;
}