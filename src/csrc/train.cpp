#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "train.h"

static void save_vocab_recursively(TrieNode* node, FILE* out, unsigned char* buffer, int depth) {
  if (node->terminal) {
    buffer[depth] = '\0';
    fprintf(out, "%s\n", buffer);
  }
  for (int c = 0; c < NUM_CHARS; c++) {
    if (node->children[c]) {
    buffer[depth] = (unsigned char)c;
    save_vocab_recursively(node->children[c], out, buffer, depth + 1);
    }
  }
}

/**
  @brief Train new vocabulary from raw text file.
  * Reads train_file, normalizes lines, splits on ▁, builds trie,
  * then saves to vocab_file (overwrite or create).
*/
void train_vocab(const char* train_file, const char* vocab_file) {
  // loading existing vocab or start fresh
  TrieNode* root;
  FILE* test = fopen(vocab_file, "r");
  if (test) { fclose(test); root = load_vocab(vocab_file); }
  else { root = create_node(); }

  // opening training data
  FILE* fin = fopen(train_file, "r");
  if (!fin) { perror("fopen(train_file)"); return; }

  char line[4096];
  char token[4096];
  while (fgets(line, sizeof(line), fin)) {
    line[strcspn(line, "\r\n")] = '\0';
    if (!line[0]) continue;

    char* norm = normalize_text(line);
    if (!norm) continue;

    size_t L = strlen(norm), i = 0;
    while (i < L) {
      // skip leading ▁ (0xE2 0x96 0x81)
      if (i + 2 < L && (unsigned char)norm[i]==0xE2 && (unsigned char)norm[i+1]==0x96 && (unsigned char)norm[i+2]==0x81) {
        i += 3;
        continue;
      }
      size_t start = i;
      while (i < L && !(i+2< L && (unsigned char)norm[i]==0xE2 && (unsigned char)norm[i+1]==0x96 && (unsigned char)norm[i+2]==0x81)) {
        i++;
      }
      size_t len = i - start;
      if (len > 0 && len < sizeof(token)) {
        memcpy(token, norm + start, len);
        token[len] = '\0';
        trie_insert(root, token);
      }
    }
    free(norm);
  }
  fclose(fin);

  // save updated vocab
  save_vocab(root, vocab_file);
  free_trie(root);
}