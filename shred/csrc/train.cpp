#include <cstring>
#include <cstdlib>
#include <cstdio>
#include "train.h"
#include "base.h"

static int trie_count_words(TrieNode* node) {
  if (!node) return 0;
  int count = node->terminal ? 1 : 0;
  for (int i = 0; i < NUM_CHARS; ++i) {
    count += trie_count_words(node->children[i]);
  }
  return count;
}

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
  @brief Train new vocabulary from a pre-normalized text file.
  * Each line should already be normalized and use ▁ (U+2581) to indicate word boundaries.
  * Extracts tokens and builds a trie-based vocabulary, which is then saved.

   * @param train_file Path to the training text file (pre-normalized).
   * @param vocab_file Path to the output vocabulary file.
   * @param vocab_limit Maximum number of unique tokens to extract.
*/
// void train_vocab(const char* train_file, const char* vocab_file, int vocab_limit) {
//   TrieNode* root = create_node();
//   FILE* test = fopen(vocab_file, "r");
//   if (test) { fclose(test); load_vocab(root, vocab_file); }

//   FILE* fin = fopen(train_file, "r");
//   if (!fin) { perror("fopen(train_file)"); return; }

//   char line[4096];
//   char token[4096];
//   int step = 0;

//   while (fgets(line, sizeof(line), fin)) {
//     line[strcspn(line, "\r\n")] = '\0';
//     if (!line[0]) continue;

//     const char* norm = line;  // already normalized
//     size_t L = strlen(norm), i = 0;
//     while (i < L) {
//       if (i + 2 < L && (unsigned char)norm[i]==0xE2 && (unsigned char)norm[i+1]==0x96 && (unsigned char)norm[i+2]==0x81) {
//         i += 3;
//         continue;
//       }
//       size_t start = i;
//       while (i < L && !(i+2 < L && (unsigned char)norm[i]==0xE2 && (unsigned char)norm[i+1]==0x96 && (unsigned char)norm[i+2]==0x81)) {
//         i++;
//       }
//       size_t len = i - start;
//       if (len > 0 && len < sizeof(token)) {
//         memcpy(token, norm + start, len);
//         token[len] = '\0';
//         trie_insert(root, token);

//         int current_vocab_size = trie_count_words(root);
//         step++;
//         if (step % 500 == 0 || current_vocab_size == vocab_limit) {
//           printf("[step %d] Current vocab size: %d\n", step, current_vocab_size);
//         }
//         if (current_vocab_size >= vocab_limit) {
//           printf("Reached vocab limit of %d tokens.\n", vocab_limit);
//           goto end_training;
//         }
//       }
//     }
//   }

// end_training:
//   fclose(fin);
//   save_vocab(root, vocab_file);
//   free_trie(root);
// }

// void train_vocab(const char* train_file, const char* vocab_file, int merge_steps) {
//   printf("[DEBUG] train_vocab(): \"%s\" -> \"%s\" (%d steps)\n", train_file, vocab_file, merge_steps);

//   TrieNode* root = create_node();
//   printf("[DEBUG] Starting BPE learning...\n");
//   bpe_learn(root, merge_steps, train_file);
//   printf("[DEBUG] BPE learning done, saving vocab...\n");
//   save_vocab(root, vocab_file);
//   printf("[DEBUG] Vocab saved, freeing trie...\n");
//   free_trie(root);
//   printf("[DEBUG] train_vocab() complete\n");
// }

void train_vocab(const char* train_file,
  const char* vocab_file,
  int merge_steps)
{
  printf("[DEBUG] train_vocab(): \"%s\" -> \"%s\" (%d steps)\n",
    train_file, vocab_file, merge_steps);

  // 1) Create the trie root
  printf("[DEBUG] Creating trie root...\n");
  TrieNode* root = create_node();

  // 2) Run BPE learning
  printf("[DEBUG] Starting BPE learning...\n");
  bpe_learn(train_file, merge_steps, root);
  printf("[DEBUG] BPE learning done.\n");

  // 3) Save the resulting vocabulary
  printf("[DEBUG] Saving vocab to \"%s\"...\n", vocab_file);
  save_vocab(root, vocab_file);
  printf("[DEBUG] Vocab saved.\n");

  // 4) Clean up
  printf("[DEBUG] Freeing trie...\n");
  free_trie(root);
  printf("[DEBUG] train_vocab() complete.\n");
}