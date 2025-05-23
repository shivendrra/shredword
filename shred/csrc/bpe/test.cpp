#include <stdio.h>
#include <stdlib.h>
#include "bpe.h"

int main() {
  const char* input_path = "train.txt";
  const char* model_path = "base_1k.model";
  const char* vocab_path = "base_1k.vocab";

  // ---- BPE config ----
  BPEConfig config;
  config.target_vocab = 900;  // desired vocab size
  config.unk_id = 0;
  config.character_coverage = 0.95;
  config.min_pair_freq = 100;

  // ---- Create trainer ----
  BpeTrainer* trainer = bpe_trainer_create(&config);
  if (!trainer) {
    fprintf(stderr, "Failed to create trainer.\n");
    return 1;
  }

  // ---- Load pre-normalized corpus ----
  if (bpe_loadCorpus(trainer, input_path) != 0) {
    fprintf(stderr, "Failed to load corpus from: %s\n", input_path);
    bpe_trainer_destroy(trainer);
    return 2;
  }

  // ---- Run BPE training ----
  int result = bpe_train(trainer);
  if (result < 0) {
    fprintf(stderr, "Training failed.\n");
    bpe_trainer_destroy(trainer);
    return 3;
  }

  // ---- Save model and vocab ----
  bpe_save(trainer, model_path, vocab_path);
  // ---- Cleanup ----
  bpe_trainer_destroy(trainer);
  return 0;
}