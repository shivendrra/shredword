import ctypes
from typing import *
from .cbase import lib, BPEConfig

class BPETrainer:
  def __init__(self, target_vocab_size=8192, unk_id=0, character_coverage=0.995, min_pair_freq=2000):
    self.config = BPEConfig(
      target_vocab_size=target_vocab_size,
      unk_id=unk_id,
      character_coverage=character_coverage,
      min_pair_freq=min_pair_freq
    )
    self.trainer = lib.create_trainer(ctypes.byref(self.config))
    if not self.trainer:
      raise RuntimeError("Failed to create BPE trainer")

  def load_corpus(self, path: str):
    result = lib.bpe_load_corpus(self.trainer, path.encode('utf-8'))
    if result != 0:
      raise IOError(f"Failed to load corpus from {path}")

  def train(self):
    merges = lib.bpe_train(self.trainer)
    if merges < 0:
      raise RuntimeError("Training failed")
    print(f"Training completed: {merges} merges performed.")

  def save(self, model_path: str, vocab_path: str):
    lib.bpe_save(self.trainer, model_path.encode('utf-8'), vocab_path.encode('utf-8'))
    print(f"Model saved to: {model_path}")
    print(f"Vocabulary saved to: {vocab_path}")

  def destroy(self):
    if self.trainer:
      lib.bpe_trainer_destroy(self.trainer)
      self.trainer = None

  def __del__(self):
    self.destroy()