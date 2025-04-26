#!/usr/bin/env python3
"""
training_test.py
- Reads a training file, trains the tokenizer using a simple training loop,
- and prints a report with elapsed time, peak memory usage, final vocab size, and merge count.
- Usage: python training_test.py
"""

import time, tracemalloc
from shred.main import Shred

def read_file(filename):
  with open(filename, "r", encoding="utf-8") as f:
    return f.read()

def main():
  train_file = "train.txt"  # your training file
  text = read_file(train_file)
  
  tokenizer = Shred()
  target_vocab_size = 3000  # target vocab size (e.g., 3000 tokens)

  print("Starting training...")
  tracemalloc.start()
  start_time = time.time()
  
  # Train the tokenizer; verbose=True will print each merge step.
  final_ids = tokenizer.train(text, target_vocab_size, verbose=True)
  
  end_time = time.time()
  current, peak = tracemalloc.get_traced_memory()
  tracemalloc.stop()
  
  elapsed = end_time - start_time
  final_vocab_size = len(tokenizer.vocab)
  merge_count = len(tokenizer.merges)
  
  print("==== Training Report ====")
  print(f"Training File: {train_file}")
  print(f"Elapsed Time: {elapsed:.2f} seconds")
  print(f"Peak Memory Usage: {peak/1024:.2f} KB")
  print(f"Final Vocab Size: {final_vocab_size}")
  print(f"Number of Merges: {merge_count}")
  print("=========================")

if __name__ == "__main__":
  main()