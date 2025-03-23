#!/usr/bin/env python3
"""
encode_decode_test.py
- Reads a test file, trains (or loads) the tokenizer model,
- Encodes the text, decodes it back, and reports metrics such as encoding/decoding times,
- compression ratio, peak memory usage, and whether the decoded text matches the original.
- Usage: python encode_decode_test.py
"""

import time, tracemalloc
from src.main import Shred


def read_file(filename):
  with open(filename, "r", encoding="utf-8") as f:
    return f.read()

def main():
  test_file = "test.txt"  # your test file
  text = read_file(test_file)
  
  tokenizer = Shred()
  # For testing, we train on the test text; alternatively, load a pre-trained model.
  tokenizer.train(text, 3000, verbose=False)
  
  print("Starting encoding/decoding test...")
  tracemalloc.start()
  start_encode = time.time()
  
  encoded_ids = tokenizer.encode(text)
  
  mid_time = time.time()
  
  decoded_text = tokenizer.decode(encoded_ids)
  
  end_time = time.time()
  current, peak = tracemalloc.get_traced_memory()
  tracemalloc.stop()
  
  encoding_time = mid_time - start_encode
  decoding_time = end_time - mid_time
  total_time = end_time - start_encode
  original_size = len(text.encode("utf-8"))
  token_count = len(encoded_ids)
  compression_ratio = original_size / token_count if token_count > 0 else 0
  
  print("==== Encode/Decode Report ====")
  print(f"Test File: {test_file}")
  print(f"Original Size: {original_size} bytes")
  print(f"Encoded Token Count: {token_count}")
  print(f"Compression Ratio (bytes per token): {compression_ratio:.2f}")
  print(f"Encoding Time: {encoding_time:.2f} seconds")
  print(f"Decoding Time: {decoding_time:.2f} seconds")
  print(f"Total Time: {total_time:.2f} seconds")
  print(f"Peak Memory Usage: {peak/1024:.2f} KB")
  if text == decoded_text:
    print("Decoded text matches original.")
  else:
    print("Decoded text does NOT match original.")
  print("===============================")

if __name__ == "__main__":
  main()
