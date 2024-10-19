import collections
import re
import os
from tqdm import tqdm

# Ensure the script runs in the current directory
current_dir = os.path.abspath(os.path.dirname(__file__))
os.chdir(current_dir)

class Tokenizer:
  def __init__(self, vocab=None, special_tokens=None):
    self.special_tokens = special_tokens if special_tokens else {
      '<unk>': 0,
      '<pad>': 1,
      '<bos>': 2,
      '<eos>': 3,
    }
    self.vocab = vocab if vocab else {}
    self.vocab.update(self.special_tokens)  # Include special tokens in vocab
    self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}
    self.cache = {}

    # Ensure the first 256 UTF-8 characters are included in the vocab
    for i in range(256):
      self.vocab.setdefault(chr(i), len(self.vocab))

    self.reverse_vocab = {idx: word for word, idx in self.vocab.items()}

  @staticmethod
  def from_file(filename):
    vocab = {}
    with open(filename, 'r', encoding='utf-8') as file:  # Specify UTF-8 encoding
      for index, line in enumerate(file):
        word = line.strip()
        vocab[word] = index
    return Tokenizer(vocab)

  def to_file(self, filename):
    with open(filename, 'w', encoding='utf-8') as file:  # Specify UTF-8 encoding
      for word, index in sorted(self.vocab.items(), key=lambda item: item[1]):
        file.write(f"{word}\n")

  def encode(self, text):
    if text in self.cache:
      return self.cache[text]
    encoded = [self.vocab.get(word, self.vocab['<unk>']) for word in text.split()]
    self.cache[text] = encoded
    return encoded

  def decode(self, tokens):
    if tuple(tokens) in self.cache:
      return self.cache[tuple(tokens)]
    decoded = " ".join([self.reverse_vocab.get(token, '<unk>') for token in tokens])
    self.cache[tuple(tokens)] = decoded
    return decoded

  def clear_cache(self):
    self.cache.clear()

  def cache_size(self):
    return len(self.cache)

  def train(self, data, vocab_size=10000):
    """Train the tokenizer using a dataset."""
    token_counts = collections.Counter()

    # Progress bar for training
    for text in tqdm(data, desc="Training Tokenizer", unit="text"):
      tokens = text.split()
      token_counts.update(tokens)

    # Create vocabulary based on the most common tokens
    most_common_tokens = token_counts.most_common(vocab_size - len(self.special_tokens))  # Leave space for special tokens
    self.vocab.update({token: len(self.vocab) for token, _ in most_common_tokens})
    self.reverse_vocab = {idx: token for token, idx in self.vocab.items()}

  def encode_bpe(self, text):
    """Encode text using Byte Pair Encoding (BPE)."""
    tokens = self._bpe_tokenize(text)
    return self.encode(tokens)

  def decode_bpe(self, tokens):
    """Decode BPE-encoded tokens."""
    return self.decode(tokens)

  def _bpe_tokenize(self, text):
    """Tokenizes the text using BPE (placeholder)."""
    text = re.sub(r'\s+', ' ', text).strip()
    return text.split()  # Placeholder for actual BPE logic

  def save_model(self, path):
    """Save the tokenizer model to a file."""
    with open(path, 'w', encoding='utf-8') as file:  # Specify UTF-8 encoding
      for word, index in self.vocab.items():
        file.write(f"{word}\t{index}\n")

  @staticmethod
  def load_model(path):
    """Load the tokenizer model from a file."""
    vocab = {}
    with open(path, 'r', encoding='utf-8') as file:  # Specify UTF-8 encoding
      for line in file:
        if line.strip():  # Only process non-empty lines
          try:
            word, index = line.strip().split('\t')
            vocab[word] = int(index)
          except ValueError:
            print(f"Warning: Skipping line due to format error: {line.strip()}")
    return Tokenizer(vocab)

if __name__ == "__main__":
  # Load data for training
  with open("./captions.txt", 'r', encoding="utf-8") as f:
    data = f.read().strip().splitlines()  # Split data into lines
  print("file closed")

  # Create and train tokenizer
  tokenizer = Tokenizer()
  tokenizer.train(data, vocab_size=2000)  # Set vocabulary size to 2000

  # Example usage of encoding and decoding
  encoded = tokenizer.encode("hello world")
  print(f"Encoded: {encoded}")
  decoded = tokenizer.decode(encoded)
  print(f"Decoded: {decoded}")

  # Save and load model
  tokenizer.save_model("tokenizer_model.txt")
  new_tokenizer = Tokenizer.load_model("tokenizer_model.txt")
  print(f"Loaded Vocab: {new_tokenizer.vocab}")

  with open("captions.txt", "r", encoding="utf-8") as f:
    test_data = f.read()
  
  encoded = tokenizer.encode(test_data)
  print(f"Encoded: {encoded}")
  decoded = tokenizer.decode(encoded)
  print(f"Decoded: {decoded}")

  # Clear cache
  tokenizer.clear_cache()
  print(f"Cache Size After Clear: {tokenizer.cache_size()}")