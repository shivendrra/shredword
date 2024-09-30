import regex as re
import unicodedata
from collections import defaultdict
from tqdm import tqdm

class Tokenizer:
  def __init__(self):
    self.merges = {}
    self.vocab = {idx: bytes([idx]) for idx in range(256)}
    self.pattern = ""
    self.special_tokens = {}
    self.cache = defaultdict(int)  # Cache for pair occurrences

  def get_stats(self, ids):
    """
      Takes list of integers and returns dictionary of counts of pairs (consecutive ones).
    """
    counts = defaultdict(int)  # Use defaultdict to avoid manual checks
    for pair in zip(ids, ids[1:]):
      counts[pair] += 1
    return counts

  def merge(self, ids, pair, idx):
    """
      In the list of integers, replaces all consecutive pairs with the new integer token idx.
    """
    new_ids = []
    i = 0
    while i < len(ids):
      if i + 1 < len(ids) and ids[i] == pair[0] and ids[i + 1] == pair[1]:
        new_ids.append(idx)
        i += 2
      else:
        new_ids.append(ids[i])
        i += 1
    return new_ids

  def apply_regex(self, text):
    r"""
      Regex patterns for tokenization.
    """
    self.pattern = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
    text = re.findall(self.pattern, text)
    return text

  def build_vocab(self):
    """
      Build the vocabulary using merges and special tokens.
    """
    self.vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in self.merges.items():
      self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
    
    for special, idx in self.special_tokens.items():
      self.vocab[idx] = special.encode("utf-8")

  def replace_control_characters(self, s: str) -> str:
    chars = []
    for ch in s:
      if unicodedata.category(ch)[0] != "C":
        chars.append(ch)
      else:
        chars.append(f"\\u{ord(ch):04x}")
    return "".join(chars)

  def render_token(self, t: bytes) -> str:
    s = t.decode('utf-8', errors='replace')
    s = self.replace_control_characters(s)
    return s

  def decode(self, ids):
    text_bytes = b"".join(self.vocab[idx] for idx in ids)
    text = text_bytes.decode("utf-8", errors="replace")
    return text

  def encode(self, text):
    text_bytes = text.encode("utf-8")
    ids = list(text_bytes)
    while len(ids) >= 2:
      stats = self.get_stats(ids)
      pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
      if pair not in self.merges:
        break

      idx = self.merges[pair]
      ids = self.merge(ids, pair, idx)
    return ids

  def train(self, text, vocab_size, verbose=False):
    """
      Training loop for the BPE tokenizer.
    """
    assert vocab_size >= 256

    num_merges = vocab_size - 256
    text_bytes = text.encode("utf-8")
    ids = list(text_bytes)

    # Precompute the initial pair statistics
    stats = self.get_stats(ids)

    for i in tqdm(range(num_merges), desc="Training", unit="merge"):
      # Check if the cache has the pair count
      if not stats:
        break
      
      pair = max(stats, key=stats.get)
      idx = 256 + i
      
      # Merge the pair in ids and update stats
      ids = self.merge(ids, pair, idx)
      self.merges[pair] = idx
      self.vocab[idx] = self.vocab[pair[0]] + self.vocab[pair[1]]

      # Update statistics for the new list of ids
      stats = self.get_stats(ids)
      
      if verbose:
        print(f"merge {i + 1}/{num_merges}: {pair} -> {idx} ({self.vocab[idx]}) had {stats[pair]} occurrences")

  def save_model(self, file_name):
    """
      Saves the model and vocabulary to files.
    """
    model_file = file_name + ".model"
    with open(model_file, 'w', encoding='utf-8') as f:
      f.write("minibpe v1\n")
      f.write(f"{self.pattern}\n")
      f.write(f"{len(self.special_tokens)}\n")
      
      for special, idx in self.special_tokens.items():
        f.write(f"{special} {idx}\n")
      
      for (idx1, idx2), idx in self.merges.items():
        f.write(f"{idx1} {idx2}\n")

    vocab_file = file_name + ".vocab"
    with open(vocab_file, 'w', encoding='utf-8') as f:
      for idx, token in self.vocab.items():
        s = self.render_token(token)
        f.write(f"[{s}] {idx}\n")

  def load(self, model_file):
    """
      Loads '.model' file and returns the vocab loaded from the file.
    """
    assert model_file.endswith('.model')
    merges = {}
    special_tokens = {}
    idx = 256
    with open(model_file, 'r', encoding='utf-8') as f:
      version = f.readline().strip()
      assert version == "minibpe v1"

      self.pattern = f.readline().strip()
      num_special = int(f.readline().strip())
      
      for _ in range(num_special):
        special, special_idx = f.readline().strip().split()
        special_tokens[special] = int(special_idx)
      
      for line in f:
        if line.strip():  # Skip empty lines
          idx1, idx2 = map(int, line.split())
          merges[(idx1, idx2)] = idx
          idx += 1

      self.merges = merges
      self.special_tokens = special_tokens
      self.build_vocab()

    return self.vocab

# Example Usage:
with open("./training_data.txt", "r", encoding="utf-8") as f:
  data = f.read()
print("File closed")

tokenizer = Tokenizer()
tokenizer.train(data, vocab_size=2000)
tokenizer.save_model("tokenizer_model")
tokenizer.load("tokenizer_model.model")