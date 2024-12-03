import unicodedata
from tqdm import tqdm

def get_stats(ids, counts=None):
  """
    given a list of integers, return a dictionary of counts of consecutive pairs
    eg: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    optionally allows to update an existing dictionary of counts
  """
  counts = {} if counts is None else counts
  for pair in zip(ids, ids[1:]):
    counts[pair] = counts.get(pair, 0) + 1
  return counts

def merge(ids, pair, idx):
  """
    in the list of integers (ids), replace all consecutive occurrences
    of pair with new integer token idx
    eg: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
  """
  newids = []
  i = 0
  while i < len(ids):
    if ids[i] == pair[0] and i < len(ids) - 1 and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

def replace_control_characters(s: str) -> str:
  # we don't want to print control characters
  # which distort the output (e.g. \n)
  chars = []
  for ch in s:
    if unicodedata.category(ch)[0] != "C":
      chars.append(ch) # this character is ok
    else:
      chars.append(f"\\u{ord(ch):04x}") # escape
  return "".join(chars)

def render_token(t: bytes) -> str:
  # pretty print a token, escaping control characters
  s = t.decode('utf-8', errors='replace')
  s = replace_control_characters(s)
  return s

class BaseTokenizer:
  def __init__(self):
    # default: vocab size of 256 (all bytes), no merges, no patterns
    self.merges = {} # (int, int) -> int
    self.pattern = ""
    self.special_tokens = {} # str -> int, e.g. {'<|endoftext|>': 100257}
    self.vocab = self._build_vocab() # int -> bytes

  def train(self, text, vocab_size, verbose=False): raise NotImplementedError
  def encode(self, text): raise NotImplementedError
  def decode(self, ids): raise NotImplementedError

  def _build_vocab(self):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in self.merges.items():
      vocab[idx] = vocab[p0] + vocab[p1]
    for special, idx in self.special_tokens.items():
      vocab[idx] = special.encode("utf-8")
    return vocab

  def save(self, file_prefix):
    model_file = file_prefix + ".model"
    with open(model_file, 'w') as f:
      f.write("shredword v1.0\n")
      f.write(f"{self.pattern}\n")
      f.write(f"{len(self.special_tokens)}\n")
      for special, idx in self.special_tokens.items():
        f.write(f"{special} {idx}\n")
      for idx1, idx2 in self.merges:
        f.write(f"{idx1} {idx2}\n")
    vocab_file, inverted_merges = file_prefix + ".vocab", {idx: pair for pair, idx in self.merges.items()}
    with open(vocab_file, "w", encoding="utf-8") as f:
      for idx, token in self.vocab.items():
        s = render_token(token)
        if idx in inverted_merges:
          idx0, idx1 = inverted_merges[idx]
          s0, s1 = render_token(self.vocab[idx0]), render_token(self.vocab[idx1])
          f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
        else: f.write(f"[{s}] {idx}\n")
      f.close()

  def load(self, model_file):
    assert model_file.endswith(".model")
    merges, special_tokens, idx = {}, {}, 256
    with open(model_file, 'r', encoding="utf-8") as f:
      version = f.readline().strip()
      assert version == "minbpe v1"
      self.pattern, num_special = f.readline().strip(), int(f.readline().strip())
      for _ in range(num_special):
        special, special_idx = f.readline().strip().split()
        special_tokens[special] = int(special_idx)
      for line in f:
        idx1, idx2 = map(int, line.split())
        merges[(idx1, idx2)] = idx
        idx += 1
    self.merges, self.special_tokens, self.vocab = merges, special_tokens, self._build_vocab()

class shredword(BaseTokenizer):
  def __init__(self): super().__init__()
  def train(self, text, vocab_size, verbose=False):
    assert vocab_size >= 256
    num_merges = vocab_size - 256
    text_bytes = text.encode("utf-8")
    ids = list(text_bytes)
    merges = {}
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for i in tqdm(range(num_merges), "training the tokenizer:\t"):
      stats = get_stats(ids)
      pair, idx = max(stats, key=stats.get), 256 + i
      ids, merges[pair] = merge(ids, pair, idx), idx
      vocab[idx] = vocab[pair[0]] + vocab[pair[1]]
      if verbose:
        print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")
    self.merges, self.vocab = merges, vocab
  
  def decode(self, ids):
    text_bytes = b"".join(self.vocab[idx] for idx in ids)
    text = text_bytes.decode("utf-8", errors="replace")
    return text
  
  def encode(self, text):
    text_bytes = text.encode("utf-8")
    ids = list(text_bytes)
    while len(ids) >= 2:
      stats = get_stats(ids)
      pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
      if pair not in self.merges:
        break
      idx = self.merges[pair]
      ids = merge(ids, pair, idx)
    return ids