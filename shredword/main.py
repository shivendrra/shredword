from tqdm import tqdm
from .base import BaseTokenizer, get_stats, merge

class shred(BaseTokenizer):
  """
    main tokenizer class, inherited from base class
    - trains & builds the vocab by merging max occuring pair, one at a time
    - removes the special characters from the corpus, separate token is assigned to
      each of them.
    - encoding can be made more efficient by implementing LRU caching properly (not included in this though!)
    - load/save vocab functions are pre-written in the main ``base``class.
    
    # regex implementation not included yet, working on it....
  """
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