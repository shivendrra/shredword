from tqdm import tqdm
from .base import BaseTokenizer, get_stats, merge
from functools import lru_cache, wraps

def dynamic_lru_cache(maxsize_func):
  """
    decorator to create a dynamic LRU cache with adjustable max size
    maxsize_func: function that calculates the max size based on input
  """
  def decorator(func):
    # wraping the function with lru_cache, but without an initial maxsize
    @lru_cache(maxsize=None)
    @wraps(func)
    def wrapper(self, *args, **kwargs):
      # dynamically setting cache size based on the function input
      new_maxsize = maxsize_func()
      if wrapper.cache_info().maxsize != new_maxsize:
        # reapplying cache with updated maxsize
        wrapper.cache_clear()
        wrapper = lru_cache(maxsize=new_maxsize)(func)
      return wrapper(self, *args, **kwargs)
    return wrapper
  return decorator


# dynamic_cache_size = 2048
# @dynamic_lru_cache(lambda: dynamic_cache_size)

lru_cache(20480)
def encode_subsequence(sub_ids, merges_tuple):
  merges = dict(merges_tuple) # convert tuple back to dict
  while len(sub_ids) >= 2:
    stats = get_stats(sub_ids)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    if pair not in merges:
      break
    sub_ids = merge(sub_ids, pair, merges[pair])
  return sub_ids

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
    ids, merges, vocab = list(text_bytes), {}, {idx: bytes([idx]) for idx in range(256)}
    for i in tqdm(range(num_merges), "training the tokenizer:\t"):
      stats = get_stats(ids)
      if not stats:
        break
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
    """
      ```def encode(self, text):
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
      ```
      previous implementation of encode function didn't use the LRU caching techniques, hence
      slower encoding rate
    """
    text_bytes = tuple(text.encode("utf-8"))  # tuples are hashable for caching
    merges_tuple = tuple(self.merges.items())  # convert dict to a hashable tuple
    ids = encode_subsequence(text_bytes, merges_tuple)
    return list(ids)