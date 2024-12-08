from tqdm import tqdm
from .base import BaseTokenizer, get_stats, merge
from functools import lru_cache, wraps
from heapq import heappush, heappop, heapify

def dynamic_lru_cache(maxsize_func):
  """
    decorator to create a dynamic LRU cache with adjustable max size
    maxsize_func: function that calculates the max size based on input
  """
  def decorator(func):
    cached_func = lru_cache(maxsize=maxsize_func())(func)
    # wraping the function with lru_cache, but without an initial maxsize
    @wraps(func)
    def wrapper(*args, **kwargs):
      nonlocal cached_func 
      # dynamically setting cache size based on the function input
      new_maxsize = maxsize_func()
      if cached_func.cache_info().maxsize != new_maxsize:
        # reapplying cache with updated maxsize
        cached_func.cache_clear()
        cached_func = lru_cache(maxsize=new_maxsize)(func)
      return cached_func(*args, **kwargs)
    return wrapper
  return decorator

dynamic_cache_size = 20480
@dynamic_lru_cache(lambda: dynamic_cache_size)
def encode_subsequence(sub_ids, merges_tuple):
  """
    optimized subsequence encoding with efficient caching and merging logic
    - uses a priority queue to merge pairs efficiently
  """
  merges = dict(merges_tuple)  # convert tuple back to dictionary
  stats = get_stats(sub_ids)  # initial stats for the whole subsequence
  # priority queue for merging (negative counts for max heap behavior)
  heap = [(-count, pair) for pair, count in stats.items()]
  heapify(heap)
  while heap:
    _, pair = heappop(heap)
    if pair not in merges:
      continue  # skipping pairs that are no longer valid
    # merge the pair and update `sub_ids`
    sub_ids = merge(sub_ids, pair, merges[pair])
    # recalculate stats for affected region (only for efficiency)
    new_stats = get_stats(sub_ids)
    heap = [(-count, p) for p, count in new_stats.items()]
    heapify(heap)
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

  # def encode(self, text):
  #   """
  #     ```def encode(self, text):
  #         text_bytes = text.encode("utf-8")
  #         ids = list(text_bytes)
  #         while len(ids) >= 2:
  #           stats = get_stats(ids)
  #           pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
  #           if pair not in self.merges:
  #             break
  #           idx = self.merges[pair]
  #           ids = merge(ids, pair, idx)
  #         return ids
  #     ```
  #     previous implementation of encode function didn't use the LRU caching techniques, hence
  #     slower encoding rate
  #   """
  #   text_bytes = tuple(text.encode("utf-8"))  # tuples are hashable for caching
  #   merges_tuple = tuple(self.merges.items())  # convert dict to a hashable tuple
  #   ids = encode_subsequence(text_bytes, merges_tuple)
  #   return list(ids)

  def encode(self, text):
    ## encodes input text using byte-pair encoding with LRU-cached optimization
    ## splits input into smaller chunks for better cache utilization
    chunk_size = 1024
    text_bytes = text.encode("utf-8")
    
    ids = []
    for i in range(0, len(text_bytes), chunk_size):
      chunk = tuple(text_bytes[i:i + chunk_size])
      merges_tuple = tuple(self.merges.items())
      ids.extend(encode_subsequence(chunk, merges_tuple))
    return ids