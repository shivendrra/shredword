import heapq, collections, hashlib
from typing import Dict, List, Tuple, Set, Optional

class TokenFreqHeap:
  def __init__(self):
    self.heap, self.token_to_freq, self.removed = [], {}, set()
  
  def push(self, token: str, freq: int):
    if token in self.removed: self.removed.discard(token)
    entry = (freq, token)
    heapq.heappush(self.heap, entry)
    self.token_to_freq[token] = freq
  
  def pop(self) -> Tuple[int, str]:
    while self.heap:
      freq, token = heapq.heappop(self.heap)
      if token not in self.removed and self.token_to_freq.get(token) == freq:
        del self.token_to_freq[token]
        return freq, token
    raise IndexError("pop from empty heap")
  
  def remove(self, token: str):
    if token in self.token_to_freq:
      self.removed.add(token)
      del self.token_to_freq[token]
  
  def update_freq(self, token: str, new_freq: int):
    if token in self.token_to_freq:
      self.removed.add(token)
    self.push(token, new_freq)
  
  def __len__(self): return len(self.token_to_freq)
  def __contains__(self, token): return token in self.token_to_freq

class LRUCache:
  def __init__(self, capacity: int = 10000):
    self.capacity, self.cache = capacity, collections.OrderedDict()
  
  def get(self, key):
    if key in self.cache:
      self.cache.move_to_end(key)
      return self.cache[key]
    return None
  
  def put(self, key, value):
    if key in self.cache: self.cache.move_to_end(key)
    else:
      if len(self.cache) >= self.capacity: self.cache.popitem(last=False)
    self.cache[key] = value

class TrieNode:
  def __init__(self):
    self.children, self.is_token, self.freq = {}, False, 0

class SubwordTrie:
  def __init__(self):
    self.root = TrieNode()
  
  def insert(self, token: str, freq: int = 1):
    node = self.root
    for char in token:
      if char not in node.children: node.children[char] = TrieNode()
      node = node.children[char]
    node.is_token, node.freq = True, freq
  
  def search(self, token: str) -> Optional[int]:
    node = self.root
    for char in token:
      if char not in node.children: return None
      node = node.children[char]
    return node.freq if node.is_token else None
  
  def get_all_tokens(self) -> List[Tuple[str, int]]:
    result = []
    def dfs(node, prefix):
      if node.is_token: result.append((prefix, node.freq))
      for char, child in node.children.items(): dfs(child, prefix + char)
    dfs(self.root, "")
    return result

class FastHashMap:
  def __init__(self, initial_size: int = 16384):
    self.size, self.count = initial_size, 0
    self.buckets = [[] for _ in range(self.size)]
  
  def _hash(self, key: str) -> int:
    return int(hashlib.md5(key.encode()).hexdigest()[:8], 16) % self.size
  
  def _resize(self):
    old_buckets = self.buckets
    self.size *= 2
    self.buckets = [[] for _ in range(self.size)]
    self.count = 0
    for bucket in old_buckets:
      for k, v in bucket: self[k] = v
  
  def __setitem__(self, key: str, value):
    if self.count >= self.size * 0.75: self._resize()
    idx = self._hash(key)
    for i, (k, v) in enumerate(self.buckets[idx]):
      if k == key:
        self.buckets[idx][i] = (key, value)
        return
    self.buckets[idx].append((key, value))
    self.count += 1
  
  def __getitem__(self, key: str):
    idx = self._hash(key)
    for k, v in self.buckets[idx]:
      if k == key: return v
    raise KeyError(key)
  
  def __contains__(self, key: str) -> bool:
    idx = self._hash(key)
    return any(k == key for k, v in self.buckets[idx])
  
  def get(self, key: str, default=None):
    try: return self[key]
    except KeyError: return default
  
  def items(self):
    for bucket in self.buckets:
      for k, v in bucket: yield k, v

  def __delitem__(self, key: str):
    idx = self._hash(key)
    for i, (k, v) in enumerate(self.buckets[idx]):
      if k == key:
        del self.buckets[idx][i]
        self.count -= 1
        return
    raise KeyError(key)

class SubwordExtractor:
  def __init__(self): self.cache = LRUCache(50000)
  def extract_subwords(self, text: str, max_len: int = 20) -> Set[str]:
    cache_key = f"{hash(text)}_{max_len}"
    cached = self.cache.get(cache_key)
    if cached: return cached

    subwords = set()
    for i in range(len(text)):
      for j in range(i + 1, min(i + max_len + 1, len(text) + 1)): subwords.add(text[i:j])
    self.cache.put(cache_key, subwords)
    return subwords

  def get_char_frequencies(self, texts: List[str]) -> Dict[str, int]:
    char_freq = collections.defaultdict(int)
    for text in texts:
      for char in text: char_freq[char] += 1
    return dict(char_freq)

class ViterbiDecoder:
  def __init__(self):
    self.cache = LRUCache(20000)
  
  def decode(self, text: str, vocab: Dict[str, float]) -> List[str]:
    cache_key = hash(text)
    cached = self.cache.get(cache_key)
    if cached: return cached
    
    n = len(text)
    dp = [-float('inf')] * (n + 1)
    parent = [-1] * (n + 1)
    dp[0] = 0.0
    
    for i in range(n):
      if dp[i] == -float('inf'): continue
      for j in range(i + 1, min(i + 21, n + 1)):
        token = text[i:j]
        if token in vocab:
          score = dp[i] + vocab[token]
          if score > dp[j]:
            dp[j], parent[j] = score, i
    
    if dp[n] == -float('inf'): return [text]
    
    path = []
    pos = n
    while pos > 0:
      start = parent[pos]
      path.append(text[start:pos])
      pos = start
    
    result = path[::-1]
    self.cache.put(cache_key, result)
    return result