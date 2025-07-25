import math, random
from typing import List, Dict, Tuple, Set
from collections import defaultdict
from .utils import TokenFreqHeap, LRUCache, SubwordTrie, FastHashMap, SubwordExtractor, ViterbiDecoder

class UnigramTrainer:
  def __init__(self, vocab_size: int = 32000, character_coverage: float = 0.9995, max_sentencepiece_length: int = 16, seed_sentencepiece_size: int = 1000000):
    self.vocab_size = vocab_size
    self.character_coverage, self.max_len = character_coverage, max_sentencepiece_length
    self.seed_size = seed_sentencepiece_size  
    self.vocab_heap = TokenFreqHeap()
    self.token_freqs = FastHashMap()
    self.subword_trie = SubwordTrie()
    self.extractor = SubwordExtractor()
    self.decoder = ViterbiDecoder()
    self.loss_cache = LRUCache(100000)
    self.vocab, self.final_vocab = {}, {}
    self.texts, self.total_chars = [], 0
  
  def _preprocess_texts(self, texts: List[str]) -> List[str]:
    processed, char_counts = [], defaultdict(int)
    for text in texts:
      clean_text = ''.join(c if c.isprintable() and c != ' ' else '▁' for c in text.strip())
      if clean_text:
        processed.append('▁' + clean_text.replace(' ', '▁'))
        for c in clean_text: char_counts[c] += 1
    total_chars = sum(char_counts.values())
    coverage_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    covered, required_chars = 0, set()
    for char, count in coverage_chars:
      covered += count
      required_chars.add(char)
      if covered / total_chars >= self.character_coverage: break
    return [text for text in processed if all(c in required_chars or c == '▁' for c in text)]

  def _initialize_seed_vocab(self, texts: List[str]):
    all_subwords = set()
    char_freq = defaultdict(int)

    for text in texts[:min(len(texts), 10000)]:
      subwords = self.extractor.extract_subwords(text, self.max_len)
      all_subwords.update(subwords)
      for char in text: char_freq[char] += 1
    for char in char_freq: all_subwords.add(char)
    subword_freq = defaultdict(int)
    for text in texts:
      text_subwords = self.extractor.extract_subwords(text, self.max_len)
      for subword in text_subwords.intersection(all_subwords): subword_freq[subword] += 1
    candidates = [(freq, subword) for subword, freq in subword_freq.items() if freq > 1]
    candidates.sort(reverse=True)

    seed_vocab = {}
    for freq, subword in candidates[:self.seed_size]:
      seed_vocab[subword] = math.log(freq)
      self.token_freqs[subword] = freq
      self.vocab_heap.push(subword, freq)
      self.subword_trie.insert(subword, freq)
    self.vocab = seed_vocab
  
  def _compute_loss(self, texts: List[str]) -> float:
    total_loss, total_len = 0.0, 0
    for text in texts:
      cache_key = hash(text)
      cached_loss = self.loss_cache.get(cache_key)
      if cached_loss is not None:
        total_loss += cached_loss
        total_len += len(text)
        continue
      
      segmentation = self.decoder.decode(text, self.vocab)
      text_loss = -sum(self.vocab.get(token, -20.0) for token in segmentation)
      self.loss_cache.put(cache_key, text_loss)
      total_loss += text_loss
      total_len += len(text)
    return total_loss / max(total_len, 1)
  
  def _compute_token_loss(self, token: str, texts: List[str]) -> float:
    temp_vocab = self.vocab.copy()
    if token in temp_vocab: del temp_vocab[token]
    decoder = ViterbiDecoder()
    total_loss = 0.0
    for text in texts:
      if token not in text: continue
      segmentation = decoder.decode(text, temp_vocab)
      total_loss -= sum(temp_vocab.get(t, -20.0) for t in segmentation)
    return total_loss
  
  def _prune_vocab_step(self, texts: List[str], reduction_ratio: float = 0.8):
    if len(self.vocab) <= self.vocab_size: return
    target_size = max(self.vocab_size, int(len(self.vocab) * reduction_ratio))
    tokens_to_remove = len(self.vocab) - target_size
    
    removal_candidates = []
    vocab_items = list(self.vocab.items())
    random.shuffle(vocab_items)
    
    for token, score in vocab_items[:min(len(vocab_items), tokens_to_remove * 3)]:
      if len(token) == 1: continue
      loss_increase = self._compute_token_loss(token, texts[:1000])
      removal_candidates.append((loss_increase, token))
    
    removal_candidates.sort()
    
    for _, token in removal_candidates[:tokens_to_remove]:
      if token in self.vocab:
        del self.vocab[token]
        self.vocab_heap.remove(token)
        if token in self.token_freqs: del self.token_freqs[token]
  
  def _update_token_scores(self, texts: List[str]):
    token_context_freq = defaultdict(int)
    
    for text in texts[:5000]:
      segmentation = self.decoder.decode(text, self.vocab)
      for token in segmentation:
        if token in self.vocab: token_context_freq[token] += 1
    
    total_freq = sum(token_context_freq.values())
    if total_freq == 0: return
    
    for token in self.vocab:
      freq = token_context_freq.get(token, 1)
      new_score = math.log(freq / total_freq) + math.log(total_freq)
      self.vocab[token] = new_score
      
      if token in self.token_freqs:
        self.vocab_heap.update_freq(token, freq)
        self.token_freqs[token] = freq
  
  def train(self, texts: List[str], num_iterations: int = 20) -> Dict[str, float]:
    print(f"Preprocessing {len(texts)} texts...")
    processed_texts = self._preprocess_texts(texts)
    self.texts = processed_texts[:50000]
    
    print(f"Initializing seed vocabulary...")
    self._initialize_seed_vocab(self.texts)
    print(f"Initial vocabulary size: {len(self.vocab)}")
    
    prev_loss = float('inf')
    for iteration in range(num_iterations):
      print(f"Iteration {iteration + 1}/{num_iterations}")
      
      current_loss = self._compute_loss(self.texts[:2000])
      print(f"  Current loss: {current_loss:.4f}")
      
      if abs(prev_loss - current_loss) < 0.001:
        print("  Convergence reached")
        break
      prev_loss = current_loss
      
      self._update_token_scores(self.texts)
      print(f"  Updated token scores")
      
      if len(self.vocab) > self.vocab_size:
        self._prune_vocab_step(self.texts)
        print(f"  Pruned vocabulary to {len(self.vocab)} tokens")
      
      self.loss_cache = LRUCache(100000)
    
    char_tokens = {token: score for token, score in self.vocab.items() if len(token) == 1}
    other_tokens = {token: score for token, score in self.vocab.items() if len(token) > 1}
    
    sorted_tokens = sorted(other_tokens.items(), key=lambda x: x[1], reverse=True)
    final_tokens = dict(sorted_tokens[:self.vocab_size - len(char_tokens)])
    final_tokens.update(char_tokens)
    
    self.final_vocab = final_tokens
    print(f"Training completed. Final vocabulary size: {len(self.final_vocab)}")
    
    return self.final_vocab
  
  def get_vocab(self) -> Dict[str, float]: return self.final_vocab.copy()
  
  def save_vocab(self, filepath: str):
    with open(filepath, 'w', encoding='utf-8') as f:
      for token, score in sorted(self.final_vocab.items(), key=lambda x: x[1], reverse=True):
        f.write(f"{token}\t{score}\n")
  
  def load_vocab(self, filepath: str):
    vocab = {}
    with open(filepath, 'r', encoding='utf-8') as f:
      for line in f:
        if '\t' in line:
          token, score = line.strip().split('\t', 1)
          vocab[token] = float(score)
    self.final_vocab = vocab
    return vocab