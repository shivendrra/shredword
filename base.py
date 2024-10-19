from typing import List, Tuple, Dict, Set
from collections import defaultdict
import regex as re
import threading

MAX_RANK = float('inf')

def _byte_pair_merge(ranks, piece):
  parts = []
  min_rank = (MAX_RANK, float('inf'))
  for i in range(len(piece) - 1):
    rank = ranks.get(piece[i:i + 2], MAX_RANK)
    if rank < min_rank[0]:
      min_rank = (rank, i)
    parts.append((i, rank))
  
  parts.append((len(piece) - 1, MAX_RANK))
  parts.append((len(piece), MAX_RANK))

  def get_rank(parts: List[Tuple[int, int]], i: int) -> int:
    if (i + 3) < len(parts):
      return ranks.get(piece[parts[i][0]:parts[i+3][0]], MAX_RANK)
    else:return MAX_RANK
  
  while min_rank[0] != MAX_RANK:
    i = min_rank[1]
    if i > 0:
      parts[i - 1] = (parts[i - 1][0], get_rank(parts, i - 1))
      parts[i] = (parts[i][0], get_rank(parts, i))
      parts.pop(i + 1)
      min_rank = (MAX_RANK, float('inf'))
      for idx, (_, rank) in enumerate(parts[:-1]):
        if rank < min_rank[0]:
          min_rank = (rank, idx)
    return parts

def _byte_pair_encode(piece: bytes, ranks: Dict[bytes, int]) -> List[int]:
  assert len(piece) > 1, "The piece must have more than one byte to perform BPE"
  parts = _byte_pair_merge(ranks, piece)
  encoded_ranks = [ranks[piece[parts[i][0]:parts[i+1][0]]] for i in range(len(parts) - 1)]
  return encoded_ranks

def _byte_pair_split(piece: bytes, ranks: Dict[bytes, int]) -> List[bytes]:
  assert len(piece) > 1, "The piece must have more than one byte to perform BPE"
  parts = _byte_pair_merge(ranks, piece)
  split_pairs = [piece[parts[i][0]:parts[i + 1][0]] for i in range(len(parts) - 1)]
  return split_pairs

MAX_NUM_THREADS = 128

def hash_current_thread() -> int:
  """
  Get a hash of the current thread ID.

  Returns:
  - An integer hash representing the current thread.
  """
  thread_id = threading.get_ident()
  return thread_id & ((1 << 64) - 1)

class CoreBPE:
  """
  CoreBPE class for managing byte pair encoding and tokenization logic.
  
  Attributes:
  - encoder: Dictionary mapping byte tokens to their ranks.
  - special_tokens_encoder: Dictionary for special token mappings to ranks.
  - decoder: Dictionary mapping ranks back to their byte sequences.
  - special_tokens_decoder: Dictionary for special token ranks back to sequences.
  - regex_tls: List of regular expression patterns for tokenization.
  - special_regex_tls: List of regular expressions for special tokens.
  - sorted_token_bytes: List of sorted byte sequences for fast lookup.
  """
  def __init__(self):
    self.encoder, self.special_tokens_encoder, self.decoder, self.special_tokens_decoder = {}, {}, {}, {}
    self.sorted_token_bytes = []
    self.regex_tls = [re.compile("") for _ in range(MAX_NUM_THREADS)]  # Regular expressions per thread
    self.special_regex_tls = [re.compile("") for _ in range(MAX_NUM_THREADS)]  # Special token regex patterns per thread

  def __repr__(self):
    """
    String representation of the CoreBPE class.
    """
    return f"CoreBPE(encoder={len(self.encoder)} tokens, special_tokens={len(self.special_tokens_encoder)} tokens)"
  
  def _get_tl_regex(self) -> re.Pattern: return self.regex_tls[hash_current_thread()]
  def _get_tl_special_regex(self) -> re.Pattern: return self.special_regex_tls[hash_current_thread()]
  def _decode_native(self, tokens: List[int]) -> bytes:
    result = bytearray()
    for token in tokens:
      token_bytes = self.decoder.get(token, self.special_tokens_decoder.get(token, b""))
      result.extend(token_bytes)
    return bytes(result)
  
  def _encode_ordinary_native(self, text: str) -> List[int]:
    regex = self._get_tl_regex()
    ret = []
    for match in regex.finditer(text):
      piece = match.group().encode("utf-8")
      if piece in self.encoder:
        ret.append(self.encoder[piece])
      else:
        ret.extend(_byte_pair_encode(piece, self.encoder))
    return ret
  
  def _encode_native(self, text: str, allowed_special: Set[str]) -> Tuple[List[int], int]:
    special_regex, regex = self._get_tl_special_regex(), self._get_tl_regex()
    ret, start, last_piece_token_len = [], 0, 0

    while True:
      next_special, start_find = None, start
      while True:
        match = special_regex.search(text, start_find)
        if match:
          if text[match.start():match.end()] in allowed_special:
            next_special = match
            break
          start_find = match.start() + 1
        else:
          break
      end = next_special.start() if next_special else len(text)

      for match in regex.finditer(text[start:end]):
        piece = match.group().encode("utf-8")
        if piece in self.encoder:
          last_piece_token_len = 1
          ret.append(self.encoder[piece])
        else:
          tokens = _byte_pair_encode(piece, self.encoder)
          last_piece_token_len = len(tokens)
          ret.extend(tokens)
      if next_special:
        piece = next_special.group()
        ret.append(self.special_tokens_encoder[piece])
        start, last_piece_token_len = next_special.end(), 0
      else:
        break
    return ret, last_piece_token_len

  def _increase_last_piece_token_len(self, tokens: List[int], last_piece_token_len: int) -> Tuple[List[int], int]:
    def token_is_all_space(token: int) -> bool:
      token_bytes = self.decoder.get(token, b"")
      return all(b in (b' ', b'\n', b'\t') for b in reversed(token_bytes))
    if last_piece_token_len > 0 and token_is_all_space(tokens[-last_piece_token_len]):
      while last_piece_token_len < len(tokens) and token_is_all_space(tokens[-last_piece_token_len - 1]):
        last_piece_token_len += 1
    return tokens, last_piece_token_len

  def _encode_unstable_native(self, text: str, allowed_special: Set[str]) -> Tuple[List[int], Set[Tuple[int]]]:
    tokens, last_piece_token_len = self._encode_native(text, allowed_special)
    if last_piece_token_len == 0:
      return tokens, set()

    tokens, last_piece_token_len = self._increase_last_piece_token_len(tokens, last_piece_token_len)
    unstable_bytes = self._decode_native(tokens[-last_piece_token_len:])
    tokens, completions = tokens[:-last_piece_token_len], set()

    if not unstable_bytes:
      return tokens, completions

    point = self._partition_point(self.sorted_token_bytes, unstable_bytes)
    while point < len(self.sorted_token_bytes) and self.sorted_token_bytes[point].startswith(unstable_bytes):
      completions.add((self.encoder[self.sorted_token_bytes[point]],))
      point += 1

    for i in range(1, len(unstable_bytes)):
      prefix, suffix = unstable_bytes[:i], unstable_bytes[i:]
      point = self._partition_point(self.sorted_token_bytes, suffix)
      while point < len(self.sorted_token_bytes) and self.sorted_token_bytes[point].startswith(suffix):
        possibility = prefix + self.sorted_token_bytes[point]
        encoded = self._encode_ordinary_native(possibility.decode("utf-8"))
        seq,seq_len = [], 0
        for token in encoded:
          seq.append(token)
          seq_len += len(self.decoder[token])
          if seq_len >= len(unstable_bytes):
            break
        completions.add(tuple(seq))
        point += 1
    return tokens, completions
  
  def _partition_point(self, arr: List[bytes], target: bytes) -> int:
    lo, hi = 0, len(arr)
    while lo < hi:
      mid = (lo + hi) // 2
      if arr[mid] < target:
        lo = mid + 1
      else:
        hi = mid
    return lo