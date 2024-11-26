"""
--> contains important functions for tokenization and training bpe
--> save/load functions are also available
--> for understanding the functionality and basic implementaion only
--> inspired(copied) from Karpathy's minbpe
"""

import regex as re
import unicodedata

merges = {}
vocab = {idx: bytes([idx]) for idx in range(256)}
pattern = ""
special_tokens = {}

def get_stats(ids, counts=None):
  """
    takes list of integers and returns dictionary of counts of pairs(consecutive ones)
    eg: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    allows to update an existing dictionary of counts
  """
  counts = {} if counts is None else counts
  for pair in zip(ids, ids[1:]):
    counts[pair] = counts.get(pair, 0) + 1
  return counts

def merge(ids, pair, idx):
  """
    in the list of integers, replaces all consecutive pair with the new integer token idx
    eg: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
  """
  new_ids = []
  i = 0
  while i < len(ids):
    if i+1 < len(ids) and ids[i] == pair[0] and ids[i+1] == pair[1]:
      new_ids.append(idx)
      i += 2
    else:
      new_ids.append(ids[i])
      i += 1
  return new_ids

def apply_regex(text):
  r"""
  	## space is merged with each word, before it as a prefix
  	## a litlle smaller than pattern2
	  regex_pattern1: '(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+
	
	  ## space is added as a preffix to each word, retains all the initial words
	  ## smaller than pattern3
  	regex_pattern2: '(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+

  	## space is considered a separate token, all words remain original, no loss of words
  	## largest in length
  	regex_pattern3: 's|'t|'re|'ve|'m|'ll|'d|[\w']+|[^\s\w\d]+|\s+(?!\S)|\s+
  
  	## spaces are added as a prefix to the words, but some words are missing hence doesn't retains original text
  	## smallest in length, due to some lost words
  	regex_pattern4: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+ | ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
	"""
  pattern = re.compile(r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+""")
  text = re.findall(pattern, text)
  return text

def build_vocab(merges, special_tokens):
  vocab = {idx: bytes([idx]) for idx in range(256)}
  for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
  
  for special, idx in special_tokens.items():
    vocab[idx] = special.encode("utf-8")
  
  return vocab

def replace_control_characters(s: str) -> str:
  chars = []
  for ch in s:
      if unicodedata.category(ch)[0] != "C":
          chars.append(ch)
      else:
          chars.append(f"\\u{ord(ch):04x}")
  return "".join(chars)

def render_token(t: bytes) -> str:
  s = t.decode('utf-8', errors='replace')
  s = replace_control_characters(s)
  return s

def decode(ids):
  text_bytes = b"".join(vocab[idx] for idx in ids)
  text = text_bytes.decode("utf-8", errors="replace")
  return text

def encode(text):
  text_bytes = text.encode("utf-8")
  ids = list(text_bytes)
  while len(ids) >= 2:
    stats = get_stats(ids)
    pair = min(stats, key=lambda p: merges.get(p, float('inf')))
    if pair not in merges:
      break

    idx = merges[pair]
    ids = merge(ids, pair, idx)
  return ids

def train(text, vocab_size, verbose=False):
  """
    training loop for the bpe tokenizer
  """
  assert vocab_size >= 256

  num_merges = vocab_size - 256
  text_bytes = text.encode("utf-8")
  ids = list(text_bytes)

  for i in range(num_merges):
    stats = get_stats(ids)
    pair = max(stats, key=stats.get)
    idx = 256 + i
    ids = merge(ids, pair, idx)
    merges[pair] = idx
    vocab[idx] = vocab[pair[0]] + vocab[pair[1]]

    if verbose:
      print(f"merge {i+1}/{num_merges}: {pair} -> {idx} ({vocab[idx]}) had {stats[pair]} occurrences")

def save_model(file_name):
  """
    Saves two files: file_prefix.vocab and file_prefix.model
    This is inspired (but not equivalent to!) sentencepiece's model saving:
    - model file is the critical one, intended for load()
    - vocab file is just a pretty printed version for human inspection only
  """
  model_file = file_name + ".model"
  inverted_merges = {idx: pair for pair, idx in merges.items()}
  with open(model_file, 'w') as f:
    f.write("minibpe v1\n")
    f.write(f"{pattern}\n")
    f.write(f"{len(special_tokens)}\n")
    for special, idx in special_tokens.items():
      f.write(f"{special} {idx}\n")
    for idx1, idx2 in merges:
      f.write(f"{idx1} {idx2}\n")
    
  vocab_file = file_name + ".vocab"
  with open(vocab_file, 'w') as f:
    for idx, token in vocab.items():
      s = render_token(token)
      if idx in inverted_merges:
        idx0, idx1 = inverted_merges[idx]
        s0 = render_token(vocab[idx0])
        s1 = render_token(vocab[idx1])
        f.write(f"[{s0}][{s1}] -> [{s}] {idx}\n")
      else:
        f.write(f"[{s}] {idx}\n")

def load(model_file):
  """
    loads '.model' file
    returns vocab loaded from the file
  """
  assert model_file.endswith('.model')
  merges = {}
  special_tokens = {}
  idx = 256
  with open(model_file, 'r', encoding='utf-8') as f:
    version = f.readline().strip()
    assert version == "minbpe v1"

    pattern = f.readline().strip()
    num_special = int(f.readline().strip())
    for _ in range(num_special):
      special, special_idx = f.readline().strip().split()
      special_tokens[special] = int(special_idx)
    
    for line in f:
      idx1, idx2 = map(int, line.split())
      merges[(idx1, idx2)] = idx
      idx += 1
    
    merges = merges
    special_tokens = special_tokens
    vocab = build_vocab(merges, special_tokens)
  
  return vocab