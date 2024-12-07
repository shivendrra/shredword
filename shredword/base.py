import regex as re
import unicodedata
from collections import deque, Counter

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
  # counts = {} if counts is None else counts
  # for pair in zip(ids, ids[1:]):
  #   counts[pair] = counts.get(pair, 0) + 1
  # return counts
  return Counter(zip(ids, ids[1:])) # using Counter over the previous code logic is better as it's optimzed for task like this

def merge(ids, pair, idx):
  """
    in the list of integers, replaces all consecutive pair with the new integer token idx
    eg: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
  """
  # merged = []
  merged, i = deque(), 0 # replaced [] -> deque to minimize the memory reallocations
  while i < len(ids):
    if i+1 < len(ids) and ids[i] == pair[0] and ids[i+1] == pair[1]:
      merged.append(idx)
      i += 2
    else:
      merged.append(ids[i])
      i += 1
  return list(merged)

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
  """
    ## this function basically builds the primary vocab (0-255)
    ## uses 256-ascii characters & put them into a key-value paired dictonary to form
    a lookup table to build merges & get stats of total pairs of bytes
    so the base vocab looks something like this: {'!':0, 'a': 1, 'b': 2, 'c':3 ....., 'x03':255}

    ## uses provided merges & adds new entries to original vocab & also incorporates the 
    special tokens: <|endoftext|>, <|mask|>, <|startoftext|>, etc.
    basically, builds a map of each byte pair to a corresponding integer value/representation

      merges = {('a', 'b'): 256, ('c', 'd'): 257}
      special_tokens = {'<pad>': 258, '<unk>': 259}
  """
  vocab = {idx: bytes([idx]) for idx in range(256)}
  for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]
  for special, idx in special_tokens.items():
    vocab[idx] = special.encode("utf-8")
  return vocab

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
    self.vocab = build_vocab(self.merges, self.special_tokens) # int -> bytes

  # placeholder functions, implemented in child class
  def train(self, text, vocab_size, verbose=False): raise NotImplementedError
  def encode(self, text): raise NotImplementedError
  def decode(self, ids): raise NotImplementedError

  def save(self, file_prefix):
    # saves two files: ``.model`` & ``.vocab``
    # `.vocab` human readable version thats just for debugging & pretty presentation
    # `.model` for furthur training & implementing merges, can be loaded in model
    model_file = file_prefix + ".model"
    with open(model_file, 'w') as f:
      f.write("shredword v1\n")
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
      assert version == "shredword v1"
      self.pattern, num_special = f.readline().strip(), int(f.readline().strip())
      for _ in range(num_special):
        special, special_idx = f.readline().strip().split()
        special_tokens[special] = int(special_idx)
      for line in f:
        idx1, idx2 = map(int, line.split())
        merges[(idx1, idx2)] = idx
        idx += 1
    self.merges, self.special_tokens, self.vocab = merges, special_tokens, build_vocab(merges, special_tokens)