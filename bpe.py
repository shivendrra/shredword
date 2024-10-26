import json
from tqdm import tqdm
from collections import Counter

class BasicTokenizer:
  def __init__(self):
    self.vocab_size = 0
    self.vocab = {}
    self.merges = {}

  def _get_stats(self, ids): 
    """
    Takes a list of integers and returns a dictionary of counts of pairs(consecutive ones).
    Eg: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    """
    pairs = [(ids[i], ids[i+1]) for i in range(len(ids) - 1)]
    return Counter(pairs)

  def _merge(self, ids, pair, idx):
    """
    Replaces all consecutive pair with the new integer token idx in the list of integers.
    Eg: ids=[1, 2, 3, 1, 2], pair=(1, 2), idx=4 -> [4, 3, 4]
    """
    return [idx if (ids[i] == pair[0] and ids[i+1] == pair[1]) else ids[i] for i in range(len(ids)-1)]

  def _build_vocab(self, merges):
    """
    Builds the new vocab by merging the merge values together.
    """
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for (p0, p1), idx in merges.items():
      vocab[idx] = vocab[p0] + vocab[p1]
    return vocab

  def train(self, train_data, target_vocab):
    assert target_vocab >= 256
    n_merges = target_vocab - 256
    text_bytes = train_data.encode('utf-8')
    ids = list(text_bytes)

    stats = self._get_stats(ids)
    merges = {}
    for i in tqdm(range(n_merges), desc='Training the tokenizer\t'):
      pair = max(stats, key=stats.get)
      idx = 256 + i
      stats = self._update_stats(stats, ids, pair, idx)
      merges[pair] = idx
    
    vocab = self._build_vocab(merges)
    self.vocab = vocab
    self.merges = merges
    self.vocab_size = len(self.vocab)

    return self.vocab, self.merges

  def _update_stats(self, stats, ids, pair, idx):
    """
    Updates the statistics after each merge.
    """
    new_stats = stats.copy()
    new_stats.pop(pair)
    for i in range(len(ids) - 1):
      if (ids[i], ids[i+1]) == pair:
        new_pair = (idx, ids[i+2] if i+2 < len(ids) else ids[i+1])
        new_stats[new_pair] = new_stats.get(new_pair, 0) + 1
    return new_stats

  def encode(self, text):
    """
    Encodes the input string using 'utf-8' encodings.
    """
    text_bytes = text.encode('utf-8')
    ids = list(text_bytes)
    stats = self._get_stats(ids)
    while len(ids) >= 2:
      pair = min(stats, key=lambda p: self.merges.get(p, float('inf')))
      if pair not in self.merges:
        break

      idx = self.merges[pair]
      stats = self._update_stats(stats, ids, pair, idx)
      ids = [idx if (ids[i] == pair[0] and ids[i+1] == pair[1]) else ids[i] for i in range(len(ids)-1)]
    return ids

  def decode(self, ids):
    """
    Decodes the input list of integers into string.
    """
    text_bytes = b"".join(self.vocab[idx] for idx in ids)
    text = text_bytes.decode('utf-8', errors='replace')
    return text

  def save_model(self, prefix):
    """
    Saves the model to files '.model' & 'vocab.json'.
    """
    model_file = prefix + '.model'
    with open(model_file, 'w') as f:
      for idx1, idx2 in self.merges:
        f.write(f"{idx1} {idx2}\n")
    f.close()

    vocab_file = prefix + '_vocab.json'
    with open(vocab_file, 'w') as f:
      serializable_vocab = {}
      for idx in self.vocab:
        try: 
          serializable_vocab[idx] = str(self.vocab[idx], 'utf-8')
        except:
          pass
      json.dump(serializable_vocab)
    f.close()

  def load_model(self, model_file):
    """
    Loads the model from '.model' file.
    """
    assert model_file.endswith('.model')

    merges = {}
    idx = 256
    with open(model_file, 'r', encoding='utf-8') as f:
      for line in f:
        idx1, idx2 = map(int, line.split())
        merges[(idx1, idx2)] = idx
        idx += 1
    f.close()

    self.merges = merges
    self.vocab = self._build_vocab(merges)