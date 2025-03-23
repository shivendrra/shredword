import ctypes, os
from .cbase import BaseTokenizer, MAX_LINE_LENGTH, lib, CShred
import regex as re

class TokenizerBase:
  def __init__(self):
    self.tokenizer = BaseTokenizer()
    lib.init_tokenizer(ctypes.byref(self.tokenizer))

  def build_vocab(self):
    lib.build_vocab(ctypes.byref(self.tokenizer))

  def replace_control_characters(self, input_str):
    output = ctypes.create_string_buffer(MAX_LINE_LENGTH)
    lib.replace_control_characters(input_str.encode("utf-8"), output)
    return output.value.decode("utf-8")

  def save_tokenizer(self, file_path):
    lib.save_tokenizer(ctypes.byref(self.tokenizer), file_path.encode("utf-8"))

  def load_tokenizer(self, file_path):
    lib.load_tokenizer(ctypes.byref(self.tokenizer), file_path.encode("utf-8"))

  def free(self):
    lib.free_tokenizer(ctypes.byref(self.tokenizer))

class Shred:
  def __init__(self):
    self._tokenizer = CShred()
    lib.init_shred(ctypes.byref(self._tokenizer))

  def train(self, text, vocab_size, min_freq=None):
    min_freq = 0 if min_freq is None else min_freq
    text_c = ctypes.create_string_buffer(text.encode("utf-8"))
    lib.dynamic_train_bpe(ctypes.byref(self._tokenizer), text_c, vocab_size, min_freq)

  def encode(self, text):
    text_c = ctypes.create_string_buffer(text.encode("utf-8"))
    output_size = ctypes.c_int()
    encoded_ptr = lib.encode(ctypes.byref(self._tokenizer), text_c, ctypes.byref(output_size))
    encoded = [encoded_ptr[i] for i in range(output_size.value)]
    return encoded

  def decode(self, ids):
    array_type = ctypes.c_int * len(ids)
    id_array = array_type(*ids)
    decoded_ptr = lib.decode(ctypes.byref(self._tokenizer), id_array, len(ids))
    decoded = ctypes.string_at(decoded_ptr).decode("utf-8")
    return decoded

  def save(self, file_path):
    file_path_c = ctypes.create_string_buffer(file_path.encode("utf-8"))
    lib.save_model(ctypes.byref(self._tokenizer), file_path_c)
    print("Saved the model sucessfully!!")

  def load(self, file_path):
    file_path_c = ctypes.create_string_buffer(file_path.encode("utf-8"))
    lib.load_model(ctypes.byref(self._tokenizer), file_path_c)
    print("Loaded the model sucessfully!!")

  def _build_vocab(self):
    vocab = {idx: bytes([idx]) for idx in range(256)}
    for p0, p1, idx in self.merges:
      vocab[idx] = vocab[p0] + vocab[p1]
    return vocab

  @property
  def merges(self):
    merges = lib.export_merges(ctypes.byref(self._tokenizer))
    result = ctypes.string_at(merges).decode("utf-8")
    # regex to match (p0, p1) idx pattern
    pattern = re.compile(r"\((\d+), (\d+)\) (\d+)")
    return [tuple(map(int, match.groups())) for match in pattern.finditer(result)]

  @property
  def vocab(self):
    return self._build_vocab()

  @property
  def pattern(self):
    pattern = lib.export_pattern(ctypes.byref(self._tokenizer))
    if not pattern: return ["None"]
    result = ctypes.string_at(pattern).decode("utf-8").strip()
    return result

  @pattern.setter
  def pattern(self, new_pattern):
    lib.set_pattern(ctypes.byref(self._tokenizer), new_pattern.encode("utf-8"))

  @property
  def special_tokens(self):
    tokens = lib.export_special_tokens(ctypes.byref(self._tokenizer))
    if not tokens: return ["None"]
    result = ctypes.string_at(tokens).decode("utf-8").strip().splitlines()
    return result
  
  @special_tokens.setter
  def special_tokens(self, token_list):
    serialized = "\n".join(f"{token} {index}" for token, index in token_list)
    lib.set_special_tokens(ctypes.byref(self._tokenizer), serialized.encode("utf-8"))

  def free(self):
    lib.free_tokenizer(ctypes.byref(self._tokenizer))