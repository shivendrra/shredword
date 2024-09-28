import ctypes
import os

# Load the shared object file
lib_path = os.path.abspath("libsrc.so")
bpe_lib = ctypes.CDLL(lib_path)

# Define ctypes mappings for BPE structs and functions
class BpeEntry(ctypes.Structure):
  _fields_ = [("data", ctypes.POINTER(ctypes.c_uint8)),
              ("length", ctypes.c_size_t),
              ("rank", ctypes.c_uint32)]

class BpeEncoder(ctypes.Structure):
  _fields_ = [("entries", ctypes.POINTER(BpeEntry)),
              ("num_entries", ctypes.c_size_t)]

# Define the argument types and return types for functions in the .so file
bpe_lib.init_bpe_encoder.argtypes = [ctypes.POINTER(BpeEncoder), 
                                     ctypes.POINTER(BpeEntry), ctypes.c_size_t]
bpe_lib.free_bpe_encoder.argtypes = [ctypes.POINTER(BpeEncoder)]

bpe_lib.byte_pair_encode.argtypes = [ctypes.POINTER(ctypes.c_uint8), 
                                     ctypes.c_size_t, ctypes.POINTER(BpeEncoder), 
                                     ctypes.POINTER(ctypes.c_size_t)]
bpe_lib.byte_pair_encode.restype = ctypes.POINTER(ctypes.c_uint32)

bpe_lib.byte_pair_split.argtypes = [ctypes.POINTER(ctypes.c_uint8), 
                                    ctypes.c_size_t, ctypes.POINTER(BpeEncoder), 
                                    ctypes.POINTER(ctypes.c_size_t)]
bpe_lib.byte_pair_split.restype = ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8))

bpe_lib.free_encoded_output.argtypes = [ctypes.POINTER(ctypes.c_uint32)]
bpe_lib.free_split_output.argtypes = [ctypes.POINTER(ctypes.POINTER(ctypes.c_uint8)), 
                                      ctypes.c_size_t]

# Define the Tokenizer class
class Tokenizer:
  def __init__(self, entries):
    # Create the encoder
    self.encoder = BpeEncoder()
    num_entries = len(entries)
    entry_array = (BpeEntry * num_entries)(*entries)
    bpe_lib.init_bpe_encoder(ctypes.byref(self.encoder), entry_array, num_entries)

  def __del__(self):
    bpe_lib.free_bpe_encoder(ctypes.byref(self.encoder))

  def encode(self, text):
    # Convert text to uint8 array
    input_data = (ctypes.c_uint8 * len(text))(*text.encode('utf-8'))
    out_length = ctypes.c_size_t(0)
    encoded_output = bpe_lib.byte_pair_encode(input_data, len(text), ctypes.byref(self.encoder), ctypes.byref(out_length))
    
    # Convert the result to a list of ranks
    result = [encoded_output[i] for i in range(out_length.value)]
    bpe_lib.free_encoded_output(encoded_output)
    return result

  def split(self, text):
    # Ensure the text is non-empty to prevent IndexError
    if not text:
      return []

    # Convert text to uint8 array with proper length
    input_length = len(text.encode('utf-8'))
    input_data = (ctypes.c_uint8 * input_length)(*text.encode('utf-8'))

    out_length = ctypes.c_size_t(0)

    # Call the byte_pair_split function
    split_output = bpe_lib.byte_pair_split(input_data, input_length, ctypes.byref(self.encoder), ctypes.byref(out_length))

    # Check if split_output is NULL
    if not split_output:
      raise ValueError("Split output is NULL. The encoding function may have failed.")

    # Convert the split output to a list of strings
    result = []
    for i in range(out_length.value):
      piece = ctypes.string_at(split_output[i])  # Convert C string to Python string
      result.append(piece.decode('utf-8'))

    # Free the allocated split memory
    bpe_lib.free_split_output(split_output, out_length.value)
    return result

  def build_vocab(self, filepath):
    # Read the file line by line and build the vocabulary
    vocab = set()
    with open(filepath, 'r', encoding='utf-8') as f:
      for line in f:
        tokens = self.split(line.strip())  # Use the split function
        vocab.update(tokens)
    return vocab

# Create an instance of the tokenizer and test
def main():
  # Example BPE entries, replace with actual entries as needed
  entries = [
    BpeEntry(data=(ctypes.c_uint8 * 2)(*b'ab'), length=2, rank=1),
    BpeEntry(data=(ctypes.c_uint8 * 2)(*b'bc'), length=2, rank=2),
    BpeEntry(data=(ctypes.c_uint8 * 1)(*b'a'), length=1, rank=3),
    BpeEntry(data=(ctypes.c_uint8 * 1)(*b'b'), length=1, rank=4)
  ]

  # Initialize the tokenizer with BPE entries
  tokenizer = Tokenizer(entries)

  # Build the vocabulary from a file
  vocab = tokenizer.build_vocab('captions.txt')
  print("Vocabulary:", vocab)

  # Example encoding
  encoded = tokenizer.encode("abc")
  print("Encoded:", encoded)

if __name__ == "__main__":
  main()
