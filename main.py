import ctypes

lib = ctypes.CDLL('./libtoken.so')

class Token(ctypes.Structure):
  _fields_ = [("data", ctypes.c_char_p), ("size", ctypes.c_size_t)]

class TokenList(ctypes.Structure):
  _fields_ = [("tokens", ctypes.POINTER(Token)), ("num_tokens", ctypes.c_size_t)]

lib.tokenize.restype = TokenList
lib.tokenize.argtypes = [ctypes.c_char_p]
lib.free_token_list.argtypes = [ctypes.POINTER(TokenList)]

def tokenize(input_str):
  input_c_str = ctypes.create_string_buffer(input_str.encode('utf-8'))
  token_list = lib.tokenize(input_c_str)
  
  tokens = []
  for i in range(token_list.num_tokens):
    token = token_list.tokens[i].data.decode('utf-8')
    tokens.append(token)
  
  lib.free_token_list(ctypes.byref(token_list))
  
  return tokens