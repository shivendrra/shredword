import ctypes, os

lib_path = os.path.join(os.path.dirname(__file__), "libtoken.so")
lib = ctypes.CDLL(lib_path)

class CToken(ctypes.Structure):
  pass

CToken._fields_ = [
  ()
]