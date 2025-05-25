import os, ctypes
from ctypes import *

libtrainer_path = os.path.join(os.path.dirname(__file__), "../build/libtrainer.so")
libtrainer = ctypes.CDLL(libtrainer_path)

