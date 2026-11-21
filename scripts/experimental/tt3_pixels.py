"""Opt-in compiled RGB segmentation, preserving the reference component order."""
import ctypes
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=1)
def _library(path):
    handle = ctypes.CDLL(path)
    fn = handle.tt3_components
    fn.argtypes = (ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p)
    fn.restype = ctypes.c_int
    return fn


def regions(frame, path):
    height, width, _ = frame.shape
    frame = np.ascontiguousarray(frame)
    records = np.empty((height * width // 2, 4), dtype=np.int32)
    n = _library(path)(frame.ctypes.data, height, width, records.ctypes.data)
    return records[:n]
