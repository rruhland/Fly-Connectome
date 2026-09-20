"""Optional native merge experiment; build the library explicitly before running."""
import ctypes
from pathlib import Path
import torch


def load_merge(path):
    library = ctypes.CDLL(str(Path(path).resolve()))
    fn = library.merge_arrivals
    fn.argtypes = [ctypes.c_int64, ctypes.c_int64] + [ctypes.c_void_p]*8
    fn.restype = ctypes.c_int64
    def merge(old_keys, old_values, old_traces, arrival_keys):
        inputs = (old_keys, old_values, old_traces, arrival_keys)
        if any(x.device.type != 'cpu' or not x.is_contiguous() or x.requires_grad for x in inputs):
            raise ValueError('native experiment requires contiguous CPU tensors without autograd')
        if (old_keys.dtype != torch.int64 or arrival_keys.dtype != torch.int64
                or old_values.dtype != torch.float32 or old_traces.dtype != torch.float32
                or any(x.ndim != 1 for x in inputs)
                or len(old_keys) != len(old_values) or len(old_keys) != len(old_traces)):
            raise ValueError('native experiment requires aligned int64 keys and float32 traces')
        incoming = arrival_keys.clone()  # native sorts only its private arrival copy
        size = len(old_keys)+len(incoming)
        keys = old_keys.new_empty(size)
        values, traces, arrivals = (old_values.new_empty(size) for _ in range(3))
        count = fn(len(old_keys),len(incoming),*(x.data_ptr() for x in
                   (old_keys,old_values,old_traces,incoming,keys,values,traces,arrivals)))
        return keys[:count], values[:count], traces[:count], arrivals[:count]
    return merge
