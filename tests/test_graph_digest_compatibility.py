"""Golden graph-v2 digests captured before streaming the pickle output."""
import pytest
from fly_connectome.control.codec import fingerprint
from collections import deque
from types import SimpleNamespace
import numpy as np
import torch

def cases():
    a=np.arange(48.,dtype=np.float64).reshape(6,8)
    t=torch.arange(24.,dtype=torch.float32).reshape(4,6)
    cycle=[];cycle.append(cycle)
    shared=torch.arange(8.)
    return [
      None,
      {'z':(True,12,-0.0,float('inf'),float('nan')), 'a':deque([b'ab','ab',None])},
      {'a':a,'alias':a,'view':a.T,'strided':a[:,::2]},
      {'t':t,'alias':t,'view':t.T,'conj':torch.tensor([1+2j]).conj()},
      cycle,
      [shared,shared]+[torch.arange(8.) for _ in range(32)],
      [torch.arange(i%3+1.) for i in range(40)],
      {'numpy':np.random.default_rng(1729),'torch':torch.Generator().manual_seed(1729)},
      SimpleNamespace(matrix=np.arange(1000000,dtype=np.float64).reshape(1000,1000),config={'a':2}),
      {'tuple':(a,a),'list':[a.copy(),a.copy()]},
    ]

EXPECTED = ['ecf1a9b28da31a4630b9690066819921b8fd25935bccb416ddb54d8f3b00a01e', '186048c937f037bf1e08587bf1c95329caf208de47df402a770acf77592c4ea5', '3e259d37e20f5966f861b5ea29dad7e07929de5bbc2b450b16e9636220c946dd', '72ac790cbc3c69ac4116b3453ee722733e2b3bc791d164819a8b38ead55e1af2', '2cff0e5025948c455322a5229616b94448059b69251ef619920d50d9b76baf30', '9fbebb526b5ca15def812a90d34b4dce51e430c5446a66802c6cdd4488da7b04', 'fd16c6fec3c3e86c9b8dd3182472c6811f3900a0c5db0f46aeb06be92ae2223d', '97e31af923d351d62c2f023d77f9ce1c909d777b6ff7a1cc45fe0b58cbbe4e25', 'afe4e53574c381d85e9d418aa497aabc51e56f62fc2d19658783cc334b9b3577', 'df8f5d86fe6c7a8da412b64f852e60349e2ed9ce881f94b1f73ffa91d97d0535']

@pytest.mark.parametrize("index", range(len(EXPECTED)))
def test_pre_stream_graph_digest_compatibility(index):
    assert fingerprint(cases()[index]) == EXPECTED[index]


def test_large_array_mutation_and_aliases_remain_visible():
    a = np.zeros((1000,1000))
    before = fingerprint([a,a])
    assert fingerprint([a,a.copy()]) != before
    a[-1,-1] = 1
    assert fingerprint([a,a]) != before
