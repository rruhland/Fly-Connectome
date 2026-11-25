from collections import deque

import numpy as np

from scripts.experimental.tt50_surprise_actor import Retrieval
from scripts.experimental.tt50_surprise_data import key


def example(ident, good, sign):
    rates=np.zeros((5,6))
    rates[:,:5]=sign*.65
    return {"id":ident,"good":good,"z":[0.]*25,"actions":rates.tolist()}


def test_action_reward_increases_only_with_independent_physical_examples():
    all_rows=[example(1,True,1),example(2,True,1),
              example(3,False,-1),example(4,False,-1),example(5,False,-1)]
    r=Retrieval(all_rows)
    z=np.zeros(25)
    guessed,note=r.choose(z,"action-conditional")
    assert guessed.shape==(5,6) and np.max(abs(guessed))<=.8
    assert note["supported_releases"]==2
    assert r.estimate(z,all_rows[0]["actions"]) > r.estimate(z) > r.estimate(z,all_rows[-1]["actions"])
    duplicate=Retrieval([all_rows[0],example(1,True,1),*all_rows[2:]])
    assert duplicate.choose(z,"action-conditional")[0] is None


def test_causal_key_needs_current_and_four_contiguous_inbound_ticks():
    x=np.array([.1,-1.1,1.03,0.,-4,0.])
    seq=deque((t,x.copy(),np.zeros(6),np.zeros(6)) for t in range(5))
    assert key(seq).shape==(25,)
    seq.append((7,x,np.zeros(6),np.zeros(6)))
    assert key(seq) is None
