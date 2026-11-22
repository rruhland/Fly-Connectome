"""Reuse an identical private opponent search across paired physical treatments."""
from copy import deepcopy
from hashlib import sha256
import pickle

import numpy as np

from .tt4_arena import mujoco
from .tt21_coach_oracle import PHASE


class FarCache:
    """One cache per physical model, opponent and candidate generator."""
    def __init__(self,opponent):
        self.opponent=opponent
        self.store={}
        self.hits=self.misses=self.checked=0
        self.plan=opponent.plan
        self.flags=mujoco.mjtState.mjSTATE_INTEGRATION
        self.state=np.empty(mujoco.mj_stateSize(opponent.arena.model,self.flags))
        opponent.plan=self.get

    def get(self):
        far=self.opponent
        arena=far.arena
        mujoco.mj_getState(arena.model,arena.data,self.state,self.flags)
        key=sha256(self.state.tobytes())
        fields=[(k,getattr(arena,k)) for k in PHASE if hasattr(arena,k)]
        key.update(pickle.dumps((arena.seed,arena.max_steps,fields,far.target()),protocol=5))
        signature=key.digest()
        if signature not in self.store:
            self.misses+=1
            result=self.plan()
            self.store[signature]=deepcopy(result)
            return result
        self.hits+=1
        tape,note=deepcopy(self.store[signature])
        if self.checked<2:
            ref,receipt=self.plan()
            if ((tape is None)!=(ref is None) or not np.array_equal(tape,ref) or
                    {k:v for k,v in note.items() if k!='wall_seconds'}!={k:v for k,v in receipt.items() if k!='wall_seconds'}):
                raise ValueError('reused opponent search differs from live deterministic rating')
            self.checked+=1
        note['wall_seconds']=0.
        return tape,note
