"""Legal frozen-vision Ogma temporal representation with a learned local prior residual."""
import ctypes as C
from pathlib import Path
import numpy as np
from fly_connectome.control.scaling import policy_features

TEMPERATURE=.008161197682294006
F32=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')


def pack_current(indices,values):
    planes=np.zeros(1792,dtype=np.float32);mask=np.asarray(indices)<1792
    planes[np.asarray(indices)[mask]]=np.asarray(values)[mask]
    return np.ascontiguousarray(planes.reshape(7,16,16).transpose(2,1,0)).reshape(-1)


class OgmaRuntime:
    def __init__(self,library,baseline,checkpoint=None,seed=2026102888):
        self.baseline=baseline;self.lib=C.CDLL(str(library));self.lib.ogma_create.argtypes=[C.c_uint];self.lib.ogma_create.restype=C.c_void_p
        self.lib.ogma_free.argtypes=[C.c_void_p];self.lib.ogma_reset.argtypes=[C.c_void_p]
        self.lib.ogma_step.argtypes=[C.c_void_p,F32,C.c_int,F32,C.c_float,C.c_float,C.c_int,C.c_int,C.c_int,F32,F32,F32];self.lib.ogma_step.restype=C.c_int
        for name in ('ogma_weights','ogma_representation','ogma_actor'):
            getattr(self.lib,name).argtypes=[C.c_void_p];getattr(self.lib,name).restype=C.c_uint64
        for n in ('save','load'):
            f=getattr(self.lib,'ogma_'+n);f.argtypes=[C.c_void_p,C.c_char_p];f.restype=C.c_int
        self.pointer=self.lib.ogma_create(seed);assert self.pointer
        if checkpoint is not None:assert self.lib.ogma_load(self.pointer,str(checkpoint).encode())==0
        self.reset()

    def reset(self):
        self.lib.ogma_reset(self.pointer);self.previous_action=1;self.previous_p=np.full(3,1/3,dtype=np.float32);self.reward=0.;self.discount=1.;self.terminal=False

    def representation(self):return int(self.lib.ogma_representation(self.pointer))
    def weights(self):return int(self.lib.ogma_weights(self.pointer))
    def save(self,path):assert self.lib.ogma_save(self.pointer,str(path).encode())==0
    def close(self):
        if self.pointer:self.lib.ogma_free(self.pointer);self.pointer=None

    def feedback(self,action,p,hits,elapsed,terminal):
        self.previous_action=int(action);self.previous_p=np.ascontiguousarray(p,dtype=np.float32);self.reward=float(hits);self.discount=float(.95**elapsed);self.terminal=bool(terminal)

    def raw_query(self,indices,values,learn_representation=False,learn_actor=False):
        greedy,baseq=self.baseline.choose(indices,values)
        image=pack_current(indices,values)
        prior=np.ascontiguousarray(baseq/TEMPERATURE,dtype=np.float32);p=np.empty(3,dtype=np.float32);residual=np.empty(3,dtype=np.float32)
        sampled=self.lib.ogma_step(self.pointer,image,self.previous_action,self.previous_p,self.reward,self.discount,self.terminal,learn_representation,learn_actor,prior,p,residual)
        assert 0<=sampled<3 and np.isfinite(p).all() and np.isfinite(residual).all() and abs(float(p.sum())-1)<2e-6
        # Literal delegation protects initial ties and arbitrarily close float64 decisions.
        if not residual.any():action,q=greedy,baseq
        else:q=baseq/TEMPERATURE+residual.astype(np.float64);action=int(np.argmax(q))
        return dict(actor_fingerprint=int(self.lib.ogma_actor(self.pointer)),action=int(sampled if learn_actor else action),greedy=int(action),base_action=int(greedy),q=np.asarray(q).tolist(),p=p.tolist(),residual=residual.tolist())

    def query(self,session,**kwargs):return self.raw_query(*policy_features(session.latest_encoded,session.scaler),**kwargs)
