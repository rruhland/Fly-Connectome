"""Fixed centered degree-two TensorSketch on already capped legal visual inputs."""
import numpy as np
import scipy.sparse as sp
from fly_connectome.control.policy import FrozenLinearPolicy

SCHEMA='fixed-centered-tensorsketch2-v1'


class FixedPairwise:
    def __init__(self,hashes,signs,scale,mean,rms):
        self.hashes=np.array(hashes,dtype=np.int64,copy=True);self.signs=np.array(signs,dtype=np.float64,copy=True);self.scale=np.array(scale,dtype=np.float64,copy=True);self.mean=np.array(mean,dtype=np.float64,copy=True);self.rms=np.array(rms,dtype=np.float64,copy=True);self.output_dim=len(self.rms)
        d=len(self.scale);m=self.output_dim
        if (self.hashes.shape!=(2,d) or self.signs.shape!=(2,d) or self.mean.shape!=(d,) or m<1 or not np.isfinite(self.signs).all() or not np.isin(self.signs,[-1,1]).all() or not np.isfinite(self.mean).all() or not np.isfinite(self.scale).all() or not (self.scale>0).all() or not np.isfinite(self.rms).all() or not (self.rms>0).all() or not ((self.hashes>=0)&(self.hashes<m)).all()):raise ValueError('finite, dimension-consistent fixed hash/sign/normalization assets required')
        self.projections=tuple(sp.csr_matrix((self.signs[j]/self.scale,(np.arange(d),self.hashes[j])),shape=(d,m)) for j in range(2));self.mean_sketch=np.array([np.bincount(self.hashes[j],weights=self.signs[j]*self.mean,minlength=m) for j in range(2)])
        for a in (self.hashes,self.signs,self.scale,self.mean,self.rms,self.mean_sketch):a.flags.writeable=False

    def features(self,indices,values):
        z=np.asarray(values,dtype=np.float64)/self.scale[indices];c=[np.bincount(self.hashes[j,indices],weights=self.signs[j,indices]*z,minlength=self.output_dim)-self.mean_sketch[j] for j in range(2)]
        return np.fft.irfft(np.fft.rfft(c[0])*np.fft.rfft(c[1]),n=self.output_dim)/self.rms

    def batch(self,X):
        X=sp.csr_matrix(X);c=[(X@self.projections[j]).toarray()-self.mean_sketch[j] for j in range(2)]
        return np.fft.irfft(np.fft.rfft(c[0],axis=1)*np.fft.rfft(c[1],axis=1),n=self.output_dim,axis=1)/self.rms


class FrozenPairwiseReadout:
    schema=SCHEMA
    def __init__(self,base_weights,basis,residual):
        self.base=FrozenLinearPolicy(base_weights);self.basis=basis;self.residual=np.array(residual,dtype=np.float64,copy=True)
        if self.residual.shape!=(3,basis.output_dim) or not np.isfinite(self.residual).all():raise ValueError('finite three-action output weights matching fixed map required')
        self.residual.flags.writeable=False;self.enabled=bool(np.any(self.residual))

    def choose(self,indices,values,legal_actions=None):
        action,q=self.base.choose(indices,values,legal_actions)
        if not self.enabled:return action,q
        q=q+self.residual@self.basis.features(indices,values);legal=np.arange(3) if legal_actions is None else np.asarray(legal_actions)
        if not np.isfinite(q).all():raise FloatingPointError('nonfinite fixed pairwise scores')
        return int(legal[np.argmax(q[legal])]),q

    def frozen_parameters(self):return SCHEMA,self.base.weights,self.basis.hashes,self.basis.signs,self.basis.scale,self.basis.mean,self.basis.rms,self.residual


def save_readout(path,base,basis,residual):
    with open(path,'wb') as f:np.savez(f,kind=SCHEMA,base=base,hashes=basis.hashes,signs=basis.signs,scale=basis.scale,mean=basis.mean,rms=basis.rms,residual=residual)
