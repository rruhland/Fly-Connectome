"""Approved fixed 64-feature RBF residual over exact capped visual inputs."""
import numpy as np
import scipy.sparse as sp
from fly_connectome.control.policy import FrozenLinearPolicy
from fly_connectome.control.pong_evaluation import FreshPongEvaluator
from fly_connectome.control.scaling import policy_features
from fly_connectome.control.durable import fingerprint

SCHEMA='fixed-local-rbf-residual-v1'


class FixedRBF:
    def __init__(self,centers,scale,sigma):
        self.centers=np.array(centers,dtype=np.float64,copy=True);self.scale=np.array(scale,dtype=np.float64,copy=True);self.sigma=float(sigma)
        if (self.centers.ndim!=2 or self.scale.shape!=(self.centers.shape[1],) or not np.isfinite(self.centers).all() or not np.isfinite(self.scale).all() or not (self.scale>0).all() or not np.isfinite(self.sigma) or self.sigma<=0):raise ValueError('finite fixed scaled centers, positive scale and bandwidth required')
        self.norm=np.einsum('ij,ij->i',self.centers,self.centers)
        for a in (self.centers,self.scale,self.norm):a.flags.writeable=False
        self.output_dim=len(self.centers)

    def features(self,indices,values):
        z=np.asarray(values,dtype=np.float64)/self.scale[indices];distance=np.maximum(0.,self.norm+z@z-2*(self.centers[:,indices]@z));return np.exp(-distance/(2*self.sigma*self.sigma))

    def batch(self,X):
        Z=sp.csr_matrix(X).multiply(1/self.scale).tocsr();norm=np.asarray(Z.power(2).sum(axis=1)).ravel();distance=np.maximum(0.,norm[:,None]+self.norm[None,:]-2*(Z@self.centers.T));return np.exp(-distance/(2*self.sigma*self.sigma))


class FrozenRBFReadout:
    schema=SCHEMA
    def __init__(self,base_weights,basis,residual):
        self.base=FrozenLinearPolicy(base_weights);self.basis=basis;self.residual=np.array(residual,dtype=np.float64,copy=True)
        if self.residual.shape!=(3,len(basis.centers)) or not np.isfinite(self.residual).all():raise ValueError('finite three-action residual matching fixed centers required')
        self.residual.flags.writeable=False;self.enabled=bool(np.any(self.residual))

    def choose(self,indices,values,legal_actions=None):
        action,q=self.base.choose(indices,values,legal_actions)
        # Literal delegation preserves exact source Q arithmetic and tie behavior.
        if not self.enabled:return action,q
        q=q+self.residual@self.basis.features(indices,values);legal=np.arange(3) if legal_actions is None else np.asarray(legal_actions)
        if not np.isfinite(q).all():raise FloatingPointError('nonfinite fixed RBF scores')
        return int(legal[np.argmax(q[legal])]),q

    def frozen_parameters(self):
        return SCHEMA,self.base.weights,self.basis.centers,self.basis.scale,self.basis.sigma,self.residual


def load_readout(path):
    with np.load(path,allow_pickle=False) as f:
        if 'kind' in f and str(f['kind'])=='fixed-continuous-entity-pair-rff-v1':
            from pong_continuous_visual import from_archive
            return from_archive(f)
        if 'kind' in f and str(f['kind'])=='fixed-centered-tensorsketch2-v1':
            from pong_fixed_pairwise import FixedPairwise,FrozenPairwiseReadout
            return FrozenPairwiseReadout(f['base'],FixedPairwise(f['hashes'],f['signs'],f['scale'],f['mean'],f['rms']),f['residual'])
        return FrozenRBFReadout(f['base'],FixedRBF(f['centers'],f['scale'],float(f['sigma'])),f['residual'])


def save_readout(path,base,basis,residual):
    with open(path,'wb') as f:np.savez(f,base=base,centers=basis.centers,scale=basis.scale,sigma=basis.sigma,residual=residual)


class FreshRBFEvaluator(FreshPongEvaluator):
    def __init__(self,readout,config):
        super().__init__(readout.base.weights,config);self.rbf_readout=readout;self.initial_learned_hash=self.learned_hash()

    def learned_hash(self):
        original=super().learned_hash()
        return original if not hasattr(self,'rbf_readout') else fingerprint((original,self.rbf_readout.frozen_parameters()))

    def _choose_policy(self):
        if hasattr(self.rbf_readout,'choose_encoded'):return self.rbf_readout.choose_encoded(self.latest_encoded,self.scaler)
        return self.rbf_readout.choose(*policy_features(self.latest_encoded,self.scaler))
