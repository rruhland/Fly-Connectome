"""Approved fixed D/F release experiment; reference execution only."""
import torch
from signed_kinetics import AreaMatchedKineticsNetwork


class DepressionNetwork(AreaMatchedKineticsNetwork):
    facilitating=False

    def __init__(self,*args,unit_release=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.unit_release=unit_release
        predictive=(self.pathways==1).nonzero().flatten()
        self.release_index=torch.full((self.e,),-1,dtype=torch.long,device=self.device)
        self.release_index[predictive]=torch.arange(len(predictive),device=self.device)
        self.release_state=torch.full((self.batch,len(predictive)),0. if self.facilitating else 1.,device=self.device)
        self.release_last=torch.zeros_like(self.release_state,dtype=torch.long)
        self.arrival_keys=torch.empty(0,dtype=torch.long,device=self.device)
        self.arrival_gains=torch.empty(0,device=self.device)

    def _arrivals(self):
        env,edges=super()._arrivals()
        mask=self.pathways[edges]==1
        pe,ee=env[mask],edges[mask]
        indices=self.release_index[ee]
        elapsed=(self.step_index-self.release_last[pe,indices])*self.config.dt
        decay=torch.exp(-elapsed/.100)
        old=self.release_state[pe,indices]
        if self.facilitating:
            before=old*decay
            after=before+.5*(1-before)
            gains=after/.5
        else:
            before=1-(1-old)*decay
            gains=before
            after=.5*before
        self.release_state[pe,indices]=after
        self.release_last[pe,indices]=self.step_index
        keys=pe*self.e+ee
        order=torch.argsort(keys)
        self.arrival_keys=keys[order]
        self.arrival_gains=torch.ones_like(gains) if self.unit_release else gains[order]
        return env,edges

    def visual_arrival_impulse(self,environments,edges):
        base=self.visual_impulse(edges)
        if not len(self.arrival_keys):return base
        keys=environments*self.e+edges
        positions=torch.searchsorted(self.arrival_keys,keys).clamp(max=len(self.arrival_keys)-1)
        # Nonarriving live eligibility entries have zero arrival count. Their old
        # contributions are never rescaled by the most recent release value.
        gain=torch.where(self.arrival_keys[positions]==keys,self.arrival_gains[positions],1.)
        return base*gain


class FacilitationNetwork(DepressionNetwork):
    facilitating=True
