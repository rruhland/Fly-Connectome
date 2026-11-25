"""Make proposal mutations around issued reference paths, not impossible requested paths."""
from copy import deepcopy
import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt46_continuous_season as season


def executable(goals, reference, increment, lower, upper, micro):
    """Project onto stock control slew and position limits, with its own microsteps."""
    goals=np.asarray(goals)
    now=np.broadcast_to(reference,(len(goals),len(reference))).copy()
    projected=np.empty_like(goals)
    for t in range(goals.shape[1]):
        aim=np.clip(goals[:,t],lower,upper)
        u=np.clip((aim-now)/(micro*increment),-.8,.8)*increment
        for _ in range(micro):
            now=np.clip(now+u,lower,upper)
        projected[:,t]=now
    return projected


class Recentered(season.Player):
    """Same initial issued path for each seed, different center of its perturbations."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.recenter=False
        self.recenter_ready=False
        self.projected_shifts=[]

    def search(self,tick,x,q,ref,base):
        self.recenter_ready=self.recenter
        return super().search(tick,x,q,ref,base)

    def trial(self,goals,*,long=False,traces=False):
        if self.recenter_ready and traces and not long:
            if not isinstance(goals,list):
                raise ValueError('Stock initial seed list changed')
            self.recenter_ready=False
            e=self.engine
            p=e.predict
            inc=e.twin._joint_increments[p.axes]
            lo=e.twin._joint_low[p.axes]
            hi=e.twin._joint_high[p.axes]
            original=np.asarray(goals)
            projected=executable(original,e.reference,inc,lo,hi,p.micro)
            self.projected_shifts=np.max(np.abs(projected-np.clip(original,lo,hi)),axis=(1,2)).tolist()
            # Stock now perturbs these equivalent issued paths. No new proposal
            # rows, random numbers, physical limits, or candidate tests.
            goals[:]=list(projected)
        return super().trial(goals,long=long,traces=traces)


FIELDS=("best","tick","lease","next","notes","diag","metrics","in_rank","decision_tick",
        "reranks","rng","recenter","recenter_ready","projected_shifts")


def snapshot(actor):
    return {label:deepcopy(getattr(actor,label)) for label in FIELDS if hasattr(actor,label)}


def restore(actor,state):
    for label in FIELDS:
        if label in state:
            setattr(actor,label,state[label])
        elif hasattr(actor,label):
            delattr(actor,label)


class ParallelDiagnostic:
    """Compare both motor seed representations, keeping the stock live trace."""
    def search(self,tick,x,q,ref,base):
        run_alt=self.audit_alt and tick in self.watch
        before=snapshot(self) if run_alt else None
        super().search(tick,x,q,ref,base)
        if not run_alt or not self.witness or self.witness[-1]['tick']!=tick:
            return
        proof=self.witness[-1]
        if proof['offer']['success'] or proof['stock']['legal_trials']!=0:
            return
        after=snapshot(self)
        try:
            restore(self,before)
            self.recenter=True
            Recentered.search(self,tick,x,q,ref,base)
            note=self.notes[-1]
            winner=(np.asarray(self.best).tolist() if self.best is not None and self.tick==tick
                    and note.get('robust',0)>0 else None)
            proof['recenter_only']=dict(note=note,robust_winner=winner,prior_shift_rad=self.projected_shifts)
        finally:
            restore(self,after)
