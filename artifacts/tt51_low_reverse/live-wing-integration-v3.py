"""A body path can be reserved without moving until an early strong RGB vote."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt54_live_wing as first
from .tt54_live_wing_v2 import Wing as RetainedWing

rp=first.rp


class Wing(RetainedWing):
    def reset(self):
        super().reset()
        self.permission=False;self.last_opposition=None;self.opposition=0

    def look(self,tick,x,ready):
        super().look(tick,x,ready)
        if not self.logs or self.logs[-1].get('action')!='far optical evidence' or self.logs[-1]['tick']!=tick:
            return
        lane,h,risk=self.forecast
        row=self.logs[-1]
        # Do not spend any motion on an unconfirmed reservation. A new late
        # guess cannot create a large body move with too little runway left.
        if not self.permission and row['gate']<=0. and .24<risk and -.29<lane<-.045 and h<1.035:
            self.permission=True;row.update(motion_authorized=True)

    def observed_incoming(self,tick,x,engine):
        if not self.permission:return None
        e=super().observed_incoming(tick,x,engine)
        if e is None:return None
        x,h=e
        opposite=x>-.060 or x<-.38 or h>1.073
        if not opposite:
            self.opposition=0;self.last_opposition=None
        else:
            self.opposition=self.opposition+1 if self.last_opposition is not None and tick-self.last_opposition<=4 else 1
            self.last_opposition=tick
            if self.opposition<2 and -.44<x<.13 and h<1.145:
                return None
        return e

    def command(self,tick,x,q,ref,plain,p):
        prior_decline=self.declined
        g=super().command(tick,x,q,ref,plain,p)
        if not self.permission:
            self.plan_phase=0;self.last_goal=None
            if self.logs and self.logs[-1]['tick']==tick:
                if self.logs[-1].get('action')=='toward acquired body':self.logs.pop()
                elif self.logs[-1].get('action')=='blocked by CAD step':
                    self.logs.pop();self.declined=prior_decline;self.recover=prior_decline
            return None
        # The first full move can start later than the original trajectory.
        # Look farther along it only while the ball is going out. Every real
        # imposed motor increment is still physically clipped and CAD-checked.
        if g is not None and x is not None and x[4]>.9 and self.logs and self.logs[-1].get('action')=='toward acquired body':
            seq=self.bank.rows[self.chosen]['next'];j=self.logs[-1]
            k=min(int(np.floor(self.level*(len(seq)-1))),j['phase']+19)
            faster=self.follow(q,ref,seq[k])
            if faster is not None:
                j.update(goal_phase=k,command=faster.tolist(),early_far_authorized=True)
                self.plan_phase=k;self.last_goal=faster.copy()
                return faster
        return g


def install(raw,checker,reference,*,enabled,frames):
    p=first.install(raw,checker,reference,enabled=False,frames=frames)
    if enabled:p.wing=Wing(rp._read(first.BANK),checker)
    return p


def run(path):
    make=first.install
    def factory(raw,checker,reference,*,enabled,frames):
        model=make(raw,checker,reference,enabled=False,frames=frames)
        if enabled:model.wing=Wing(rp._read(first.BANK),checker)
        return model
    with patch.object(first,'install',factory):first.run(path)
    d=rp._read(path);d['wing_extension_sha256']=rp._hash(Path(__file__));rp._write(path,d)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=first.root/'checked-exact.json.gz');a=p.parse_args()
    first.root.mkdir(parents=True,exist_ok=True);run(a.out)
