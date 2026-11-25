"""Rewarded preparation continuation indexed by own body state, not source tick.

The training starts are a disclosed low-practice scaffold. This module does not
decide that a previously unseen outgoing shot will become a low incoming ball.
"""
from copy import deepcopy
import numpy as np


class BodyContinuation:
    def __init__(self,packet,cuts):
        self.rows=[]
        for sample,cut in zip(packet['rows'],cuts,strict=True):
            if sample['reward']!='physical legal and answered':continue
            rows=[r for r in sample['observations'] if 4030<=r['tick']<=cut]
            if len(rows)<5:continue
            self.rows.append(dict(name=sample['name'],q=np.array([r['q'] for r in rows]),
                ref=np.array([r['ref'] for r in rows]),next=np.array([r['next_ref'] for r in rows])))
        self.reset()

    def reset(self):
        self.selected=None;self.phase=None;self.finished=False;self.records=[];self.hold=None

    def command(self,q,ref):
        if self.finished:return None if self.hold is None else self.hold.copy()
        q=np.asarray(q);ref=np.asarray(ref)
        possible=[]
        for i,row in enumerate(self.rows):
            if self.selected is not None and i!=self.selected:continue
            first=max(0,self.phase-1) if self.phase is not None else 0
            last=min(len(row['q']),self.phase+5) if self.phase is not None else len(row['q'])
            distance=np.mean(((row['q'][first:last,:5]-q[:5])/.12)**2,axis=1)
            distance+=np.mean(((row['ref'][first:last,:5]-ref[:5])/.16)**2,axis=1)
            at=first+int(np.argmin(distance));possible.append((float(distance[at-first]),i,at))
        if not possible:return None
        distance,index,phase=min(possible)
        if distance>3.:self.finished=True;return None
        self.selected=index;self.phase=phase
        row=self.rows[index]
        if phase>=len(row['q'])-1:
            self.finished=True;self.hold=row['next'][-1].copy();return self.hold.copy()
        command=row['next'][phase].copy()
        self.records.append(dict(source=row['name'],body_phase=phase,body_distance=distance,
            q=q.tolist(),ref=ref.tolist(),next_ref=command.tolist()))
        return command


def earlier_by_body(sample,cut,radius):
    rows=sample['observations'];current=min(rows,key=lambda r:abs(r['tick']-cut))
    q=np.asarray(current['q'])
    eligible=[r for r in rows if r['tick']<cut and np.max(abs(np.asarray(r['q'])-q))>=radius]
    if not eligible:raise ValueError('No observed actual teacher body at requested distance')
    return cut-eligible[-1]['tick']
