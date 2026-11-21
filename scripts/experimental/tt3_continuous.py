"""Reuse earned visual identity evidence when the same physical table stays in view."""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from tt3_arena import TT3Arena, heldout_cases
from tt3_flight import ForecastLearner
from tt3_learning import ReturnMemory, TableLearner, play_point
from tt3_visual import Binding, RoleModel


class SceneRecall:
    """Scene-local appearance votes from the ordinary action/reward-trained binder.

    Votes accrue only from the underlying binder; an appearance recalled by this
    wrapper cannot vote for itself. A fresh scene without matching evidence falls
    back to the usual motor probes and role comparisons.
    """
    def __init__(self, binder):
        self.binder=binder
        self.votes=(Counter(),Counter())
        self.reset()

    @property
    def model(self):
        return self.binder.model

    def reset(self):
        self.binder.reset()
        self.recalled=False
        self.known=[]
        for counts in self.votes:
            ranked=counts.most_common(2)
            reliable=ranked and ranked[0][1]>=10 and (len(ranked)==1 or ranked[0][1]>1.6*ranked[1][1])
            self.known.append(ranked[0][0] if reliable else None)

    def observe_reward(self, score):
        self.binder.observe_reward(score)

    def update(self, frames, previous_action=None):
        current=self.binder.update(frames,previous_action)
        hypotheses=self.binder._scene.tracker._top.hypotheses
        seen={hypotheses[t.id].color:t for t in current.tracks if t.observed and t.id in hypotheses}
        for track, counts in zip((current.controlled,current.relevant),self.votes):
            if track is not None and track.observed and track.id in hypotheses:
                counts[hypotheses[track.id].color]+=1
        c,r=(seen.get(color) for color in self.known)
        self.recalled=c is not None and r is not None and c.id!=r.id
        if self.recalled:
            return Binding(c,r,current.tracks,current.controlled_evidence,current.relevant_evidence)
        return current


def build_agent(roles, memory, seed, *, recall=True, forecast=False):
    cls=ForecastLearner if forecast else TableLearner
    agent=cls(roles,memory,seed=seed,learn_roles=True)
    if recall:
        agent.binder=SceneRecall(agent.binder)
        agent.stop_probing=lambda learner,binding: learner.binder.recalled and learner.motor.samples>=16
    return agent


def aggreg(rows):
    return dict(points=len(rows),landed=sum(r['legal_returns']>0 for r in rows),
                returns=sum(r['legal_returns'] for r in rows),replies=sum(r['rival_returns'] for r in rows),
                sustained=sum(r['legal_returns']>=2 for r in rows),wins=sum(r['points_for'] for r in rows),
                probes=sum(r['probes'] for r in rows),frames=sum(r['frames'] for r in rows))


def one_match(job):
    idx,checkpoint,base_seed,count,kind,modes=job
    schedule=heldout_cases(base_seed+idx,count,variant=kind)
    common_view=schedule[0].view_seed
    runs={}
    for mode in modes:
        loaded=mode!='scratch'
        roles=RoleModel(**checkpoint['role_model']) if loaded else RoleModel.untrained()
        memory=ReturnMemory.restore(checkpoint['return_memory']) if loaded else ReturnMemory()
        agent=build_agent(roles,memory,base_seed+idx,recall=mode!='no_recall',forecast=mode=='forecast')
        rows=[]
        for case in schedule:
            arena=TT3Arena(case.seed,view_seed=common_view,variant=kind)
            rec=play_point(arena,agent)
            rows.append({key:rec[key] for key in ('option','legal_returns','rival_returns','points_for',
                                                   'points_against','probes','frames','first_binding')})
        runs[mode]=rows
    return dict(schedule=[asdict(c) for c in schedule],view_seed=common_view,runs=runs)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--seed',type=int,default=14059770)
    p.add_argument('--points',type=int,default=12)
    p.add_argument('--matches',type=int,default=6)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--variant',default='standard',choices=('height','standard','shifted'))
    p.add_argument('--modes',nargs='+',default=['carried','scratch','no_recall','forecast'])
    args=p.parse_args()
    src=json.loads(args.checkpoint.read_text())
    jobs=[(i,src,args.seed,args.points,args.variant,args.modes) for i in range(args.matches)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        matches=list(pool.map(one_match,jobs))
    summary={mode:{phase:aggreg([r for match in matches for r in match['runs'][mode][start:stop]])
                   for phase,start,stop in [('opening',0,1),('first_eight',0,8),('total',0,None)]}
             for mode in matches[0]['runs']}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(dict(seed=args.seed,variant=args.variant,source=src,
                                       summary=summary,matches=matches),separators=(',',':'))+'\n')
    print(json.dumps(summary))
