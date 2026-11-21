"""Test whether image-guided changes of racket height actually help legal returns."""
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from tt3_arena import HZ, TT3Arena, heldout_cases
from tt3_flight import ForecastLearner
from tt3_learning import ReturnMemory, TableLearner, play_point, relation_goal
from tt3_visual import RoleModel


class FixedHeight(ReturnMemory):
    """Negative control. Lock only the requested side-camera elevation after probing."""
    def goal(self, option, actor_position, mover_position, mover_velocity, ready_position, **kwargs):
        target = relation_goal(option,actor_position,mover_position,mover_velocity,ready_position,**kwargs)
        target[-1] = ready_position[-1]
        return target


def summarise(rows):
    return dict(points=len(rows),contacts=sum(r['contacts']>0 for r in rows),
                landed=sum(r['legal_returns']>0 for r in rows),total_legal=sum(r['legal_returns'] for r in rows),
                replies=sum(r['rival_returns'] for r in rows),repeated=sum(r['legal_returns']>=2 for r in rows),
                wins=sum(r['points_for'] for r in rows),seconds=sum(r['frames'] for r in rows)/HZ)


def one_course(job):
    idx, checkpoint, variant, count = job
    schedule = heldout_cases(450278911+idx,count,variant=variant)
    out = {}
    for mode in ('prior','learn_motion','fixed_height','parked'):
        cls = FixedHeight if mode=='fixed_height' else ReturnMemory
        memory = cls.restore(checkpoint['return_memory'])
        agent = (ForecastLearner if mode=='learn_motion' else TableLearner)(
            RoleModel(**checkpoint['role_model']),memory,seed=17732190+idx,learn_roles=True)
        rows = []
        for case in schedule:
            arena = TT3Arena(case.seed,view_seed=case.view_seed,variant=variant)
            height = []
            if mode=='parked':
                arena.reset(render=False)
                while not arena.done:
                    arena.step((0.,0.,0.),render=False)
                row=arena.point_result()
            else:
                def on_step(frames,action,reward):
                    if arena.frames>=16:  # exclude the matched starting probe
                        height.append(float(arena.paddle[2]))
                row=play_point(arena,agent,capture=on_step)
            rows.append({**{k:int(row[k]) for k in ('contacts','legal_returns','rival_returns','points_for','frames')},
                         'seed':case.seed,'goal_option':row.get('option'),'height_range':float(np.ptp(height)) if height else 0.})
        out[mode]=rows
    return out


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('checkpoint',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--variant',default='height',choices=('standard','height','shifted'))
    p.add_argument('--courses',type=int,default=4)
    p.add_argument('--points',type=int,default=12)
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    source=json.loads(args.checkpoint.read_text())
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        courses=list(executor.map(one_course,[(i,source,args.variant,args.points) for i in range(args.courses)]))
    stats={k:summarise([r for g in courses for r in g[k]]) for k in courses[0]}
    receipt=dict(variant=args.variant,source=source,courses=courses,summary=stats)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(receipt,separators=(',',':'))+'\n')
    print(json.dumps(stats))
