"""Transfer scalar coefficients learned on Pong to an arm with a new action basis."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import numpy as np

from tt3_arena import TT3Arena, heldout_cases
from tt3_continuous import SceneRecall
from tt3_flight import ForecastHistory, MotionEvidence
from tt3_learning import play_point
from tt3_motor import IntegratedTableLearner
from tt3_numeric import NumericMemory
from tt3_visual import RoleModel


class PortableMemory(NumericMemory):
    """The scalar gains are invariant to a change of visual basis. An offset
    fitted along a particular output axis has no signed direction in a new basis.
    """
    def __init__(self,*args,source_dimension=2,**kwargs):
        super().__init__(*args,**kwargs)
        self.source_dimension=source_dimension

    def goal(self,option,actor_position,*args,response_axis=None,**kwargs):
        axis=response_axis if len(actor_position)==self.source_dimension else None
        return super().goal(option,actor_position,*args,response_axis=axis,**kwargs)


class VisualPlayer(IntegratedTableLearner):
    def __init__(self,*args,forecast=False,recall=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.evidence=MotionEvidence() if forecast else None
        if recall:
            self.binder=SceneRecall(self.binder)
            self.stop_probing=lambda learner,binding: learner.binder.recalled and learner.motor.samples>=16

    def reset(self,option=None):
        super().reset(option)
        if self.evidence is not None:
            self.flight=ForecastHistory(self,self.evidence)


def start_memory(condition, source, rng):
    memory=PortableMemory.restore(source)
    if condition=='body_bias':
        return NumericMemory.restore(source)
    if condition in ('scratch','fresh_goal'):
        noise=np.stack([(rng.permutation(6)+rng.random(6))/3-1 for _ in range(3)],axis=1)
        return PortableMemory(noise*np.array([.11,1.4,1.4]))
    if condition=='reversed_geometry':
        memory.coefficients[:,1:]*=-1
    if condition=='fixed_extrapolation':
        return PortableMemory([[0.,1.,1.]])
    return memory


def run_match(job, *, arena_factory=TT3Arena):
    index,variant,roles,source,conditions,number,seed,stable,forecast=job
    schedule=heldout_cases(seed+index,number,variant=variant)
    out={}
    for condition in conditions:
        source_role=RoleModel.untrained() if condition=='scratch' else RoleModel(**roles)
        memory=start_memory(condition,source,np.random.default_rng(seed+index))
        learner=VisualPlayer(source_role,memory,seed=seed+index,forecast=forecast,recall=stable,learn_roles=True)
        points=[]
        tape=None
        for i,case in enumerate(schedule):
            view=schedule[0].view_seed if stable else case.view_seed
            arena=arena_factory(case.seed,view_seed=view,variant=variant)
            actions=[]
            def capture(frames, action, reward):
                actions.append(action.tolist())
            r=play_point(arena,learner,capture=capture if condition=='trained' and tape is None else None)
            summary={k:r[k] for k in ('legal_returns','contacts','rival_returns','points_for','points_against',
                                      'public_reward','frames','probes','first_binding')}
            summary.update(option_index=int(learner.option),seed=case.seed,view_seed=view)
            points.append(summary)
            if actions and r['legal_returns']>=2 and r['rival_returns']>=1 and r['points_for']:
                tape=dict(seed=case.seed,view_seed=view,variant=variant,extra_resets=1,actions=actions,
                          expected=arena.point_result(),point=i+1,match=index)
                if hasattr(arena, 'preparation_frames'):
                    tape['preparation_frames'] = arena.preparation_frames
        out[condition]=dict(points=points,memory=memory.state(),roles=asdict(learner.binder.model),replay=tape)
    return out


def summary(matches):
    stats={}
    for condition in matches[0]:
        phases={}
        for key,start,stop in [('opening',0,1),('first_eight',0,8),('later',8,None),('overall',0,None)]:
            rows=[p for match in matches for p in match[condition]['points'][start:stop]]
            phases[key]=dict(points=len(rows),landed=sum(p['legal_returns']>0 for p in rows),
                             returns=sum(p['legal_returns'] for p in rows),rival=sum(p['rival_returns'] for p in rows),
                             sustained=sum(p['legal_returns']>=2 for p in rows),wins=sum(p['points_for'] for p in rows),
                             probes=sum(p['probes'] for p in rows),seconds=sum(p['frames'] for p in rows)/60)
        stats[condition]=phases
    return stats


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('roles',type=Path)
    p.add_argument('source',type=Path)
    p.add_argument('out',type=Path)
    p.add_argument('--seed',type=int,default=7733191)
    p.add_argument('--matches',type=int,default=6)
    p.add_argument('--points',type=int,default=12)
    p.add_argument('--variant',choices=('standard','shifted','height'),default='standard')
    p.add_argument('--stable',action='store_true')
    p.add_argument('--forecast',action='store_true')
    p.add_argument('--conditions',nargs='+',default=['trained','scratch','fresh_goal','reversed_geometry','fixed_extrapolation'])
    p.add_argument('--workers',type=int,default=2)
    args=p.parse_args()
    roles=json.loads(args.roles.read_text()); roles=roles.get('role_model',roles)
    source=json.loads(args.source.read_text())
    jobs=[(i,args.variant,roles,source,args.conditions,args.points,args.seed,args.stable,args.forecast) for i in range(args.matches)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        matches=list(pool.map(run_match,jobs))
    stats=summary(matches)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(args.out,'wt') as f:
        json.dump(dict(seed=args.seed,variant=args.variant,stable=args.stable,forecast=args.forecast,source=source,
                       roles=roles,stats=stats,matches=matches),f,separators=(',',':'))
    for k,v in stats.items():print(k,v['first_eight'],v['overall'],flush=True)
