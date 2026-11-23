"""Separate visual, motor-grounding, and source-dynamics errors in simulated transfer."""
import argparse
import gzip
import json
import numpy as np
import mujoco
from .tt24_bounce import BounceArena
from .tt24_bounce_ident import CleanState, _INC
from .tt24_world_prior import WorldPrior
from .tt24_world_arm import DIR, Grounding, perception


class PhysicalInverse:
    """Evaluation-only privileged local inverse, with the same physical motor limits."""
    def __init__(self,arena,joints):
        self.arena=arena
        self.home=joints.copy()
        self.issued=self.home.copy()
        self.origin=arena._data.geom_xpos[arena._blade].copy()
        self.scale=np.array((.045,.045,.045,.29,.29))

    def target(self,effect):
        a=self.arena
        q=a._data.qpos[a._joint_position[:5]]
        mat=a._data.geom_xmat[a._blade].reshape(3,3)
        normal=mat[:,2]*np.sign(mat[2,2])
        pos=a._data.geom_xpos[a._blade]-self.origin
        raw=np.r_[pos,normal[:2]]
        jt,jr=np.zeros((3,a._model.nv)),np.zeros((3,a._model.nv))
        mujoco.mj_comPos(a._model,a._data)
        mujoco.mj_jac(a._model,a._data,jt,jr,a._data.geom_xpos[a._blade],a._grip)
        dof=a._joint_velocity[:5]
        j=np.vstack((jt[:,dof],np.cross(jr[:,dof].T,normal).T[:2]))/self.scale[:,None]
        if np.linalg.norm(j)<1.e-8:
            raise ValueError('the grasped point has no actuated Jacobian')
        err=(effect-raw)/self.scale
        delta=np.linalg.solve(j.T@j+np.eye(5)*.24,j.T@err+.055*(self.home-q))
        dest=np.clip(q+np.clip(delta,-.22,.22),self.home-.55,self.home+.55)
        u=np.clip((dest-self.issued)/_INC,-.75,.75)
        self.issued+=_INC*u
        return u


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=51738174)
    p.add_argument('--arm',default='stock')
    p.add_argument('--count',type=int,default=16)
    p.add_argument('--horizon',type=int,default=450)
    args=p.parse_args()
    arena=BounceArena(view='none',arm=args.arm,max_steps=args.horizon)
    seeds=np.random.default_rng(args.seed).integers(1,2**31-1,args.count).tolist()
    prior=WorldPrior.load()
    regime='standard' if args.arm=='stock' else 'changed'
    doc=dict(arm=args.arm,seeds=seeds,horizon=args.horizon,privileged=True,modes={})
    try:
        for mode in ('clean_observation','clean_observation_and_inverse','hold'):
            rows=[]
            for seed in seeds:
                obs=arena.reset(seed=seed,split='held',regime=regime,render=False)
                sensor=CleanState(arena)
                g=PhysicalInverse(arena,obs['joints']) if mode.endswith('inverse') else Grounding(perception(args.arm!='stock').decoder,obs['joints'])
                actions=[]
                while True:
                    effect=prior.target(sensor.observe(obs))
                    action=np.zeros(5) if mode=='hold' else g.target(effect)
                    actions.append(action.tolist())
                    obs,_,done=arena.step(action,render=False)
                    if done:break
                judge=arena.judge_metrics()
                arena.reset(seed=seed,split='held',regime=regime,render=False)
                for u in actions:
                    arena.step(u,render=False)
                if arena.judge_metrics()!=judge:
                    raise ValueError('physical trajectory differs')
                rows.append(dict(run=judge['longest_consecutive_rebounds'],score=judge['rebounds'],ticks=len(actions),actions=actions))
            print(mode,[r['run'] for r in rows],flush=True)
            doc['modes'][mode]=rows
            with gzip.open(DIR/f'world-diagnostic-{args.arm}.json.gz','wt') as f:
                json.dump(doc,f,separators=(',',':'))
    finally:
        arena.close()


if __name__=='__main__':
    main()
