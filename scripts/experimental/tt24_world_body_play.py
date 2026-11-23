"""Collect task-independent movement experience from the real simulated arm actuators."""
import argparse
import json
import numpy as np
import mujoco
from .tt24_bounce import BounceArena,SUBSTEPS
from .tt24_world_arm import DIR, Grounding, perception


def advance_body(arena,u):
    command=np.r_[np.clip(u,-1.,1.),0.]
    for _ in range(SUBSTEPS):
        arena._control+=command*arena._increments
        np.clip(arena._control,arena._control_low,arena._control_high,out=arena._control)
        mujoco.mj_step(arena._model,arena._data)
    return arena._observation(False)


def collect(args):
    rng=np.random.default_rng(args.seed)
    arena=BounceArena(view='none',arm=args.arm)
    # The ball is removed from contact and carries no control signal.
    arena._model.geom_contype[arena._ball_geom]=0
    arena._model.geom_conaffinity[arena._ball_geom]=0
    goals,measured,actions,joints=[],[],[],[]
    terminal=[]
    diagnostic_only=args.instrument and not args.public
    observed_geometry=[]
    true_geometry=[]
    try:
        for _ in range(args.runs):
            obs=arena.reset(seed=int(rng.integers(2**31-1)),split='practice',render=False,
                            regime='standard' if args.arm=='stock' else 'changed')
            decoder=perception(args.arm=='alternate',args.public).decoder
            plant=Grounding(decoder,obs['joints'])
            true_origin=arena._data.geom_xpos[arena._blade].copy()
            observed_run, true_run=[],[]
            desired=np.r_[np.zeros(3),plant.origin[3:]]
            rows=[[] for _ in range(4)]
            left=0
            for tick in range(args.ticks):
                if left==0:
                    goal=rng.uniform(-1,1,5)*np.array((.060,.060,.036,.16,.16))
                    left=int(rng.integers(3,15))
                left-=1
                desired=.24*desired+.76*goal
                effect=plant.pose(obs['joints'])
                effect[:3]-=plant.origin[:3]
                mujoco.mj_kinematics(arena._model,arena._data)
                normal=arena._data.geom_xmat[arena._blade].reshape(3,3)[:,2]
                true_run.append(np.r_[arena._data.geom_xpos[arena._blade]-true_origin,normal[:2]*np.sign(normal[2])])
                observed_run.append(effect.copy())
                u=plant.target(desired)
                for row,ob in zip(rows,(desired,effect,u,obs['joints']),strict=True):
                    row.append(ob.copy())
                obs=advance_body(arena,u)
            last=plant.pose(obs['joints'])
            last[:3]-=plant.origin[:3]
            if diagnostic_only:
                mujoco.mj_kinematics(arena._model,arena._data)
                normal=arena._data.geom_xmat[arena._blade].reshape(3,3)[:,2]
                last=np.r_[arena._data.geom_xpos[arena._blade]-true_origin,normal[:2]*np.sign(normal[2])]
            terminal.append(last)
            observed_geometry.append(observed_run)
            true_geometry.append(true_run)
            for seq,row in zip((goals,measured,actions,joints),rows,strict=True):
                seq.append(np.array(row))
    finally:
        arena.close()
    suffix='' if args.arm=='stock' else '-alternate'
    stem = 'world-arm-public-play' if args.public else 'world-arm-metrology' if args.instrument else 'world-arm-play'
    selected=true_geometry if diagnostic_only else measured
    np.savez_compressed(DIR/f'{stem}{suffix}.npz',goal=goals,measured=selected,
                        observed=observed_geometry,true_diagnostic=true_geometry,terminal=terminal,
                        command=actions,joints=joints,seed=args.seed,train_runs=args.runs-2,dt=1./60.)
    doc=dict(seed=args.seed,arm=args.arm,states='simulator pose metrology only' if diagnostic_only else 'calibrated encoder pose estimate before each command',
             public_sensor=args.public,game_reward_observations=0,ball_contacts=0,runs=args.runs,ticks_per_run=args.ticks,
             total_seconds=args.runs*args.ticks/60.,train_runs=args.runs-2,held_runs=2,
             peak_to_peak_effect=np.mean(np.ptp(selected,axis=1),axis=0).tolist(),
             peak_to_peak_goal=np.mean(np.ptp(goals,axis=1),axis=0).tolist())
    (DIR/f'{stem}{suffix}.json').write_text(json.dumps(doc,indent=2)+'\n')
    print(doc,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['stock','alternate'],default='stock')
    p.add_argument('--public',action='store_true')
    p.add_argument('--instrument',action='store_true')
    p.add_argument('--runs',type=int,default=8)
    p.add_argument('--ticks',type=int,default=96)
    p.add_argument('--seed',type=int,default=552917)
    collect(p.parse_args())
