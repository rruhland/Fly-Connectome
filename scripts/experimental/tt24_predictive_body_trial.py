"""Test frozen motor control on novel blade targets and camera-only bouncing."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from .tt24_bounce import BounceArena
from .tt24_predictive_body import MODEL, OUT, PredictiveGrounding
from .tt24_public_body import load_public
from .tt24_world_arm import Grounding
from .tt24_world_body_play import advance_body
from .tt24_world_prior import WorldPrior
from .tt4_transfer_trial import _hash, _write

SENSOR=OUT/'public-body-fused.npz'
SOURCE=OUT/'world-prior-body.npz'
REACH_MODES=('kinematic','predictive')
REACH_SEEDS=tuple(range(366001,366017))
BOUNCE_SEEDS=tuple(range(367001,367017))
REACH_TICKS=156


def issue(controller, mode, goal, obs):
    if mode=='predictive':
        return controller.target(goal,obs)
    if mode=='kinematic':
        return controller.target(goal)
    return np.zeros(5)


def controller_for(mode,decoder,home,model):
    if mode=='predictive':
        return PredictiveGrounding(decoder,home,model)
    return Grounding(decoder,home)


def goal_sequence(seed,home,ticks,family):
    rng=np.random.default_rng(seed+496873)
    amplitude=np.array((.045,.046,.027,.135,.135))
    if family=='continuous':
        dt=np.arange(ticks)[:,None]/60.
        phase=rng.uniform(-np.pi,np.pi,5)
        tones=rng.uniform(.9,2.0,5)
        requested=amplitude*(.77*np.sin(2*np.pi*dt*tones+phase)
                             +.23*np.sin(2*np.pi*dt*tones*.47-phase))
        requested*=np.minimum(1.,dt*8.)
        requested[:,3:]+=home
        return requested
    anchors=[np.r_[np.zeros(3),home]]
    times=[0]
    while times[-1]<ticks-1:
        start=times[-1]
        goal=rng.uniform(-1.,1.,5)*amplitude
        goal[3:]+=home
        end=start+int(rng.integers(4,10))
        anchors.extend((goal,goal))
        times.extend((end,end+int(rng.integers(7,19))))
    return np.column_stack([np.interp(np.arange(ticks),times,axis)
                            for axis in np.asarray(anchors).T])


def diagnostic_pose(arena,reference):
    mujoco.mj_kinematics(arena._model,arena._data)
    world_xyz=arena._data.geom_xpos[arena._blade]
    normal=arena._data.geom_xmat[arena._blade].reshape(3,3)[:,2]
    return np.r_[world_xyz-reference,normal[:2]*np.sign(normal[2])]


def joint_reach(arena,decoder,mode,seed,regime,model,ticks):
    obs=arena.reset(seed=seed,split='held',regime=regime,render=False)
    observed=Grounding(decoder,obs['joints'])
    servo=controller_for(mode,decoder,obs['joints'],model)
    family='continuous' if seed%2 else 'ramp_and_hold'
    goals=goal_sequence(seed,observed.origin[3:],ticks,family)
    mujoco.mj_kinematics(arena._model,arena._data)
    reference=arena._data.geom_xpos[arena._blade].copy()
    public,hidden,actions,joints=[],[],[],[]
    for goal in goals:
        u=issue(servo,mode,goal,obs)
        obs=advance_body(arena,u)
        actions.append(u)
        joints.append(obs['joints'].copy())
        measured=observed.pose(obs['joints']).copy()
        measured[:3]-=observed.origin[:3]
        public.append(measured)
        hidden.append(diagnostic_pose(arena,reference))
    last=arena.reset(seed=seed,split='held',regime=regime,render=False)
    replay=[]
    for u in actions:
        last=advance_body(arena,u)
        replay.append(last['joints'].copy())
    if not np.array_equal(joints,replay):
        raise ValueError('reaching action tape differs from online proprioception')
    return dict(seed=seed,body=regime,family=family,goal=goals.tolist(),actions=np.asarray(actions).tolist(),
                settled_ticks=8,public_pose=np.asarray(public).tolist(),diagnostic_pose=np.asarray(hidden).tolist())


def reach_summary(rows):
    result={}
    for key in ('all','continuous','ramp_and_hold'):
        subset=[r for r in rows if key=='all' or r['family']==key]
        group={}
        for space in ('public_pose','diagnostic_pose'):
            residual=np.vstack([np.asarray(r[space])[r['settled_ticks']:]-np.asarray(r['goal'])[r['settled_ticks']:]
                                for r in subset])
            per_axis=np.sqrt(np.mean(residual**2,axis=0))
            group[space]=dict(xyz_mm=float(np.sqrt(np.mean(residual[:,:3]**2))*1000.),
                              normal_xy=float(np.sqrt(np.mean(residual[:,3:]**2))),
                              normalized=float(np.sqrt(np.mean((residual/np.array((.045,.045,.045,.29,.29)))**2))),
                              axis=per_axis.tolist())
        result[key]=group
    return result


def reaching(args):
    decoder=load_public(args.sensor).decoder
    arena=BounceArena(view='none')
    arena._model.geom_contype[arena._ball_geom]=0
    arena._model.geom_conaffinity[arena._ball_geom]=0
    doc=dict(schema='tt24-closed-loop-blade-pose-1',source='ball-free arm, no policy reward or target demonstrations',
             sensor_sha256=_hash(args.sensor),model_sha256=_hash(args.model),
             seeds=list(REACH_SEEDS),horizon=REACH_TICKS,
             exact_encoder_replay=True,geometry_scoring='held diagnostic only',regimes={})
    try:
        for regime in ('standard','changed'):
            doc['regimes'][regime]={}
            for mode in REACH_MODES:
                rows=[joint_reach(arena,decoder,mode,seed,regime,args.model,REACH_TICKS) for seed in REACH_SEEDS]
                metrics=reach_summary(rows)
                doc['regimes'][regime][mode]=dict(rows=rows,metrics=metrics)
                print(regime,mode,metrics['all'],flush=True)
                _write(args.out,doc)
    finally:
        arena.close()


def wrist_bounce(arena,prior,mode,seed,args):
    sensor=load_public(args.sensor)
    obs=arena.reset(seed=seed,split='held',regime='standard')
    servo=controller_for(mode,sensor.decoder,obs['joints'],args.model)
    actions=[]
    while True:
        goal=prior.target(sensor.observe(obs)) if mode!='hold' else None
        u=issue(servo,mode,goal,obs)
        actions.append(u.tolist())
        obs,_,done=arena.step(u)
        if done:
            break
    judge=arena.judge_metrics()
    arena.reset(seed=seed,split='held',regime='standard',render=False)
    for u in actions:
        arena.step(u,render=False)
    if arena.judge_metrics()!=judge:
        raise ValueError('wrist bounce action tape differs from the online judge')
    return dict(seed=seed,judge=judge,actions=actions)


def bouncing(args):
    prior=WorldPrior.load(args.source)
    arena=BounceArena(view='wrist',width=192,height=144,max_steps=300)
    doc=dict(schema='tt24-closed-loop-source-wrist-bounce-1',seeds=list(BOUNCE_SEEDS),
             rule='frozen source policy, same wrist calibration and encoders with two controllers',
             horizon=300,source=args.source.name,source_sha256=_hash(args.source),
             sensor_sha256=_hash(args.sensor),model_sha256=_hash(args.model),
             exact_judge_replay=True,modes={})
    try:
        for mode in (*REACH_MODES,'hold'):
            rows=[]
            for seed in BOUNCE_SEEDS:
                row=wrist_bounce(arena,prior,mode,seed,args)
                rows.append(row)
                print(mode,seed,row['judge']['longest_consecutive_rebounds'],flush=True)
                doc['modes'][mode]=rows
                _write(args.out,doc)
    finally:
        arena.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=('reach','bounce'))
    parser.add_argument('--out',type=Path)
    parser.add_argument('--sensor',type=Path,default=SENSOR)
    parser.add_argument('--model',type=Path,default=MODEL)
    parser.add_argument('--source',type=Path,default=SOURCE)
    args=parser.parse_args()
    if args.out is None:
        args.out=OUT/('predictive-body-reaching.json.gz' if args.task=='reach'
                      else 'predictive-body-bounce.json.gz')
    (reaching if args.task=='reach' else bouncing)(args)
