"""Ground a policy from abstract source worlds in the wrist-only physical arm."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np

from .tt24_bounce import BounceArena
from .tt24_bounce_ident import MotionFit, WristDecoder, WristPupil, WristState, _INC, _features, _restore
from .tt24_metric_transfer import ResidualRig

DIR=Path('artifacts/tt24')
SOURCE=DIR/'bounce-ident-pupil.npz'
PUBLIC=DIR/'metric-transfer-map.npz'


def perception(alternate, public=False, sensor_file=None):
    if public or sensor_file is not None:
        from .tt24_public_body import load_public
        return load_public(sensor_file or DIR/"public-body.npz")
    # Load only the calibration arrays. The target's demonstrated motor readout
    # is reserved for the explicit round-trip diagnostic.
    with np.load(SOURCE,allow_pickle=False) as checkpoint:
        decoder=_restore(WristDecoder,checkpoint,'decoder_')
        motion=_restore(MotionFit,checkpoint,'motion_')
    if alternate:
        with np.load(PUBLIC,allow_pickle=False) as data:
            if data['source_sha256'].item()!=hashlib.sha256(SOURCE.read_bytes()).hexdigest():
                raise ValueError('wrong sensory source')
            rig=ResidualRig.__new__(ResidualRig)
            rig.source=decoder
            rig.default=decoder.default
            rig.image_height=getattr(decoder,"image_height",144)
            for name in ('center','rotation','translation','blade_delta'):
                setattr(rig,name,data[name])
            decoder=rig
            motion.weights=data['motion_weights']
    return WristState(decoder,motion)


class Grounding:
    """Damped local inverse of a learned sensor-to-joint relationship."""
    def __init__(self,decoder,home):
        self.decoder=decoder
        self.home=home.astype(float).copy()
        self.issued=self.home.copy()
        self.origin=self.pose(self.home)
        self.scale=np.array((.045,.045,.045,.29,.29))
        self.steps=np.eye(5)*.003

    def pose(self,q):
        return self.decoder.body(q)[12:17]

    def jacobian(self,q):
        return np.stack([(self.pose(q+s)-self.pose(q-s))/.006 for s in self.steps],axis=1)

    def target(self,effect):
        desired=np.r_[self.origin[:3]+np.clip(effect[:3],-.085,.085),np.clip(effect[3:],-.70,.70)]
        q=self.issued.copy()
        # Recompute for the issued posture; all estimates come from the public
        # fitted calibration and are independent of MuJoCo's kinematics at test.
        for _ in range(2):
            j=self.jacobian(q)/self.scale[:,None]
            residual=(desired-self.pose(q))/self.scale
            error=j.T@residual + .055*(self.home-q)
            step=np.linalg.solve(j.T@j+np.eye(5)*.24,error)
            q=np.clip(q+np.clip(step,-.14,.14),self.home-.55,self.home+.55)
        u=np.clip((q-self.issued)/_INC,-.75,.75)
        self.issued+=_INC*u
        return u


def trial(arena,seed,mode,prior,alternate,public,sensor_file=None):
    sensor=perception(alternate,public,sensor_file)
    regime='changed' if alternate else 'standard'
    obs=arena.reset(seed=seed,split='held',regime=regime)
    ground=Grounding(sensor.decoder,obs['joints'])
    diagnostic=WristPupil.load(SOURCE) if mode in ('roundtrip','direct','carried') else None
    if mode=='carried':
        diagnostic.sensor=sensor
    rng=np.random.default_rng(2351)
    permutation=rng.permutation(5)
    tape=[]
    while True:
        if diagnostic is not None:
            if mode in ('direct','carried'):
                u=diagnostic.command(obs)
            else:
                x=_features(sensor.observe(obs),obs['joints'],ground.home)
                # Ask the demonstrated head for its sensory goal, then throw
                # out its joint commands and independently recover actuators.
                qgoal=ground.home+diagnostic.actor.target(x)
                effect=ground.pose(qgoal)
                effect[:3]-=ground.origin[:3]
                u=ground.target(effect)
        elif mode=='hold':
            u=np.zeros(5)
        elif mode=='anchor':
            u=ground.target(np.r_[np.zeros(3),ground.origin[3:]])
        else:
            state=sensor.observe(obs)
            if mode=='centered':
                state[9:11]-=ground.origin[3:]
            target=np.asarray(prior.target(state),dtype=float).copy()
            if mode=='centered':
                target[3:]+=ground.origin[3:]
            if target.shape!=(5,) or not np.isfinite(target).all():
                raise ValueError('source policy returned invalid target')
            if mode=='scrambled':
                scales=np.array((.06,.06,.06,.30,.30))
                target=scales*(target/scales)[permutation]
            u=ground.target(target)
        tape.append(u.tolist())
        obs,_,done=arena.step(u)
        if done:break
    judge=arena.judge_metrics()
    arena.reset(seed=seed,split='held',regime=regime,render=False)
    for u in tape:
        arena.step(u,render=False)
    if arena.judge_metrics()!=judge:
        raise ValueError('motor tape differs')
    return dict(seed=seed,run=judge['longest_consecutive_rebounds'],score=judge['rebounds'],ticks=len(tape),actions=tape)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('out',type=Path)
    p.add_argument('--source',type=Path,default=DIR/'world-prior.npz')
    p.add_argument('--modes',nargs='+',default=['prior','scrambled','anchor','hold'])
    p.add_argument('--seed',type=int,default=60021783)
    p.add_argument('--count',type=int,default=12)
    p.add_argument('--horizon',type=int,default=450)
    p.add_argument('--public',action='store_true')
    p.add_argument('--sensor-file',type=Path,help='Public wrist calibration to load for this comparison.')
    p.add_argument('--arm',default='stock',choices=['stock','alternate'])
    args=p.parse_args()
    prior=None
    if any(m in ('prior','scrambled','centered') for m in args.modes):
        from .tt24_world_prior import WorldPrior
        prior=WorldPrior.load(args.source)
    arena=BounceArena(view='wrist',width=192,height=144,arm=args.arm,max_steps=args.horizon)
    seeds=np.random.default_rng(args.seed).integers(1,2**31-1,args.count).tolist()
    doc=dict(arm=args.arm,public_sensor=args.public or args.sensor_file is not None,
             public_calibration=args.sensor_file.name if args.sensor_file else None,seeds=seeds,horizon=args.horizon,verified=True,modes={},source=args.source.name if prior else None)
    try:
        for mode in args.modes:
            rows=[trial(arena,s,mode,prior,args.arm=='alternate',args.public,args.sensor_file) for s in seeds]
            print(mode,[r['run'] for r in rows],flush=True)
            doc['modes'][mode]=rows
            with gzip.open(args.out,'wt') as f:
                json.dump(doc,f,separators=(',',':'))
    finally:
        arena.close()


if __name__=='__main__':
    main()
