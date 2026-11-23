"""Single-stream reward-only practice after six demonstrated receiving blocks."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt25_nearby_lessons import features
from .tt25_pose_demonstrations import episode
from .tt25_reference_receiver import ReferenceReceiver
from .tt25_recorded_flight import OUTPUT


class Corrections:
    def __init__(self, keys, goals, scale):
        self.keys,self.goals,self.scale=keys,goals,scale
        self.successes=[[] for _ in goals]

    def choose(self, key):
        normalized=(key-self.keys)/self.scale
        origin=int(np.argmin(np.linalg.norm(normalized,axis=1)))
        successes=self.successes[origin]
        if not successes:
            return origin,self.goals[origin].copy()
        # Each successful joint displacement is a local example. Anchor the
        # original demonstration and shrink directions with little evidence.
        X=np.stack([r[0] for r in successes])
        Y=np.stack([r[1] for r in successes])
        distances=np.linalg.norm(X-normalized[origin],axis=1)
        weights=np.exp(-.5*(distances/1.5)**2)
        design=np.column_stack([np.ones(len(X)),X])
        ridge=np.eye(design.shape[1])*2.
        fitted=np.linalg.solve(design.T@(weights[:,None]*design)+ridge,design.T@(weights[:,None]*Y))
        delta=np.r_[1.,normalized[origin]]@fitted
        return origin, self.goals[origin]+np.clip(delta,-.16,.16)

    def learn(self, key, action):
        i,_=self.choose(key)
        self.successes[i].append(((key-self.keys[i])/self.scale,action-self.goals[i]))


def run(path):
    raw=path.read_bytes()
    if sha256(raw).hexdigest()!=json.loads((OUTPUT/"recorded-secondary-reference.json").read_text())["sha256"]:
        raise ValueError("Unexpected DeepMind data")
    shots={r["id"]:r for r in json.loads(raw)}
    first=_read(OUTPUT/"recorded-nearby-lessons.json.gz")
    recent=_read(OUTPUT/"recorded-block-confirmation.json.gz")
    exclude={r["id"] for r in json.loads((OUTPUT/"recorded-reach.json").read_text())["cases"]}
    exclude.update(r["id"] for group in recent["cohorts"] for r in group)
    remaining=[r for r in shots.values() if r["pos_y"]>0 and r["vel_y"]<0 and r["id"] not in exclude]
    values=np.stack([features(s) for s in remaining])
    nearby=[]
    for sid in first["donors"]:
        distance=np.linalg.norm((values-features(shots[sid]))/first["metric"],axis=1)
        accepted=0
        for i in np.argsort(distance):
            shot_id=remaining[i]["id"]
            if shot_id in exclude or distance[i]<=1e-5:
                continue
            nearby.append(shot_id)
            exclude.add(shot_id)
            accepted+=1
            if accepted==8:
                break
    unfiltered=np.random.default_rng(499873).choice([r["id"] for r in remaining if r["id"] not in exclude],32,replace=False).tolist()
    with np.load(OUTPUT/"recorded-pose-imitation.npz", allow_pickle=False) as f:
        keys,goals,scale=[f[k] for k in ("image_keys","joint_goals","scale")]
    model=Corrections(keys,goals,scale)
    rng=np.random.default_rng(311509)
    arena=ReferenceReceiver()
    history=[]
    try:
        for tick in range(192):
            # Only the environment uses the source launch. The learner gets the
            # camera signature and its own action before seeing the landing reward.
            donor=first["donors"][int(rng.integers(len(goals)))]
            shot=dict(shots[donor])
            width=(1,2,3)[tick//64]
            for prefix,sigma in [("pos",.012),("vel",.040),("w_vel",2.)]:
                for a,jitter in zip("xyz",rng.normal(0,width*sigma,3),strict=True):
                    shot[f"{prefix}_{a}"]+=jitter
            exploration=rng.normal(0,.05,5) if tick%4!=0 else np.zeros(5)
            issued=[]
            def practice_choice(key):
                _,proposal=model.choose(key)
                issued.append(proposal+exploration)
                return issued[-1]
            stats,key=episode(arena,shot,choose_goal=practice_choice)
            if stats["legal"]:
                model.learn(key,issued[-1])
            history.append(dict(tick=tick,donor=donor,legal=stats["legal"],contact=stats["contact"],
                                key=key.tolist(),issued=issued[-1].tolist(),width=width))
        checkpoints=dict(keys=keys,goals=goals,scale=scale,training_keys=np.array([r["key"] for r in history]),
                         training_goals=np.array([r["issued"] for r in history]),legal=np.array([r["legal"] for r in history]))
        np.savez_compressed(OUTPUT/"recorded-progressive.npz",**checkpoints)
        tests=[]
        for group,ids in [("original",first["donors"]),("nearby",nearby),("unfiltered",unfiltered)]:
            for sid in ids:
                pair={}
                for label,fn in [("adaptive",lambda key:model.choose(key)[1]),
                                 ("frozen",lambda key:goals[np.argmin(np.linalg.norm((keys-key)/scale,axis=1))])]:
                    stats,_=episode(arena,shots[sid],choose_goal=fn)
                    if stats["legal"]:
                        again,_=episode(arena,shots[sid],choose_goal=fn)
                        if again!=stats:
                            raise ValueError("Held success did not replay")
                    pair[label]=stats
                tests.append(dict(group=group,id=sid,results=pair))
        _write(OUTPUT/"recorded-progressive.json.gz",dict(training=history,tests=tests,practice_seed=311509))
        print("practice",sum(r["legal"] for r in history))
        for group in ("original","nearby","unfiltered"):
            print(group,{k:sum(r["results"][k]["legal"] for r in tests if r["group"]==group) for k in ("adaptive","frozen")})
    finally:
        arena.close()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("deepmind",type=Path)
    run(p.parse_args().deepmind)
