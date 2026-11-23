"""Independently replay physical contacts and render the two-racket source game."""
import argparse
import ctypes
import gzip
import json
import math
from pathlib import Path

import mujoco
import numpy as np

from .tt24_donor_trial import DESIGNS, ROOT, Native
from .tt24_overhead import PartnerWorld


def replay(arena,seed,actions,frames=None):
    arena.reset(seed=seed,split="held",regime="changed",render=False)
    top=arena._model.geom("opponent_racket").id
    data,model=arena._data,arena._model
    pos=data.mocap_pos[0]
    ceiling=pos[2]
    home=data.geom_xpos[arena._blade].copy()
    speed_limit=.12 if arena.precise else .50
    prev=0
    touching=ended=False
    pairs=returns=0
    for tick,action in enumerate(actions):
        if ended:
            raise AssertionError("recorded extra actions after loss")
        if frames is not None and tick%2==0:
            frames.append((arena.render_external(),arena.render_wrist(),pairs,returns))
        for _ in range(8):
            q,v=arena._ball_position,arena._ball_velocity
            dq=action * (5.2*model.opt.timestep)
            data.ctrl[:5]=np.clip(data.ctrl[:5]+dq,arena._control_low[:5],arena._control_high[:5])
            for i in (0,1):
                aim=np.clip(q[i],home[i]-.025,home[i]+.025) if arena.precise else q[i]
                pos[i]+=np.clip(aim-pos[i],-speed_limit*model.opt.timestep,speed_limit*model.opt.timestep)
            pos[2]=ceiling+.008*math.sin(17*data.time)
            incoming=v[2]
            speed=math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
            arena._ball_force[:]=arena._drag_force_scale*speed*v
            mujoco.mj_step(model,data)
            hit=0
            for contact in data.contact:
                if contact.efc_address<0 or arena._ball_geom not in contact.geom:
                    continue
                other=next(g for g in contact.geom if g!=arena._ball_geom)
                if other==top:
                    hit=2
                elif other==arena._blade:
                    hit=1
                else:
                    ended=True
            if hit and not touching:
                if hit==prev or (hit==1 and incoming>-.20) or (hit==2 and (incoming<.20 or prev!=1)):
                    ended=True
                elif hit==1:
                    returns+=1
                else:
                    pairs+=1
                prev=hit
            touching=bool(hit)
            if ended or q[2]<.51 or q[2]>2.1 or abs(q[0])>.66 or q[1]<=-1.65 or q[1]>=-.33:
                ended=True
                break
    return [pairs,pairs,tick+1,returns]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--precision",action="store_true")
    cfg=parser.parse_args()
    prefix="overhead-precision" if cfg.precision else "overhead-source"
    with np.load(ROOT/(prefix+".npz")) as arch:
        w=np.ascontiguousarray(arch["weights"])
    with gzip.open(ROOT/(prefix+"-held.json.gz"),"rt") as f:
        report=json.load(f)
    native=Native(Path(__file__).with_name("tt24_overhead_native.cpp"))
    trace=native.so.tt24_overhead_trace
    trace.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]+[ctypes.c_void_p]*2
    trace.restype=None
    audited=[]
    try:
        for i,row in enumerate(report["source"]):
            world=PartnerWorld(native,DESIGNS[i],cfg.precision)
            try:
                for j,seed in enumerate(row["seeds"]):
                    batch=world.prepare([seed],"held")
                    actions=np.zeros((300,5))
                    stats=np.zeros(4)
                    trace(world.handle,w.ctypes.data,batch[0].ctypes.data,batch[1].ctypes.data,300,
                          stats.ctypes.data,actions.ctypes.data)
                    actions=actions[:int(stats[2])]
                    got=replay(world.arena,seed,actions)
                    if not np.array_equal(stats,row["scores"][0][j]) or not np.array_equal(stats,got):
                        raise RuntimeError((i,seed,stats,got))
                    audited.append(dict(body=i,seed=seed,actions=actions.tolist(),pairs=int(stats[0]),near=int(stats[3])))
            finally:
                world.close()
    finally:
        native.working.cleanup()
    with gzip.open(ROOT/(prefix+"-replay.json.gz"),"wt") as f:
        json.dump(dict(verified=audited),f,separators=(",",":"))
    print("Near and far physically verified",len(audited),"rollouts",flush=True)


if __name__=="__main__":
    main()
