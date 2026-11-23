"""Calibrate the motor observer against actual encoder histories on previously seen bodies."""
import gzip
import json
import numpy as np
from .tt24_motor_recall import MotorObserver, example, sensor, STOCK
from .tt24_predict_witness import BODY_SEEDS, DATA as BODY_TRACE
from .tt24_world_arm import DIR


def read(path):
    with gzip.open(DIR/path,"rt") as src:
        return json.load(src)


def recover(arena, rows, split, regime):
    clips = []
    for row in rows:
        command = row.get("commands",row.get("actions"))
        obs=arena.reset(seed=row["seed"],split=split,regime=regime,render=False)
        states=[sensor(obs)]
        for issued in command:
            obs,_,_=arena.step(issued,render=False)
            states.append(sensor(obs))
        if "score" in row and arena.judge_metrics()["rebounds"]!=row["score"]:
            raise ValueError("nominal trajectory changed")
        clips.append((np.stack(states),np.asarray(command)))
    return clips


def fit(arenas):
    with np.load(BODY_TRACE,allow_pickle=False) as data:
        probes=[(data[f"body_obs_{s}"],data[f"body_action_{s}"]) for s in BODY_SEEDS]
    stock_play=read("public-carry.json.gz")["modes"]["carried"][:4]
    alt_play=read("public-morph.json.gz")["modes"]["recalibrated_public"]["episodes"][:4]
    stock_probe=read("public-body.json.gz")["data"]
    alt_probe=read("public-morph-calibration.json.gz")["data"]
    runs = [(recover(arenas[0],stock_probe,"practice","standard"),recover(arenas[0],stock_play,"held","standard")),
            (recover(arenas[1],alt_probe,"held","changed"),recover(arenas[1],alt_play,"held","changed"))]
    observers=[]
    charges=[]
    for calib,ordinary in runs:
        # A common background of generic excitation anchors seldom-used joints.
        background=probes+calib*3
        errors=[]
        for i,(states,commands) in enumerate(ordinary):
            observer=MotorObserver(background+3*(ordinary[:i]+ordinary[i+1:]))
            x,y=example(states,commands)
            errors.append((observer.design(x)@observer.weights-y)[:,:5])
        scale=np.maximum(.025,np.sqrt(np.mean(np.vstack(errors)**2,axis=0)))
        observers.append((MotorObserver(background+3*ordinary),scale))
        charges.append(dict(extra_nominal_game_frames=sum(len(u) for _,u in ordinary),
                            generic_probe_frames=sum(len(u) for _,u in calib)))
    return observers,charges
