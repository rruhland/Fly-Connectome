"""Apply a scalar learned in visual native Pong to the reference coordinates of a five-joint arm."""
import argparse
import gzip
import json

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_broad_donors import fixed_seeds
from .tt24_donor_assess import CHECKPOINT
from .tt24_donor_trial import ROOT, Native
from .tt24_posture import WidePosture
from .tt24_ready import episode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native",type=float,required=True)
    parser.add_argument("--shuffled",type=float,required=True)
    parser.add_argument("--course",choices=["fixed_stock","fixed_changed","roving"],required=True)
    cfg=parser.parse_args()
    i=("fixed_stock","fixed_changed","roving").index(cfg.course)
    native=Native()
    probabilities={"pong":cfg.native,"shuffled":cfg.shuffled,"neutral":.5,"starting_pose":0.,"familiar_pose":1.}
    body="alternate" if i==1 else "stock"
    regime="changed" if i==1 else "standard"
    server=(WidePosture if i==2 else BounceArena)(view="none",arm=body)
    arena=WidePosture(view="wrist",width=192,height=144,max_steps=360,arm=body)
    seeds=fixed_seeds(20 if i==0 else 12,50244611+i)
    try:
        with np.load(CHECKPOINT,allow_pickle=False) as data:
            w=data["weights"]
        with np.load(ROOT/"posture-source.npz",allow_pickle=False) as data:
            mu=data["center"]
        by_alpha={}
        for alpha in dict.fromkeys(probabilities.values()):
            rows=[episode(arena,server,s,native,w,mu,"scalar",regime,alpha=alpha) for s in seeds]
            by_alpha[str(alpha)]=rows
            print(cfg.course,alpha,sum(r["run"] for r in rows),flush=True)
        with gzip.open(ROOT/f"reference-transfer-{cfg.course}.json.gz","wt") as f:
            json.dump(dict(probabilities=probabilities,by_alpha=by_alpha,seeds=seeds,physical_replay=True),f,separators=(",",":"))
    finally:
        arena.close()
        server.close()
        native.working.cleanup()


if __name__=="__main__":
    main()
