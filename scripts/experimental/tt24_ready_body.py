"""Test an optically identified body calibration together with a remembered motor/visual reference."""
from concurrent.futures import ProcessPoolExecutor
import gzip
import json

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_broad_donors import fixed_seeds
from .tt24_compound_carry import templates
from .tt24_donor_assess import CHECKPOINT
from .tt24_donor_trial import ROOT, Native, FUSED_OUTPUT, load_public
from .tt24_posture import WidePosture
from .tt24_public_morph import load_morph
from .tt24_ready import episode
from .tt24_view_recall import descriptor

ROSTER=np.random.default_rng(40196287).permutation(np.repeat([0,1],16)).tolist()


def run(mode):
    native=Native()
    regular,wide=[],[]
    try:
        for body in ("stock","alternate"):
            regular.append(BounceArena(view="wrist",width=192,height=144,arm=body))
            wide.append(WidePosture(view="wrist",width=192,height=144,max_steps=360,arm=body))
        database=templates(regular,archived=True)
        with np.load(ROOT/"posture-source.npz",allow_pickle=False) as d:
            mu=d["center"]
        with np.load(CHECKPOINT,allow_pickle=False) as d:
            w=d["weights"]
        results=[]
        for body,seed in zip(ROSTER,fixed_seeds(len(ROSTER),1504290033),strict=True):
            record={}
            def make_sensor(obs):
                similarities=(database@descriptor(obs["rgb"])).max(axis=1)
                guess=int(np.argmax(similarities))
                selected=body if mode=="oracle" else 0 if mode in ("starting","familiar") else guess
                record.update(guess=guess,selected=selected,similarity=similarities.tolist())
                return load_morph() if selected else load_public(FUSED_OUTPUT)
            alpha=0. if mode in ("starting","optics") else 1.
            result=episode(wide[body],regular[body],seed,native,w,mu,"scalar",
                           "changed" if body else "standard",alpha=alpha,sensor_factory=make_sensor)
            results.append(result|record|dict(body=body))
        with gzip.open(ROOT/f"ready-body-{mode}.json.gz","wt") as f:
            json.dump(results,f,separators=(",",":"))
        print(mode,[(b,sum(x["run"] for x in results if x["body"]==b)) for b in (0,1)],flush=True)
    finally:
        for a in regular+wide:
            a.close()
        native.working.cleanup()


if __name__=="__main__":
    with ProcessPoolExecutor(max_workers=3) as pool:
        list(pool.map(run,("combined","familiar","optics","starting","oracle")))
