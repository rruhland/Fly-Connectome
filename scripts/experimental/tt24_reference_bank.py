"""Pair six independently trained native representation preferences on fresh physical releases."""
import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import gzip
import json

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_donor_trial import ROOT, Native
from .tt24_donor_assess import CHECKPOINT
from .tt24_posture import WidePosture
from .tt24_broad_donors import fixed_seeds
from .tt24_ready import episode

LOCK="7f5bf3958cc2581ba6b906e5e94fd011c7e8f4b8da9ea7c8b327b288bb12697d"


def qualify(job):
    replica,confirm=job
    prefix="reference-confirmation" if confirm else "reference-bank"
    path=ROOT/("tt24_pong_origin-confirmation-bank.json.gz" if confirm else "tt24_pong_origin-bank.json.gz")
    expected="bf96928059b032c635840e56ebe1d9d98c57b48bfdeeccce06655b874e693871" if confirm else LOCK
    if sha256(path.read_bytes()).hexdigest()!=expected:
        raise ValueError("native source changed before physical transfer")
    with gzip.open(path,"rt") as f:
        selections=json.load(f)["replicas"][replica]["selected"]
    with np.load(CHECKPOINT,allow_pickle=False) as d:
        w=d["weights"]
    with np.load(ROOT/"posture-source.npz",allow_pickle=False) as d:
        mu=d["center"]
    native=Native()
    groups={}
    try:
        for i,(body,n) in enumerate((("stock",6 if confirm else 8),("alternate",4))):
            regime="changed" if i else "standard"
            seeds=fixed_seeds(n,(64258329 if confirm else 105710322)+29*replica+i)
            arena=WidePosture(view="wrist",width=192,height=144,max_steps=360,arm=body)
            server=BounceArena(view="none",arm=body)
            try:
                rows={}
                for alpha in dict.fromkeys(selections.values()):
                    rows[str(alpha)]=[episode(arena,server,s,native,w,mu,"scalar",regime,alpha=alpha) for s in seeds]
                groups[body]=dict(seeds=seeds,by_alpha=rows)
                print(replica,body,{k:sum(r["run"] for r in rows[str(a)]) for k,a in selections.items()},flush=True)
            finally:
                arena.close()
                server.close()
        with gzip.open(ROOT/f"{prefix}-{replica}.json.gz","wt") as f:
            json.dump(dict(source_sha=expected,selections=selections,groups=groups),f,separators=(",",":"))
    finally:
        native.working.cleanup()


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--confirm",action="store_true")
    args=p.parse_args()
    with ProcessPoolExecutor(max_workers=4) as pool:
        list(pool.map(qualify,[(i,args.confirm) for i in range(12 if args.confirm else 6)]))
