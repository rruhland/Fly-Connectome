"""A separately frozen full-point course of ordinary playable measured launches."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt83_population_eval as tt
from .tt4_transfer_trial import _hash,_read,_write

ROOT=tt.ROOT
PLAN=ROOT/"playable.protocol.json"
OUT=ROOT/"playable.games.json.gz"
SOURCES=tuple(Path("artifacts")/(n+".json.gz") for n in ("tt52_practice_dose/live-from-12","tt52_practice_dose/live-from-48"))
N=4


def freeze(feeds):
    p=tt.read_plan()
    ban={r["id"] for r in p["cases"]}
    ban|={r["card"]["id"] for r in p["games"]}
    ban|={r["id"] for r in _read(tt.school.PLAN)["held"]}
    bank=_read(tt.opt.sensitivity.MODEL)
    ban|={r["id"] for r in bank["observations"] if "entry" in r}
    with np.load(tt.opt.few.MODEL) as f:
        ban.update(f["ids"].tolist())
    carry=_read(tt.season.CARRY)
    ban.update(c["id"] for c in carry["cards"][:len(carry["rows"])])
    for path in (tt.mpc.BANK,tt.course.PRACTICE):
        ban.update(r["card"]["id"] for r in _read(path.with_suffix(".json"))["practice"])
    offers={}
    evidence={}
    for path in SOURCES:
        a=_read(path)
        for c,r in zip(a["cards"],a["rows"]):
            if c["id"] in ban or c["opening"]!="random" or c["profile"]!="reach" or r["score"]["learner"]<1:
                continue
            offers[c["id"]]=c
            evidence.setdefault(c["id"],[]).append(r["score"]["learner"])
    rng=np.random.default_rng(831007)
    keys=rng.permutation(sorted(offers))[:N]
    if len(keys)!=N:
        raise ValueError("Not enough viable measured releases disjoint from all fitted paths")
    shots=tt.mpc.measured_launches(feeds)
    roster=[dict(card=dict(offers[k],seed=int(rng.integers(2**30)),reply_seed=int(rng.integers(2**30))),
                 shot=shots[k],exploratory_seed=int(rng.integers(2**30)), untreated_archive=evidence[k]) for k in keys]
    _write(PLAN,dict(ready=p["ready"],rounds=p["rounds"],games=roster,
                    sources={**p["sources"],**{str(x):_hash(x) for x in (*SOURCES,tt.PLAN)}},
                    role="Supplemental four preassigned one-launch points. There are no treated results in choosing this roster. Select only releases that previously yielded at least one legal ordinary return, excluding every release in the fitted scored movements, primary physical probes, primary game assignments, and old school qualification. New standard continuous opponent seeds. One independent complete point per condition and no resets."))
    print("additional playable TT83 games",[r["card"]["id"] for r in roster],flush=True)


def run():
    with patch.multiple(tt,PLAN=PLAN,GAMES=OUT):
        tt.games(N)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","run"))
    p.add_argument("--feeds",type=Path,default=Path("/tmp/tt25-rallies.json"))
    args=p.parse_args()
    freeze(args.feeds) if args.mode=="plan" else run()
