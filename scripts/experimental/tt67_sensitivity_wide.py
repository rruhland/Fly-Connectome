"""Preassign wider independent replies only where identified two-plane authority is appreciable."""
import argparse
from unittest.mock import patch

import numpy as np

from . import tt67_action_sensitivity as tt
from . import tt67_sensitivity_report as report
from .tt4_transfer_trial import _hash, _read, _write

ROOT = tt.ROOT / "wide"
PLAN, MODEL, SCOUT, OUT = (ROOT / p.name for p in (tt.PLAN, tt.MODEL, tt.SCOUT, tt.OUT))
AUTHORITY = .05
TAKE = 2


def freeze():
    prior, fit, scout = tt.design(), _read(tt.MODEL), _read(tt.SCOUT)
    if not fit["complete"] or not scout["complete"]:
        raise ValueError("Wide test needs finished native identification and original scouts")
    models = {x["id"]:x for x in fit["rows"]}
    cases, new_fit, new_scout = [], [], []
    for source, pool in zip(prior["cases"],scout["rows"],strict=True):
        basis = models[source["id"]]
        strength = np.linalg.svd(np.asarray(basis["J"])[[0,2]],compute_uv=False)[-1]
        if strength < AUTHORITY:
            continue
        original = np.asarray(source["state"])
        eligible=[]
        for j,v in enumerate(pool["candidates"]):
            if j in pool["retained"] or v["state"] is None:
                continue
            d=np.asarray(v["state"])-original
            shift=float(np.linalg.norm(d[:2]))
            if .065 <= shift <= .098 and abs(d[0])<.090 and abs(d[1])<.075 and abs(d[2])<.65:
                eligible.append((j,d,shift))
        if len(eligible)<TAKE:
            continue
        a = min(eligible,key=lambda x:abs(x[2]-.074))
        b = max((v for v in eligible if v[0]!=a[0]),key=lambda v:np.linalg.norm(v[1][:2]-a[1][:2]))
        cases.append(dict(source, identified_two_plane_m_per_rad=float(strength)))
        new_fit.append(basis)
        new_scout.append(dict(pool,retained=[a[0],b[0]]))
    if len(cases)<2:
        raise ValueError("Insufficient wide authority or disjoint physical goals")
    ROOT.mkdir(exist_ok=True, parents=True)
    p=dict(prior, cases=cases, authority_threshold=float(AUTHORITY), shifted_m=[.065,.098], retained=TAKE,
           scope="Exploratory wide follow-up, fixed separately before these four new receiving results. Retain sources by minimum x/z singular value of the source-only action sensitivity. Use independent held seeds unused by original assessment, with larger measured shifts, one near 74 mm and one spanning the farthest direction. Exact same published transfer law and control as main cohort. Both arms still replay the easy original anchors.")
    p["sources"] = dict(prior["sources"]) | {str(k):_hash(k) for k in (tt.PLAN,tt.MODEL,tt.SCOUT)}
    _write(PLAN,p)
    _write(MODEL,dict(protocol=_hash(PLAN),rows=new_fit,complete=True))
    _write(SCOUT,dict(protocol=_hash(PLAN),rows=new_scout,complete=True))
    print("wide",[(x["source"],x["retained"]) for x in new_scout],flush=True)


def run():
    with patch.multiple(tt, PLAN=PLAN, MODEL=MODEL, SCOUT=SCOUT, OUT=OUT):
        tt.held()
        with patch.object(report, "REPORT", ROOT/"report.json"):
            report.summarize()


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("plan","run"))
    a=p.parse_args()
    freeze() if a.mode=="plan" else run()
