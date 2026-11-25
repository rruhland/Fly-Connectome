"""Measure repeated same-state action effects, then test a causal candidate preference."""

import argparse
from pathlib import Path

import numpy as np

from . import tt43_outgoing_coverage as old
from . import tt55_paired_afterstate as assay
from .tt4_transfer_trial import _hash, _read, _write


REPORT = {k: assay.ROOT / f"{k}-report.json" for k in assay.OUT}
MODEL = assay.ROOT / "preference.json"
RIDGE = 4.0


def robust_average(groups):
    x = np.asarray(groups,dtype=float)
    if not len(x):
        return None
    ix = np.random.default_rng(550825).integers(len(x),size=(12000,len(x)))
    values = x[ix].mean(axis=1)
    return dict(releases=len(x),mean=float(x.mean()),ci95=np.quantile(values,[.025,.975]).tolist())


def signs(l,r):
    x,y = np.asarray(l,dtype=float),np.asarray(r,dtype=float)
    keep = np.isfinite(x)&np.isfinite(y)
    x,y = x[keep],y[keep]
    active = (x!=0.)&(y!=0.)
    n = int(active.sum())
    return dict(releases=len(x),both_nonzero=n,
                same_sign=int(sum(np.sign(x[active])==np.sign(y[active]))),
                correlation=float(np.corrcoef(x,y)[0,1]) if len(x)>=3 and np.std(x)>0 and np.std(y)>0 else None)


def collect(cohort):
    plan,data = _read(assay.PLAN),_read(assay.OUT[cohort])
    if data["plan_sha256"]!=_hash(assay.PLAN) or len(data["rows"])>len(plan[cohort]):
        raise ValueError("Mismatched empirical outcome roster")
    rows=[]
    for record,p in zip(data["rows"],plan[cohort]):
        if record["pair_index"]!=p["pair_index"] or record["id"]!=p["id"]:
            raise ValueError("Study assignment order changed")
        a,b=record["branches"]
        if [r["seed"] for r in a]!=p["goals"] or [r["seed"] for r in b]!=p["goals"]:
            raise ValueError("Empirical pair lost its common random goals")
        if any(u["goal"]!=v["goal"] or not (u["prior_landing_verified"] and v["prior_landing_verified"])
               for u,v in zip(a,b,strict=True)):
            raise ValueError("Actual prior stroke label or common target differs")
        landed=np.array([np.array([int(q["landed"]) for q in r]) for r in (a,b)])
        present=np.array([np.array([int(q["answered"]) for q in r]) for r in (a,b)])
        z=np.array([np.array([q["geometry"][0] if q["geometry"] else np.nan for q in r]) for r in (a,b)])
        low=np.array([np.array([int(q["central_low"]) if q["central_low"] is not None else np.nan for q in r]) for r in (a,b)])
        good=np.isfinite(z).all(axis=0)
        by_slice = dict(first=list(range(0,3)),second=list(range(3,6)),all=list(range(6)))
        contrasts={}
        for name, ix in by_slice.items():
            both=[j for j in ix if good[j]]
            contrasts[name]=dict(physical_next_legal=float(np.diff(landed[:,ix],axis=0).mean()),
                                 offered=float(np.diff(present[:,ix],axis=0).mean()),
                                 height_m=float(np.diff(z[:,both],axis=0).mean()) if both else None,
                                 central_low=float(np.diff(low[:,both],axis=0).mean()) if both else None,
                                 jointly_observed=len(both))
        rows.append(dict(id=p["id"],contrasts=contrasts,scored=landed.sum(axis=1).tolist(),offered=present.sum(axis=1).tolist(),
                         answered_both=int(present.all(axis=0).sum()),
                         touch=[sum(q["touched"] for q in r) for r in (a,b)],
                         low_center=np.nansum(low,axis=1).astype(int).tolist(),
                         geometry_n=np.isfinite(z).sum(axis=1).tolist(),
                         extrapolated=[sum(q["line_kind"]=="extrapolated" for q in r) for r in (a,b)],
                         candidates=[v["decision"] for v in p["arms"]]))
    return rows,data["complete"]


def report(cohort):
    rows,complete=collect(cohort)
    modes=("physical_next_legal","offered","height_m","central_low")
    aggregate={}
    for name in modes:
        d=[r["contrasts"]["all"][name] for r in rows]
        u,v=([r["contrasts"][h][name] for r in rows] for h in ("first","second"))
        valid=[x for x in d if x is not None]
        repeated=[.25*(np.sign(a)*b+np.sign(b)*a) for a,b in zip(u,v,strict=True)
                  if a is not None and b is not None]
        aggregate[name]=dict(paired_effect=robust_average(valid),repeat=signs(u,v),
                             split_chosen_vs_uninformed=robust_average(repeated))
    aggregate["counts"]={k:np.asarray([r[k] for r in rows]).sum(axis=0).tolist() for k in
                         ("scored","offered","touch","low_center","geometry_n","extrapolated","answered_both")}
    _write(REPORT[cohort],dict(complete=complete,n=len(rows),source_sha256=_hash(assay.OUT[cohort]),
                               effects=aggregate,rows=rows))
    print(cohort,len(rows),aggregate,flush=True)


def values(candidates):
    return np.clip(np.array([q["target_errors"] for q in candidates])/old.SCALE,-6,6)


def fit(z,d):
    return np.linalg.solve(z.T@z + RIDGE*np.eye(z.shape[1]),z.T@d)


def comparison(rows,context):
    rows=[r for r in rows if all(r["candidates"])]
    choices=[r["candidates"] for r in rows]
    z=np.stack([np.diff(values(c),axis=0)[0] for c in choices])
    y=np.array([r["contrasts"]["all"]["physical_next_legal"] for r in rows])
    t=np.array([r["contrasts"]["all"]["central_low"] for r in rows],dtype=float)
    rank=[]
    quality=[]
    for c in choices:
        r,s=context.risk(np.array([q["target_errors"] for q in c]))
        rank.append(np.sign(r[0]-r[1]) if np.all(s) else 0.)
        quality.append(np.sign(c[0]["score"]-c[1]["score"]))
    return dict(z=z,y=y,t=t,rank=np.asarray(rank),quality=np.asarray(quality),rows=rows)


def learn():
    tr,tr_complete=collect("train")
    he,he_complete=collect("held")
    if not (tr_complete and he_complete) or {r["id"] for r in tr}&{r["id"] for r in he}:
        raise ValueError("Fit needs the completed, release-disjoint empirical pairs")
    context=old.Coverage(_read(assay.season.fresh.RISK))
    A,B=(comparison(r,context) for r in (tr,he))
    w=fit(A["z"],A["y"])
    ok=np.isfinite(A["t"])
    risk=fit(A["z"][ok],A["t"][ok])
    n=len(A["y"])
    cv=np.empty(n)
    for v in np.array_split(np.random.default_rng(55069).permutation(n),5):
        keep=np.ones(n,dtype=bool)
        keep[v]=False
        cv[v]=A["z"][v]@fit(A["z"][keep],A["y"][keep])
    def options(p,data):
        sign=np.sign(p)
        y=data["y"]
        return dict(closed_loop_vs_random=robust_average(.5*y*sign),
                    closed_loop_vs_existing_risk=robust_average(.5*y*(sign-data["rank"])),
                    closed_loop_vs_existing_quality=robust_average(.5*y*(sign-data["quality"])))
    predicted=B["z"]@w
    low=-B["z"]@risk
    metrics=dict(cv=options(cv,A),held=options(predicted,B),held_low_preference=options(low,B))
    # A full point is only justified if the paired actual score advantage over
    # the unchanged outgoing ranker is positive on disjoint releases.
    promote=bool(metrics["held"]["closed_loop_vs_existing_risk"]["ci95"][0]>0)
    recipe=dict(scale=old.SCALE.tolist(),weights=w.tolist(),central_low_weights=risk.tolist(),ridge=RIDGE,
                training=[r["id"] for r in A["rows"]],held=[r["id"] for r in B["rows"]],
                source_sha256={k:_hash(p) for k,p in assay.OUT.items()},metrics=metrics,deploy=promote,
                scope="Only the original accepted candidate's pre-contact target estimates on a common decision tick. "
                      "No native departure, geometry, trial seed or post-fork observation is a model input.")
    _write(MODEL,recipe)
    print("fit",len(A["y"]),"held",len(B["y"]),"deploy",promote,metrics,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("phase",choices=("held","train","learn"))
    a=p.parse_args()
    learn() if a.phase=="learn" else report(a.phase)
