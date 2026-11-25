"""Screen actual opponent trajectories for a common readiness, not a preferred strike plane."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .tt4_transfer_trial import _hash, _read, _write
from .tt31_ready import centered
from .tt37_opponent import arena
from .tt40_low_feasibility import Faces
from .tt41_shallow_model import Chart, impulse
from .tt43_stance import OUT, RATE, ROOT, minimax

DEPTHS = (-1.08, -1.15, -1.24, -1.34, -1.43, -1.52, -1.61)
PITCH = ((3.9, .03), (4.4, .15), (4.8, .25))
RADIUS = .071  # empirical support, not the disc's full collision radius


def geometry(chart, row, with_replay=True):
    """Enumerate clocked contact poses on the actual unconstrained post-bounce flight."""
    ball = row["table"]
    flight = chart.guard.a.partner.flight
    path = flight.path(ball["ball"], ball["velocity"], ball["spin"], 1, bounced=True, horizon=.65)
    far_elapsed = ball["time"] - row["far"]["time"]
    candidates = path_options(chart, path, far_elapsed, seed=np.asarray(row["joints_rad"]),
                              signed_seed=int(np.sign(row["face_alignment"])))
    if with_replay:
        # Preserve the known physical solution even if the coarse depth/normal grid misses it.
        candidates.append(dict(q=row["joints_rad"], face=int(np.sign(row["face_alignment"])),
                               eta=row["clocks"]["rebound"], far=row["clocks"]["far"],
                               p=row["xyz"], replay=True))
    return candidates


def path_options(chart, path, phase=0., *, seed=None, signed_seed=0, quick=False, depths=DEPTHS):
    positions = []
    if len(path):
        for y in depths:
            if path[:, 2].min() <= y <= path[:, 2].max():
                positions.append(path[np.argmin(abs(path[:, 2]-y))])
    ik = chart.ik
    candidates = []
    for sample in positions:
        t, p, v = sample[0], sample[1:4], sample[4:7]
        if p[2] < .827 or p[2] > 1.21 or abs(p[0]) > .47:
            continue
        for pace, pitch in (PITCH[1:2] if quick else PITCH):
            n, _ = impulse(p, v, pace, pitch)
            center = p - .025 * n
            for face in (-1, 1):
                idx = np.atleast_1d(chart.tree.query(np.r_[center, face*n] / [ .14,.13,.105,.45,.45,.45], k=12)[1])
                q0 = chart.raw[idx[0]]
                separate = next((chart.raw[k] for k in idx[1:] if np.max(abs(chart.raw[k]-q0)) > .38), None)
                seeds = (q0,) if separate is None else (q0, separate)
                if seed is not None and face in (signed_seed,) and not quick:
                    seeds = (*seeds, seed)
                for guess in (seeds[:1] if quick else seeds):
                    q, (c, axis, *_) = ik.fit(center, face*n, guess)
                    axis = face * axis
                    d = p-c
                    axial = d @ axis
                    radius = np.linalg.norm(d - axial*axis)
                    if not (.009 < axial < .040 and radius < RADIUS and n @ axis > .925 and chart.guard.clear(q)):
                        continue
                    if any(old["face"]==face and abs(old["eta"]-t) < .003
                           and np.max(abs(np.asarray(old["q"])-q)) < .09 for old in candidates):
                        continue
                    candidates.append(dict(q=q.tolist(), face=face, eta=float(t), far=float(t+phase),
                                           p=p.tolist(), v=v.tolist(), radius=float(radius)))
    return candidates


def joint_coverage(hubs, banks, cue, option=lambda _:True):
    """Best angular travel slack across bank poses for each arc and readiness."""
    score = np.full((len(hubs),len(banks)), -np.inf)
    for i, group in enumerate(banks):
        kept = [r for r in group if option(r)]
        if kept:
            q = np.array([r["q"] for r in kept])
            clock = np.array([r[cue] for r in kept])
            score[:,i] = np.max(clock - np.max(abs(np.asarray(hubs)[:,None,:] - q[None,:,:]),axis=2)/RATE, axis=1)
    return score


def exact_routes(chart, q, banks, cue, option=lambda _:True):
    out = []
    for group in banks:
        best = sorted(((a[cue]-np.max(abs(q-a["q"]))/RATE, a) for a in group if option(a)),
                      key=lambda r:r[0], reverse=True)
        row = next(((float(s),m) for s,m in best if chart.guard.route(q,np.array(m["q"]),steps=10)),
                   (None, None))
        out.append(dict(slack=row[0], endpoint=row[1]))
    return out


def study(source, output):
    d = _read(source)
    home = np.asarray(d["boot"])
    atlas_path = ROOT / "tt41/shallow-atlas.json.gz"
    atlas = _read(atlas_path)
    regular, low = d["regular"], d["low"]
    # 38 demonstrated normals plus three distinct low replies.
    negative = [x for x in low if x["face"] == "negative"]
    rows = regular + negative
    with centered(home):
        a = arena(detail="cad")
        try:
            chart = Chart(a, home, atlas["seeds"][0], atlas=atlas)
            banks = []
            for row in rows:
                options = geometry(chart, row)
                banks.append(options)
                print(row["group"],row["leg"],row["id"],len(options),
                      sum(o["face"]>0 for o in options),flush=True)
                if len(banks) % 8 == 0:
                    _write(output,dict(complete=False,banks=banks))
            # Draw a dense deterministic bank near the analytic witness center plus broad hubs
            # including shallow readiness from prior training. Refine finalist geometry by CAD.
            mm = minimax([w["joints_rad"] for w in rows], [w["clocks"]["rebound"] for w in rows])
            ik = Faces(a)
            q0 = np.asarray(mm["ready"])
            s = qmc.Sobol(5, scramble=True, seed=430194).random_base2(12)
            alt = qmc.scale(s[:2048], np.maximum(ik.low, q0-[.95, .82, .85, .94, .95]),
                            np.minimum(ik.high, q0+[.95, .82, .85, .94, .95]))
            coarse = qmc.scale(s[2048:], ik.low, ik.high)
            atlas_hubs = np.asarray([h["q"] for h in atlas["hubs"]])
            phases = np.linspace(0.,1.,20)[:,None]
            bridges = [(1.-phases)*home[:5]+phases*np.asarray(w["joints_rad"]) for w in negative]
            hubs = np.vstack([home[:5], q0, atlas_hubs, alt, coarse, *bridges])
            predicates = {"either": lambda _:True, "negative": lambda p:p["face"] == -1,
                          "positive": lambda p:p["face"] == 1}
            routes = {}
            for name, fn in predicates.items():
                for cue in ("far","eta"):
                    score = joint_coverage(hubs, banks, cue, fn)
                    # Prioritize floor (no sacrificed group), then median, then body distance.
                    worst=np.min(score,axis=1)
                    if not np.isfinite(worst).any():
                        fallback = np.sum(np.isfinite(score),axis=1)
                        merit = np.mean(np.maximum(score,-1.),axis=1)
                    else:
                        fallback = worst
                        merit = np.mean(score,axis=1)
                    order = np.lexsort((np.max(abs(hubs-home[:5]),axis=1), -merit, -fallback))
                    results=[]
                    for label, choices in (("stock", np.array([0])), ("common",order)):
                        for idx in choices:
                            q=hubs[idx]
                            if not chart.guard.route(home[:5], q, steps=16):
                                continue
                            actual=exact_routes(chart,q,banks,cue,fn)
                            if label=="common" and any(e["slack"] is None for e in actual) and len(results)>1:
                                continue
                            results.append(dict(name=label,q=q.tolist(),routes=actual))
                            break
                    routes[f"{name}_{cue}"]=results
                    print(name,cue,[(r["name"],round(min((e["slack"] for e in r["routes"] if e["slack"] is not None),
                                                       default=-2.),3),len([e for e in r["routes"] if e["slack"] is not None])) for r in results],flush=True)
            out=dict(complete=True,schema="tt43-contact-chart-1", source_sha256=_hash(source), atlas_sha256=_hash(atlas_path),
                     depths=DEPTHS,pitch=PITCH,radius=RADIUS,banks=banks,routes=routes,
                     scope="Finite geometric proposal census on actual native post-bounce flights, not legal shot validation.")
            _write(output,out)
        finally:
            a.close()


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--witnesses",type=Path,default=OUT/"witness-envelope.json.gz")
    p.add_argument("--out",type=Path,default=OUT/"chart.json.gz")
    a=p.parse_args()
    study(a.witnesses,a.out)
