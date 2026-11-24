"""Causal camera anticipation of rare low returns in strong articulated points."""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_vision import BOUNCE_Z, VERTICAL, flight_terms
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt37_cue_vision import PaddleEye, quality
from .tt37_opponent import arena

ROOT = Path("artifacts")
OUT = ROOT / "tt41_anticipate"
TAPES = ("tt37/mix-rollin", "tt39/endurance-02", "tt39/endurance-05")
OUT_LINES = (-.70, -.10, .45, .95)
IN_LINES = (.75, .10, -.55, -.98)
OUT_NAMES = tuple(f"out_{v:+.2f}" for v in OUT_LINES)
IN_NAMES = tuple(f"in_{v:+.2f}" for v in IN_LINES)
GATES = OUT_NAMES + ("first_in",) + IN_NAMES + ("seen_near_bounce",)
LOW_THRESHOLDS = (.99, .95)


class GateEye:
    """Generate every decision online from RGB direction, line crossings and rebound fit."""
    def __init__(self):
        self.direction = 0
        self.seen = set()
        self.farward = None
        self.extreme = None

    def step(self, fit, now, blade):
        if not quality(fit, now):
            return []
        p, v = fit.position, fit.velocity
        vy = v[1]
        direction = (1 if vy > 1.15 else -1 if vy < -1.15 else 0)
        if not direction:
            return []
        if direction != self.direction:
            # Use only earlier measured progress and current motion. Some real far
            # returns are struck at y≈+.5 and the first stable inward fit is much closer.
            if self.direction == 1 and max(self.extreme, p[1]) < .22:
                return []
            if self.direction == -1 and min(self.extreme, p[1]) > -.67:
                return []
            self.seen.clear()
            self.direction = direction
            self.extreme = p[1]
        self.extreme = (max if direction == 1 else min)(self.extreme, p[1])
        optical = np.r_[p, v].tolist()
        if direction == 1:
            self.farward = optical
            ready = [name for name, lim in zip(OUT_NAMES, OUT_LINES) if lim <= p[1] < 1.95]
        else:
            ready = ["first_in"] + [name for name, lim in zip(IN_NAMES, IN_LINES) if p[1] <= lim]
            if fit.bounced and p[1] < -.55:
                ready.append("seen_near_bounce")
        markers = []
        for name in ready:
            if name not in self.seen:
                markers.append(dict(gate=name, decision_s=now, ball=optical,
                                    paddle=None if blade is None else blade.tolist(),
                                    bounced=fit.bounced, rms=fit.rms, sigma=fit.depth_sigma,
                                    previous_out=self.farward if direction == -1 else None))
                self.seen.add(name)
        return markers


def physical_projection(state, bounced, *, target=-1.53):
    """Stock-style local gravity/drag projection with generic near closure, no latent spin."""
    if state is None:
        return None
    p, v = np.asarray(state, dtype=float).reshape(2, 3).copy()
    if v[1] >= -.8:
        return None
    damp = .133*np.clip(np.linalg.norm(v), 2.5, 12.)
    origin = 0.

    def eta(distance):
        travel = distance/v[1]
        return None if 1.-damp*travel <= .08 else -np.log1p(-damp*travel)/damp

    def evolve(t):
        travel, fall = flight_terms(t, damp)
        return p+travel*v+fall*VERTICAL, np.exp(-damp*t)*v+travel*VERTICAL

    t = eta(target-p[1])
    if t is None or not -.10 <= t <= 1.6:
        return None
    if not bounced and t > .02 and p[2] > BOUNCE_Z > evolve(t)[0][2]:
        hit = brentq(lambda h: evolve(h)[0][2]-BOUNCE_Z, 0., t)
        bp, bv = evolve(hit)
        if abs(bp[0]) < .76 and -1.37 < bp[1] < 0.:
            p, v = bp, bv*np.array([.95, .95, -.92])
            origin = hit
            t = eta(target-p[1])
            if t is None or t < 0:
                return None
    x, _, z = evolve(t)[0]
    if not abs(x) < 1.35 or not .60 < z < 1.80:
        return None
    return x, z, origin+t


def outbound_projection(state, bounced=False):
    """Estimate the upcoming far table crossing from an outward RGB flight segment."""
    if state is None:
        return None
    p, v = np.asarray(state,dtype=float).reshape(2,3)
    if v[1] <= 1.05 or p[2] <= BOUNCE_Z-.01 or (bounced and p[1] >= .05):
        return None
    damp = .133*np.clip(np.linalg.norm(v),2.5,12.)

    def evolve(t):
        travel, fall = flight_terms(t,damp)
        return p+travel*v+fall*VERTICAL

    if evolve(1.55)[2] > BOUNCE_Z or evolve(0.)[2] < BOUNCE_Z:
        return None
    t = brentq(lambda h:evolve(h)[2]-BOUNCE_Z,0.,1.55)
    x,y,_ = evolve(t)
    if not (-.65 <= y <= 2.7 and abs(x)<1.45):
        return None
    reach = -p[1]/v[1]
    margin = (evolve(-np.log1p(-damp*reach)/damp)[2]-.915
              if 0. < reach < .9 and 1.-damp*reach > .10 else 0.)
    vz = (v[2]+9.81/damp)*np.exp(-damp*t)-9.81/damp
    return x,y,float(t),float(margin),float(vz)


def poly_arc(raw, end, *, dt):
    """Fit a damped gravity arc from stereo exposures after an RGB-observed bounce."""
    ready = np.asarray([r for r in raw if end-.92 <= r[0] < end and -1.72 < r[2] < -.22])
    if ready.ndim != 2 or len(ready) < 6:
        return None
    # No native bounce timing is used to pick an optical rebound. Require inward observations
    # on both sides plus a vertical turn near table height.
    choices = []
    for i in range(2, len(ready)-3):
        if ready[i, 3] > .846 or ready[i, 3] < .748:
            continue
        before = (ready[i, 3]-ready[i-2, 3])/(ready[i, 0]-ready[i-2, 0])
        early = (ready[i+2, 3]-ready[i, 3])/(ready[i+2, 0]-ready[i, 0])
        if before < -.35 and early > .27 and ready[i+2, 2] < ready[i-2, 2]-.07:
            choices.append(i)
    if not choices:
        return None
    i = min(choices, key=lambda k: ready[k, 3])
    arc = ready[i+1:]
    if len(arc) < 4 or arc[-1, 0]-arc[0, 0] < 2.8*dt:
        return None
    if np.any(np.diff(arc[:, 2]) > .010):
        return None
    arc = arc[-min(len(arc), 9):]
    t = arc[:, 0]-arc[-1, 0]
    xyz = arc[:, 1:]
    velocity = (xyz[-1]-xyz[0])/(t[-1]-t[0])
    damp = .133*np.clip(np.linalg.norm(velocity), 2.5, 12.)
    travel, fall = flight_terms(t, damp)
    terms = np.c_[np.ones(len(t)), travel]
    fit = np.linalg.lstsq(terms, xyz-fall[:, None]*VERTICAL, rcond=None)[0]
    error = float(np.sqrt(np.mean(np.sum((xyz-terms@fit-fall[:, None]*VERTICAL)**2, axis=1))))
    if error > .018 or fit[1, 1] >= -1.15 or fit[0, 1] > -.91:
        return None
    proj = physical_projection(fit.ravel(), True)
    if proj is None or not .65 < proj[1] < 1.43 or not -.14 < proj[2] < .29:
        return None
    return dict(height=proj[1], x=proj[0], at_s=arc[-1, 0]+proj[2], label_s=arc[-1, 0],
                optical_bounce_s=ready[i, 0], n=len(arc), rms_m=error)


def match(audit, markers, points, *, period, end):
    rows = []
    events = audit["events"]
    own = [e for e in events if e.get("collider") == "learner_racket"]
    for leg, delivery in enumerate(audit["placements"]["delivered"]):
        far, bounce, near = (delivery[k] for k in ("far_contact", "near_bounce", "near_contact"))
        if far is None:
            continue
        t, bt = far["time"], bounce["time"]
        end_label = min(end, near["time"] if near else bt+.33)
        # Keep labels post-far with native labels used only to audit timing and avoid a
        # second racket movement contaminating the held optical ground truth.
        label = poly_arc([p for p in points if t < p[0]], end_label, dt=period)
        if label is not None and abs(label["optical_bounce_s"]-bt) > 2.6*period:
            label = None
        previous = max([e["time"] for e in own if e["time"] < t], default=-.001)
        allowed = [g for g in markers if previous < g["decision_s"] < end_label and
                   (g["decision_s"] < t) == g["gate"].startswith("out_")]
        by_gate = {}
        for g in allowed:
            # A native hit can reject a spurious stand-alone marker, but never select
            # the actual calling time of a gate or add a marker.
            by_gate.setdefault(g["gate"], g)
        rows.append(dict(leg=leg, far_contact_s=t, near_bounce_s=bt,
                         near_contact_s=None if near is None else near["time"], optical=label,
                         gate=by_gate))
    return rows


def stereo_xyz(eye, now):
    """Triangulate the same genuine ball sightings already consumed by the online eye."""
    seen = [r for r in reversed(eye.rays) if r.time == now and r.depth_weight > 0.]
    if len(seen) != 2:
        return None
    a, b = seen
    t, u = np.linalg.lstsq(np.c_[a.direction,-b.direction],b.eye-a.eye,rcond=None)[0]
    if t <= 0. or u <= 0.:
        return None
    p, q = a.eye+t*a.direction,b.eye+u*b.direction
    return .5*(p+q) if np.linalg.norm(p-q) < .030 else None


def capture(out, sources, *, limit=None, stride=2, resume=False, priority_source=None):
    inputs = [(ROOT/(name+".json.gz"), None) for name in sources]
    inputs = [(p, _read(p)) for p, _ in inputs]
    target = max(len(d["rows"]) for _, d in inputs) if limit is None else limit
    sources = {str(p): _hash(p) for p, _ in inputs}
    rows = _read(out)["rows"] if resume and out.exists() else []
    done = {(r["source"], r["card"]["id"], r["card"]["seed"]) for r in rows}
    if rows:
        old = _read(out)
        if old.get("schema") != "tt41-anticipate-2" or old["sources"] != sources or old["stride"] != stride:
            raise ValueError("Resume would mix different optical collectors")
    result = dict(schema="tt41-anticipate-2", sources=sources, rows=rows, limit=target,
                  stride=stride, paired_native="exact event digest", completed=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    ready = inputs[0][1]["ready"]
    if any(d["ready"] != ready for _,d in inputs):
        raise ValueError("Capture sources use different initial ready profiles")
    with centered(ready):
        a = arena()
        s, rig, *_ = resources(a)
        try:
            jobs = [(i,path,data) for i in range(target) for path,data in inputs]
            if priority_source is not None:
                first = str(ROOT/(priority_source+".json.gz"))
                if first not in sources:
                    raise ValueError("Priority source is not part of the fixed capture roster")
                jobs.sort(key=lambda x:str(x[1])!=first)
            for i,path,data in jobs:
                if i >= len(data["rows"]):
                    continue
                c, r = data["cards"][i], data["rows"][i]
                if (str(path),c["id"],c["seed"]) in done:
                    continue
                a.launch(r["shot"], card=c)
                a.partner.reset(seed=c["seed"])
                eye = Vision(rig, a.control_dt, raw=True)
                paddle = PaddleEye({name: rig.fixed[name] for name in ("left", "right")})
                gate = GateEye()
                markers, points = [], []
                for tick, u in enumerate(r["actions"]):
                    if tick % stride == 0:
                        t, frames = tick*a.control_dt, s.capture()
                        eye.update(tick, frames, a.joints)
                        blade = paddle.update(t, frames)
                        markers.extend(gate.step(eye.eye.latest, t, blade))
                        xyz = stereo_xyz(eye.eye,t)
                        if xyz is not None:
                            points.append([t,*xyz.tolist()])
                    a.exchange(u, FAR_COMMAND)
                if digest_json(a.events) != digest_json(r["events"]):
                    raise ValueError(f"Game diverged on {path.name} {c['id']}")
                examples = match(r, markers, points, period=stride*a.control_dt, end=a.data.time)
                rows.append(dict(source=str(path), card=c, gate_calls=len(markers), rows=examples))
                _write(out, result)
                print(path.name, c["id"], "offers", len(examples), "labels",
                      sum(e["optical"] is not None for e in examples),
                      "low", sum(e["optical"] is not None and e["optical"]["height"] < .99 for e in examples),
                      flush=True)
        finally:
            s.close()
            a.close()
    result["completed"] = True
    _write(out, result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("out", type=Path, nargs="?", default=OUT/"receipts.json.gz")
    parser.add_argument("--sources", nargs="+", default=TAPES)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--priority-source")
    args = parser.parse_args()
    capture(args.out, args.sources, limit=args.limit, stride=args.stride, resume=args.resume,
            priority_source=args.priority_source)
