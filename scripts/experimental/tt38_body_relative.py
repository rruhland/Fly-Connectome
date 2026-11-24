"""Test continuous body-relative visual feedback and motor labels without backpropagation."""

import argparse
from itertools import product
from pathlib import Path
from time import process_time

import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import qmc

from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import BladeIK
from .tt37_joint_capture import OUTPUT as TT37
from .tt37_joint_deep import SCALES, run
from .tt37_opponent import arena


OUT = TT37.parent / "tt38-body-relative"
BANK = TT37 / "long-rgb.npz"
PLAN = TT37 / "opponent-plan.json"
DT = 1. / 60.
# The product of five revolute transforms contains only these harmonics.
FREQ = np.array([f for f in product((-1, 0, 1), repeat=5)
                 if next((v for v in f if v), 0) > 0], dtype=float)
MODES = ("absolute", "position", "motion", "transported")
FREQ_STOP = 1 + len(FREQ)


def embedding(q):
    phase = np.asarray(q)[..., :5] @ FREQ.T
    return np.concatenate((np.ones((*phase.shape[:-1], 1)), np.cos(phase), np.sin(phase)), axis=-1)


class Body:
    """A frozen encoder-to-blade map. No simulator or geometry state is queried in play."""

    def __init__(self, weights):
        self.weights = np.asarray(weights)

    def position(self, q):
        return embedding(q) @ self.weights

    def jacobian(self, q):
        phase = np.asarray(q)[..., :5] @ FREQ.T
        derivative = (self.weights[FREQ_STOP:].T * np.cos(phase)[..., None, :] -
                      self.weights[1:FREQ_STOP].T * np.sin(phase)[..., None, :])
        return derivative @ FREQ


def calibrate(path):
    """Use a body-only CAD calibration once. This labels encoder readings, not ball motion."""
    station = arena(protocol="random")
    try:
        fixture = BladeIK(station)
        joints = qmc.scale(qmc.Sobol(5, scramble=True, seed=382019).random_base2(11), fixture.low, fixture.high)
        position = np.asarray([fixture.pose(q)[0] for q in joints])
    finally:
        station.close()
    x = embedding(joints)
    weights = np.linalg.lstsq(x[:1536], position[:1536], rcond=1e-11)[0]
    error = np.linalg.norm(x[1536:] @ weights - position[1536:], axis=1)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, weights=weights)
    _write(path.with_suffix(".json"), dict(
        training_poses=1536, test_poses=512, mean_m=error.mean(), p95_m=np.quantile(error, .95),
        labels="Independent body-only revolute geometry. No ball rollout or labeled stroke.",
        sha256=_hash(path),
    ))
    print(path, "calibrated error (m)", error.mean(), error.max(), flush=True)


def groups(receipt):
    counts = [r["frames"] for r in receipt["practice"]]
    return (np.repeat(np.arange(len(counts)), counts),
            np.repeat([r["card"]["id"] for r in receipt["practice"]], counts))


def input_motion(states, mark, point, body):
    """Causal encoder difference only. End/start gaps must not fabricate paddle speed."""
    q = states[:, 6:11]
    p = body.position(q)
    v = np.zeros_like(p)
    contiguous = (np.diff(mark[:, 0]) == 1) & (np.diff(point) == 0)
    v[1:][contiguous] = np.diff(p, axis=0)[contiguous] / DT
    return p, np.clip(v, -3., 3.)


def body_inputs(x, q, position, speed, mode):
    scale = SCALES["local"]
    if mode == "absolute":
        return np.concatenate((x, q), axis=-1) / scale
    # Preserve the absolute table and gravity cues while changing the local metric.
    position_mix = np.array([.80, .55, .60])
    blocks = [np.sqrt(1 - position_mix) * x[..., :3] / scale[:3],
              np.sqrt(position_mix) * (x[..., :3] - position) / scale[:3]]
    if mode == "position":
        blocks.append(x[..., 3:6] / scale[3:6])
    else:
        speed_mix = np.array([.42, .22, .40])
        blocks.extend((np.sqrt(1 - speed_mix) * x[..., 3:6] / scale[3:6],
                       np.sqrt(speed_mix) * (x[..., 3:6] - speed) / scale[3:6]))
    blocks.append(q / scale[6:])
    return np.concatenate(blocks, axis=-1)


def regress(neighbors, label, dist, query):
    """Same local affine solve, weighting, support, and absolute channel clipping as DeepField."""
    offset = neighbors - query
    weight = np.exp(-.5 * (dist / max(.14, dist[4]))**2)
    weight /= weight.sum()
    mean, reference = weight @ offset, weight @ label
    dev = offset - mean
    fit = np.linalg.solve(dev.T @ (weight[:, None] * dev) + .045 * np.eye(len(query)),
                          dev.T @ (weight[:, None] * (label - reference)))
    return reference - np.clip(mean @ fit, -.23, .23)


class RelativeField:
    """One continuous command per frame from optical+encoder feedback."""

    def __init__(self, bank, receipt, weights, mode, *, use=slice(None), additional=()):
        self.mode = mode
        self.body = Body(weights)
        s = bank["states"]
        q = s[:, 6:11]
        point, _ = groups(receipt)
        p, v = input_motion(s, bank["marks"], point, self.body)
        features = body_inputs(s[:, :6], q, p, v, mode)[use]
        label = bank["targets_now"]
        if mode == "transported":
            # Demonstration effects are command-equivalent center displacements,
            # not measured future paddle positions or counterfactual ball flights.
            label = np.c_[label, self.body.position(label) - p]
        label = label[use]
        directions = np.sign(s[use, 4])
        if additional:
            features = np.concatenate([features, *(add[0] for add in additional)])
            label = np.concatenate([label, *(add[1] for add in additional)])
            directions = np.concatenate([directions, *(add[2] for add in additional)])
        self.branches = {}
        for sign in (-1, 1):
            positive = directions == sign
            branch = features[positive]
            self.branches[sign] = cKDTree(branch), branch, label[positive]
        self.begin_point()

    def begin_point(self):
        self.last_position = None
        self.last_tick = None
        self.tick = 0
        self.diagnostics = {"queries": 0, "supported": 0, "transported": 0,
                            "translation_m_sum": 0., "support_distance_sum": 0.}

    def clock_tick(self, tick):
        self.tick = tick

    def at(self, optical, q, p, speed):
        self.diagnostics["queries"] += 1
        query = body_inputs(optical, q[:5], p, speed, self.mode)
        tree, x, label = self.branches[-1 if optical[4] < 0. else 1]
        d, idx = tree.query(query, 12)
        confidence = float(np.clip(2.6 - d[2], 0., 1.))
        self.diagnostics["support_distance_sum"] += float(d[2])
        if not confidence:
            return None, 0.
        self.diagnostics["supported"] += 1
        command = regress(x[idx], label[idx], d, query)
        base = command[:6].copy()
        if self.mode == "transported" and optical[4] < -.90:
            predicted = self.body.position(base)
            mismatch = (p + command[6:]) - predicted
            jac = self.body.jacobian(base)
            # Favor axes with available travel, retaining a damped absolute joint anchor.
            gain = np.diag([1., .12, .8, .7, .48])
            j = jac @ gain
            step = gain @ j.T @ np.linalg.solve(j @ j.T + .020**2 * np.eye(3), mismatch)
            adjustment = np.clip(.70 * step, -.18, .18)
            base[:5] += adjustment
            self.diagnostics["transported"] += 1
            self.diagnostics["translation_m_sum"] += float(np.linalg.norm(jac @ adjustment))
        return base, confidence

    def get(self, optical, q, *, reference=None):
        p = self.body.position(q)
        speed = np.zeros(3) if self.last_tick != self.tick - 1 else np.clip((p - self.last_position) / DT, -3., 3.)
        self.last_tick, self.last_position = self.tick, p
        return self.at(optical, q, p, speed)


def physical(bank, body, feeds, deck, mode, output, *, limit=None):
    if mode == "absolute":
        hook = None
    else:
        receipt = _read(bank.with_suffix(".json"))
        with np.load(bank) as doc, np.load(body) as calibration:
            field = RelativeField(doc, receipt, calibration["weights"], mode)
        hook = lambda _: field
    run(feeds, bank, PLAN, deck, "local", output, cap=200, limit=limit, field_hook=hook,
        details=dict(body_coordinates=mode, body_sha256=_hash(body) if mode != "absolute" else None))


def offline(bank, path, body):
    """Hold all frames from each release out together, with special attention to late return."""
    receipt = _read(bank.with_suffix(".json"))
    with np.load(bank) as bank_doc, np.load(body) as calibration:
        doc, weights = dict(bank_doc), calibration["weights"]
    point, releases = groups(receipt)
    s, targets, marks = (doc[key] for key in ("states", "targets_now", "marks"))
    p, v = input_motion(s, marks, point, Body(weights))
    timing = {"all": np.ones(len(s), bool), "last_130_ms": (marks[:, 1] >= -.13) & (marks[:, 1] < 0)}
    progress = {}
    for mode in MODES:
        begin = process_time()
        predicted = np.full_like(targets, np.nan)
        support = np.zeros(len(s))
        for release in np.unique(releases):
            selected = releases == release
            model = RelativeField(doc, receipt, weights, mode, use=~selected)
            for i in np.flatnonzero(selected):
                estimate, confidence = model.at(s[i, :6], s[i, 6:11], p[i], v[i])
                if estimate is not None:
                    predicted[i] = estimate
                    support[i] = confidence
        ready = np.array(receipt["ready"])
        effective = ready + support[:, None] * (np.nan_to_num(predicted) - ready)
        error = np.abs(targets[:, :5] - effective[:, :5])
        pos_error = np.linalg.norm(Body(weights).position(targets) - Body(weights).position(effective), axis=1)
        progress[mode] = {
            key: dict(samples=int(mask.sum()), supported=int(np.count_nonzero(support[mask])),
                      mean_support=float(support[mask].mean()),
                      absolute_target_error_rad=float(error[mask].mean()),
                      blade_target_median_error_m=float(np.median(pos_error[mask])),
                      blade_target_p90_error_m=float(np.quantile(pos_error[mask], .90)))
            for key, mask in timing.items()
        }
        progress[mode]["cpu_s"] = process_time() - begin
        _write(path, {"bank_sha256": _hash(bank), "body_sha256": _hash(body), "releases": len(set(releases)),
                      "fold": "one entire release held out", "modes": progress})
        print(mode, progress[mode], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    steps = p.add_subparsers(dest="phase", required=True)
    fit = steps.add_parser("calibrate")
    fit.add_argument("--body", type=Path, default=OUT / "body.npz")
    check = steps.add_parser("offline")
    test = steps.add_parser("physical")
    for phase in (check, test):
        phase.add_argument("output", type=Path)
        phase.add_argument("--bank", type=Path, default=BANK)
        phase.add_argument("--body", type=Path, default=OUT / "body.npz")
    test.add_argument("mode", choices=MODES)
    test.add_argument("deck", choices=("held_openings", "held_replies", "familiar", "practice"))
    test.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    test.add_argument("--limit", type=int)
    a = p.parse_args()
    if a.phase == "calibrate":
        calibrate(a.body)
    elif a.phase == "offline":
        offline(a.bank, a.output, a.body)
    else:
        physical(a.bank, a.body, a.feeds, a.deck, a.mode, a.output, limit=a.limit)


if __name__ == "__main__":
    main()
