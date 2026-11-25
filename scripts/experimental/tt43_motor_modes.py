"""Learn separate continuous joint-movement readouts from awarded optical practice."""
import argparse
from collections import deque
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.spatial import cKDTree

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent import draw
from .tt39_carry_joints import Carried

ROOT = Path("artifacts/tt43_motor_modes")
CARRY = course.ROOT / "learning-24.json.gz"
CONTROL = course.ROOT / "after24.json.gz"
FEEDS = Path("/tmp/tt25-rallies.json")
PROTOCOL = ROOT / "held.protocol.json"
# Phase compression changes coordinates for regression, never determines a contact or release.
KNOTS = np.array([-14., -10., -7., -4., -2., 0., 2., 5., 9., 14., 20.])
JOINT_SCALE = np.array([.45, .31, .36, .47, .40, .15])
FRAMES = np.arange(mpc.HORIZON)


def flight(x):
    """Expose comparable incoming paths at a common depth using only the observed fit."""
    x = np.asarray(x)
    p, v = x[..., :3], x[..., 3:6]
    # Approximate flight phase rather than trying to infer an impact timestamp.
    t = np.clip((p[..., 1] + 1.54) / np.maximum(-v[..., 1], 1.8), -.07, .29)
    out = np.stack((p[..., 0] + t * v[..., 0],
                    p[..., 2] + t * (v[..., 2] - 4.905 * t),
                    v[..., 0], v[..., 1], v[..., 2] - 9.81 * t, t), axis=-1)
    return out, t / mpc.DT - 1.


def coordinates(state):
    state = np.asarray(state) * mpc.OPEN_NORM
    p, phase = flight(state[..., :6])
    key = np.concatenate((p / [.095, .065, .78, 1.10, 1.05, .12],
                          state[..., 6:11] / [2.1, 1.4, 1.4, 1.8, 2.1],
                          state[..., 11:] / [.55, .42, .52, .60, .65]), axis=-1)
    return key, phase


def encode(phase, goals, start):
    times = np.clip(phase[..., None] + KNOTS, -1., mpc.HORIZON - 1) + 1.
    lo = np.floor(times).astype(int)
    hi = np.minimum(lo + 1, mpc.HORIZON)
    trace = np.concatenate((start[..., None, :], goals), axis=-2)
    a = np.take_along_axis(trace, lo[..., None], axis=-2)
    b = np.take_along_axis(trace, hi[..., None], axis=-2)
    return a + (b - a) * (times - lo)[..., None]


def decode(knots, phase, advance=0.):
    t = FRAMES - phase + advance
    return np.stack([np.interp(t, KNOTS, col) for col in knots.T], axis=-1)


def rows_to_bank(rows):
    state = np.asarray([r[0] for r in rows])
    key, phase = coordinates(state)
    goals = np.asarray([r[1] for r in rows])
    physical = state * mpc.OPEN_NORM
    start5 = physical[:, 6:11] + physical[:, 11:]
    start = np.c_[start5, goals[:, 0, -1]]
    return key, encode(phase, goals, start), [r[2] for r in rows]


def local_fit(delta, outcome, weights):
    """A weighted ridge solution evaluated at delta=0 plus learned residual variation."""
    weights = weights / np.sum(weights)
    center = weights @ delta
    mean = np.einsum("n,nhj->hj", weights, outcome)
    delta = delta - center
    residual = (outcome - mean).reshape(len(outcome), -1)
    gram = (delta.T * weights) @ delta + .13 * np.eye(delta.shape[1])
    slope = np.linalg.solve(gram, (delta.T * weights) @ residual)
    correction = -(center @ slope).reshape(mean.shape)
    fit = mean + np.clip(correction, -.32, .32)
    residual = residual - delta @ slope
    return fit, mean, (residual * np.sqrt(weights[:, None])).reshape(outcome.shape)


class Readout:
    def __init__(self, rows):
        admitted = [r for r in rows if -.62 > r[0][1] * mpc.OPEN_NORM[1] > -1.65]
        self.keys, self.knots, self.tags = rows_to_bank(admitted)
        self.tree = cKDTree(self.keys)
        self.online_digest = None
        self.misses = deque(maxlen=128)

    def refresh(self, online):
        # A returned point changes the right edge of the ring even at full capacity.
        marker = (len(online), online[-1][2] if online else None)
        if marker != self.online_digest:
            self.recent = rows_to_bank(list(online)) if online else None
            self.online_digest = marker

    def proposals(self, x, q, ref, online):
        key, phase = coordinates(mpc.query(x, q, ref))
        self.refresh(online)
        dist, idx = self.tree.query(key, k=min(168, len(self.keys)))
        bank = [(d, self.keys[i], self.knots[i], self.tags[i]) for d, i in zip(dist, idx)]
        if self.recent is not None:
            a, b, tags = self.recent
            ds = np.linalg.norm(a - key, axis=1)
            bank.extend((max(0., d - .12), k, g, tag) for d, k, g, tag in zip(ds, a, b, tags))
        picked, seen = [], set()
        for d, k, g, tag in sorted(bank, key=lambda row: row[0]):
            if d > 3.35 or len(picked) >= 35:
                break
            if tag not in seen:
                picked.append((d, k, g))
                seen.add(tag)
        if len(picked) < 5:
            return key, phase, []
        distances, keys, goals = map(np.asarray, zip(*picked))
        # Isolate temporal motor shapes before fitting. Interpolating across
        # incompatible wrist/elbow modes can produce no plausible swing.
        signatures = (goals[:, 2:9, :5] / JOINT_SCALE[:5]).reshape(len(goals), -1)
        supports = []
        candidates = []
        for anchor in range(len(goals)):
            distances_mode = np.linalg.norm(signatures - signatures[anchor], axis=1) / np.sqrt(7.)
            if any(np.linalg.norm(signatures[anchor] - center) / np.sqrt(7.) < 1.25 for center in supports):
                continue
            good = distances_mode < 1.65
            if good.sum() < 5:
                continue
            supports.append(signatures[anchor])
            w = np.exp(-.50 * ((distances[good] / .9)**2 + distances_mode[good]**2))
            fit, mean, residual = local_fit(keys[good] - key, goals[good], w)
            ready = np.clip(1. - np.maximum(0., FRAMES - 1.) / 5., 0., 1.)
            # The saved fragment has its own reference baseline. Ramp from
            # here to the novel prediction when the remaining flight allows.
            def emit(g, shift=0.):
                seq = decode(g, phase, shift)
                return seq + (.28 * ready[:, None]) * np.clip(ref - seq[0], -.28, .28)
            candidates.extend([emit(fit), emit(mean), emit(fit, -1.7), emit(fit, 1.7)])
            n, h, j = residual.shape
            _, s, shape = np.linalg.svd(residual.reshape(n, h * j), full_matrices=False)
            for k in range(min(2, len(shape))):
                axis = np.clip(1.55 * s[k] * shape[k].reshape(h, j), -.27, .27)
                candidates.extend([emit(fit + axis), emit(fit - axis)])
            if len(supports) == 3:
                break
        return key, phase, candidates

    def penalty(self, key, phase, options):
        if not self.misses:
            return np.zeros(len(options))
        active = [(np.linalg.norm(k - key), g) for k, g in self.misses]
        near = [(d, g) for d, g in active if d < 1.4]
        if not near:
            return np.zeros(len(options))
        out = []
        for option in options:
            shape = np.stack([np.interp(phase + k, FRAMES, option[:, j])
                              for k in (-5., -2., 0., 2.) for j in range(6)]).reshape(4, 6)
            out.append(max(.65 * np.exp(-.5 * (d / .65)**2
                                        -.5 * np.mean(((shape - g) / (.30 * JOINT_SCALE))**2))
                           for d, g in near))
        return np.asarray(out)

    def miss(self, memory):
        inbound = [i for i, (_, x, *_) in enumerate(memory)
                   if x is not None and x[4] < -.9 and x[1] < -.76]
        if not inbound:
            return 0
        end = inbound[-1]
        added = 0
        for i in inbound[-13::4]:
            if end - i > 20:
                continue
            _, x, q, ref, _ = memory[i]
            if x[1] < -1.45:
                continue
            key, phase = coordinates(mpc.query(x, q, ref))
            if not 1 < phase < 15 or i + phase > len(memory) + 1:
                continue
            issued = mpc.tail(np.asarray([r[-1] for r in memory[i:i + mpc.HORIZON]]), mpc.HORIZON)
            shape = np.stack([np.interp(phase + k, FRAMES, issued[:, j])
                              for k in (-5., -2., 0., 2.) for j in range(6)]).reshape(4, 6)
            self.misses.append((key, shape))
            added += 1
        return added


class MotorMPC(mpc.MPC):
    def __init__(self, *, rescue, **kwargs):
        super().__init__(**kwargs)
        self.rescue = rescue
        self.readout = Readout(self.priors.rows) if rescue else None

    def missed(self, archive):
        if self.rescue:
            self.diag["mode_miss_samples"] += self.readout.miss(archive)

    def search(self, tick, x, q, ref, base):
        before = len(self.notes)
        super().search(tick, x, q, ref, base)
        if not self.rescue or len(self.notes) == before:
            return
        note = self.notes[-1]
        if note.get("robust") or not self.engine.bounced or not -1.53 < x[1] < -.76:
            return
        key, phase, options = self.readout.proposals(x, q, ref, self.priors.online)
        if not options:
            return
        self.diag["mode_attempts"] += 1
        lo = self.engine.twin._joint_low[self.engine.actuators]
        hi = self.engine.twin._joint_high[self.engine.actuators]
        options = np.clip(options, lo, hi)
        first, _, tracks, _, f = self.trial(options, traces=True)
        order = np.argsort(first)
        variants = []
        # Give nearby mode predictions an independent physical correction,
        # but retain the uncorrected counterfactuals in the candidate pool.
        for i in order[:5]:
            adjust = self.transport(tracks[i])
            if adjust is not None:
                variants.extend([options[i] + .8 * adjust, options[i] + self.transport(tracks[i], face=True)])
        if variants:
            variants = np.clip(variants, lo, hi)
            first2, _, _, _, f2 = self.trial(variants)
            first = np.r_[first, first2]
            f = np.vstack((f, f2))
            options = np.concatenate((options, variants))
        note["mode_contacts"] = int(np.sum(f[:, 1] > 0))
        note["mode_options"] = len(options)
        if not note["mode_contacts"]:
            return
        rank = np.argsort(first)[:8]
        qloss, qlegal, _, y, _ = self.trial(options[rank], long=True)
        legal = np.flatnonzero(qlegal)
        note["mode_legal"] = len(legal)
        if len(legal):
            elite = legal[np.argsort(qloss[legal])[:5]]
            agreement, risk = self.uncertainty(options[rank[elite]], qloss[elite])
            cost = risk + np.where(agreement == 2, 0., np.where(agreement == 1, .55, 20.))
            cost += self.readout.penalty(key, phase, options[rank[elite]])
            winner = np.argmin(cost)
            if agreement[winner]:
                self.best, self.tick, self.lease = options[rank[elite[winner]]], tick, mpc.LEASE
                note.update(chosen="motor_mode", robust=int(agreement[winner]), score=float(cost[winner]),
                            contact_delay_s=float(y[elite[winner], 1]),
                            target_errors=y[elite[winner], 3:10].tolist())
                self.diag["mode_selected"] += 1


def checkpoint(model, path):
    bank, misses = model.priors.online, model.readout.misses if model.rescue else []
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(
            f, states=[r[0] for r in bank], goals=[r[1] for r in bank],
            tags=[repr(r[2]) for r in bank], serial=model.serial,
            miss_keys=np.array([r[0] for r in misses]).reshape(-1, 16),
            miss_shapes=np.array([r[1] for r in misses]).reshape(-1, 4, 6),
        )
    tmp.replace(path)
    return _hash(path)


def path_for(out):
    return out.with_suffix("").with_suffix(".npz")


def plan(a):
    if not 0 < a.n <= 24:
        raise ValueError("Use the unfiltered prefix of the 24 assessment cards")
    cards = _read(course.PLAN)["assessment"][:a.n]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    _write(a.out, {
        "cards": cards, "assessment_plan_sha256": _hash(course.PLAN),
        "carry_sha256": _hash(a.carry), "carry_npz_sha256": _hash(path_for(a.carry)),
        "feeds_sha256": _hash(a.feeds), "bank_sha256": _hash(mpc.BANK),
        "optics_sha256": _hash(course.OPTIC), "cap": 240, "mode": a.mode,
        "model_sha256": {Path(p).name: _hash(Path(p)) for p in (__file__, mpc.__file__)},
        "design": "Unfiltered prefix of existing prospective assessment, same measured start and exact broad independent opening/reply draw per leg. Stock and rescue share the original controller, rewarded prior memory, and physical settings.",
    })


def run(a):
    spec = _read(a.protocol)
    cards, mode = spec["cards"], spec["mode"]
    expected = {
        "assessment_plan_sha256": _hash(course.PLAN),
        "carry_sha256": _hash(a.carry), "carry_npz_sha256": _hash(path_for(a.carry)),
        "feeds_sha256": _hash(a.feeds), "bank_sha256": _hash(mpc.BANK),
        "optics_sha256": _hash(course.OPTIC),
        "model_sha256": {Path(p).name: _hash(Path(p)) for p in (__file__, mpc.__file__)},
    }
    if any(spec.get(k) != v for k, v in expected.items()) or cards != _read(course.PLAN)["assessment"][:len(cards)]:
        raise ValueError("Frozen assessment conditions differ")
    held = {c["id"] for c in cards}
    base = _read(a.carry)
    if {c["id"] for c in base["cards"][:len(base["rows"])]} & held:
        raise ValueError("Assessment overlaps carried practice")
    stock = _read(a.stock) if a.stock else None
    if stock and (
        stock["cards"][:len(cards)] != cards or stock["mode"] != mode
        or stock["cap"] != spec["cap"] or stock["carry_sha256"] != spec["carry_sha256"]
        or stock["source_sha256"] != spec["bank_sha256"] or stock["optical_sha256"] != spec["optics_sha256"]
        or len(stock["rows"]) < len(cards)
    ):
        raise ValueError("Archived stock conditions differ")
    ready = _read(mpc.BANK.with_suffix(".json"))["ready"]
    feeds = mpc.measured_launches(a.feeds)
    with np.load(mpc.BANK) as b:
        raw = mpc.DeepField(b, "local")
    doc = {
        "protocol_sha256": _hash(a.protocol), "cards": cards, "mode": mode, "cap": spec["cap"],
        "stock_sha256": _hash(a.stock) if a.stock else None, "ready": ready,
        "rescue": a.kind == "motor", "rows": [],
    }
    with mpc.centered(ready):
        arena = mpc.arena()
        camera, rig, stride, lo, hi = mpc.resources(arena)
        with patch.object(mpc, "Fragments", lambda file, engine: Carried(file, engine, practice=[course.PRACTICE], held=held)):
            model = MotorMPC(rescue=doc["rescue"], bank=mpc.BANK, raw=raw, variant="elastic")
        try:
            course.resume(model, path_for(a.carry), base)
            if a.resume:
                previous = _read(a.out)
                if any(previous.get(k) != v for k, v in doc.items() if k != "rows"):
                    raise ValueError("Assessment receipt differs")
                doc = previous
                ck = path_for(a.out)
                course.resume(model, ck, {**doc, "rows": [None] * len(base["rows"]) + doc["rows"]})
                with np.load(ck) as f:
                    if model.rescue:
                        model.readout.misses.extend(zip(f["miss_keys"], f["miss_shapes"], strict=True))
            for index in range(len(doc["rows"]), len(cards)):
                c = cards[index]
                r = mpc.play(arena, camera, rig, stride, lo, hi, ready, c, feeds[c["id"]],
                             raw, model, course.OPTIC, mode, spec["cap"], 20)
                for leg, move in enumerate(r["placements"]["requested"]):
                    if list(draw(c, leg, "random", None)[0]) != move["target"]:
                        raise ValueError("Physical game used a different opponent draw")
                if stock:
                    old = stock["rows"][index]
                    diff = next((t for t, (x, y) in enumerate(zip(old["actions"], r["actions"])) if x != y),
                                None if len(old["actions"]) == len(r["actions"]) else min(len(old["actions"]), len(r["actions"])))
                    if index == 0 and r["memory"]["before"] != old["memory"]["before"]:
                        raise ValueError("Starting checkpoint differs from stock")
                    if diff is None and digest_json(old["events"]) != digest_json(r["events"]):
                        raise ValueError("Equal commands did not reproduce stock")
                    r["stock_learner"], r["first_action_diff"] = old["score"]["learner"], diff
                    r["stock_event_sha256"] = digest_json(old["events"])
                doc["rows"].append(r)
                doc["checkpoint_sha256"] = checkpoint(model, path_for(a.out))
                doc["complete"] = len(doc["rows"]) == len(cards)
                _write(a.out, doc)
                print(index + 1, c["id"], r.get("stock_learner"), r["score"]["learner"],
                      r["planner"].get("mode_attempts", 0), r["planner"].get("mode_selected", 0),
                      round(r["runtime_cpu_s"], 2), flush=True)
        finally:
            model.close()
            camera.close()
            arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("kind", choices=("plan", "motor", "stock"))
    p.add_argument("out", type=Path)
    p.add_argument("--mode", choices=("live", "frozen"), default="frozen")
    p.add_argument("--carry", type=Path, default=CARRY)
    p.add_argument("--stock", type=Path)
    p.add_argument("--feeds", type=Path, default=FEEDS)
    p.add_argument("--protocol", type=Path, default=PROTOCOL)
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.kind == "plan":
        plan(args)
    else:
        run(args)
