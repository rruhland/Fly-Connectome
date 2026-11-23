"""Carry an optical estimate of an unannounced lateral force into the next receiving motion."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_arena import BALL_MASS
from .tt4_transfer_trial import _read, _write
from .tt24_rally_trial import PublicBody
from .tt26_grip_course import ReferenceReceiver, load_match
from .tt26_receiving_speed import draw
from .tt27_flight_feedback import OUT, Eyes, Player, command, load_bank


class WindCourt(ReferenceReceiver):
    wind = 0.0

    def _aerodynamics(self):
        super()._aerodynamics()
        if self.velocity[1] < 0 and -0.48 < self.ball[1] < 0.16:
            self.data.xfrc_applied[self._ball_body, 0] += BALL_MASS * self.wind


def deflection(trace):
    a = np.array(trace)
    if len(a) < 6:
        return None
    pre = a[a[:, 2] > 0.16][:8]
    post = a[(a[:, 2] < -0.55) & (a[:, 2] > -1.37)]
    if len(pre) < 3 or len(post) < 3 or np.ptp(post[:, 2]) < 0.19 or np.ptp(pre[:, 2]) < 0.17:
        return None

    def at_end(part):
        y = part[:, 2]
        design = np.column_stack([np.ones(len(y)), y + 1.53])
        return np.linalg.lstsq(design, part[:, 1], rcond=None)[0][0]

    return at_end(post) - at_end(pre)


class WindReceiver(Player):
    def __init__(self):
        super().__init__()
        archive = load_bank()["rows"]
        self.bias = [deflection(r["trace"]) for r in archive]
        self.speeds = []
        for r in archive:
            a = np.array(r["trace"][:4])
            self.speeds.append(abs(30 * (a[-1, 2] - a[0, 2]) / (a[-1, 0] - a[0, 0])))
        self.slopes = []
        for t, b in zip(self.templates, _read(OUT / "lateral-practice.json.gz")["source"], strict=True):
            valid = [r for r in b["rows"] if r["key"] is not None]
            # Estimate how joint goals changed in previously successful lateral motions.
            # The calibrated static Jacobian supplies a weak fallback for sparse neighborhoods.
            prior = t.matrix[:, 0]
            x = np.array([r["dx"] for r in valid])
            y = np.array([np.array(r["attempts"][-1]["goal"]) - t.goal for r in valid]).reshape(-1, 5)
            self.slopes.append((x @ y + 0.003 * prior) / (x @ x + 0.003))


class DriftMemory:
    def __init__(self):
        self.history = []

    def estimate(self):
        return np.median(self.history[-3:]) if self.history else 0.0

    def observe(self, value):
        if value is not None:
            self.history.append(float(np.clip(value, -9.0, 9.0)))


def point(arena, shot, model, state, method):
    arena.launch(shot)
    eye = Eyes(arena)
    body = PublicBody(arena)
    track = []
    goal = idx = None
    collecting = True
    for tick in range(arena.max_steps):
        if collecting:
            k, p = eye.observe(arena, body)
            if p is not None and (not track or p[1] < track[-1][2] - 0.014):
                track.append([tick, *p])
                collecting = p[1] > -1.37 or tick < 3
        if tick == 3:
            idx = model.choose(k)
            t = model.templates[idx]
            shift = np.clip(state.estimate() / model.speeds[idx] ** 2, -0.16, 0.16)
            if method == "frozen":
                shift = 0.0
            slope = t.matrix[:, 0] if method == "static_body" else model.slopes[idx]
            goal = t.goal + np.clip(slope * shift, -0.6, 0.6)
        u = command(goal, body)
        _, _, done, cap, _ = arena.step(u, render=False)
        body.issued(u)
        eye.view.issued(u)
        if done or cap or arena.legal_returns:
            break
    obs = deflection(track)
    estimate = (
        float((obs - model.bias[idx]) * model.speeds[idx] ** 2)
        if idx is not None and obs is not None and model.bias[idx] is not None
        else None
    )
    if method != "frozen":
        state.observe(estimate)
    return dict(
        legal=arena.legal_returns,
        contact=arena.contacts,
        origin=idx,
        estimate=estimate,
        shift=float(shift) if idx is not None else None,
        frames=tick + 1,
    )


def stream(args):
    path, arm = args
    shots, _ = load_match(path)
    model = WindReceiver()
    ids = [r["id"] for r in load_bank()["rows"]]
    rng = np.random.default_rng(948217 if arm == "familiar" else 948218)
    arena = WindCourt()
    result = []
    try:
        if arm == "recorded":
            roster = _read(OUT / "growing-confirmation.json.gz")["rows"]
            ids = [r["id"] for r in roster if r["group"] == "nearby"]
        states = {k: DriftMemory() for k in ("frozen", "static_body", "learned_body", "reset_at_change")}
        for scale in (0.0, 3.0, -3.5, 1.75, -1.9, 0.0):
            arena.wind = scale
            states["reset_at_change"] = DriftMemory()
            rows = []
            for t in range(15):
                feed = draw(shots, ids, rng, t) if arm == "familiar" else shots[int(rng.choice(ids))]
                cache = {}
                by = {}
                for k, memory in states.items():
                    typ = "learned_body" if k == "reset_at_change" else k
                    checkpoint = (typ, memory.estimate())
                    if checkpoint in cache:
                        r = cache[checkpoint]
                        memory.observe(r["estimate"])
                    else:
                        r = point(arena, feed, model, memory, typ)
                        cache[checkpoint] = r
                    by[k] = r
                rows.append(dict(t=t, id=feed["id"], by=by))
            result.append(dict(group=arm, acceleration=scale, rows=rows))
            print(arm, scale, {k: sum(r["by"][k]["legal"] for r in rows) for k in states}, flush=True)

    finally:
        arena.close()

    return result


def run(path):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        first, second = pool.map(stream, [(path, "familiar"), (path, "recorded")])
    _write(OUT / "crosswind.json.gz", dict(seeds=[948217, 948218], blocks=first + second))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    run(p.parse_args().feeds)
