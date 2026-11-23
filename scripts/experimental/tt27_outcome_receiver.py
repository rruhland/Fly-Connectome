"""Test a reward-conditioned motor memory on disjoint recorded launch trajectories."""

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from .tt4_transfer_trial import _read, _write
from .tt25_nearby_lessons import features
from .tt25_pose_demonstrations import episode
from .tt26_grip_course import ReferenceReceiver, PRIOR, load_match
from .tt26_match_broader import fixed_rosters
from .tt27_flight_feedback import OUT, SOURCE, load_bank
from .tt27_growing_receiver import strategy


class OutcomeModel:
    def __init__(self):
        with np.load(OUT / "growing-receiver.npz") as f:
            self.keys, self.goals, self.scale, self.n = [
                f[k] for k in ("keys", "goals", "scale", "original")
            ]
        v, a, y = [], [], []
        for k, g in zip(self.keys[: self.n], self.goals[: self.n], strict=True):
            v.append(k)
            a.append(g)
            y.append(1.0)
        for grp in _read(OUT / "growing-practice.json.gz")["training"]:
            for row in grp["rows"]:
                for t in row["attempts"]:
                    v.append(row["key"])
                    a.append(t["goal"])
                    y.append(float(t["legal"]))
        self.observations = np.asarray(v) / self.scale
        self.outcomes = np.asarray(y)
        self.candidates = np.unique(self.goals, axis=0)
        self.body_weight = np.exp(-0.5 * cdist(self.candidates, np.asarray(a), "sqeuclidean") / 0.22**2)
        self.motor_distance = cdist(self.candidates, self.goals)

    def act(self, k, name):
        if name in ("original", "blend"):
            return strategy(k, self.keys, self.goals, self.scale, self.n, name)
        d = np.linalg.norm(self.observations - k / self.scale, axis=1)
        sigma = max(0.25, np.partition(d, 52)[52] * 0.55)
        w = np.exp(-0.5 * (d / sigma) ** 2)
        count = self.body_weight @ w
        wins = self.body_weight @ (w * self.outcomes)
        probability = (wins + 0.05) / (count + 0.8)
        dsuccess = np.linalg.norm((self.keys - k) / self.scale, axis=1)
        # A motor option may have earned a return on several distinct flights.
        support = np.min(dsuccess[None, :] + 4 * self.motor_distance, axis=1)
        i = int(np.argmax(probability / (1.0 + 0.5 * support)))
        return self.candidates[i]


def independent(shots, *, seed=591301, excluded=(), near_count=6, unrestricted_count=96):
    _, ex, _ = fixed_rosters(shots)
    trained = _read(SOURCE / "match-broader.json.gz")
    ex.update(trained["training_roster"])
    ex.update(excluded)
    practiced = [r["id"] for r in load_bank()["rows"]]
    metric = _read(PRIOR / "recorded-nearby-lessons.json.gz")["metric"]
    pool = [s for s in shots.values() if s["id"] not in ex and s["pos_y"] > 0 and s["vel_y"] < 0]
    xs = np.stack([features(s) for s in pool])
    rng = np.random.default_rng(seed)
    used = set()
    out = []
    for i in practiced:
        distance = np.linalg.norm((xs - features(shots[i])) / metric, axis=1)
        choices = [int(j) for j in np.argsort(distance) if distance[j] > 1.0e-5 and j not in used][:40]
        for j in rng.choice(choices, size=near_count, replace=False):
            out.append(("nearby", pool[j]))
            used.add(j)
    more = rng.choice([j for j in range(len(pool)) if j not in used], size=unrestricted_count, replace=False)
    out.extend(("unrestricted", pool[j]) for j in more)
    return out


def run(path):
    shots, _ = load_match(path)
    model = OutcomeModel()
    arena = ReferenceReceiver()
    rows = []
    try:
        for group, shot in independent(shots):
            blank, k = episode(arena, shot, collect=True)
            seen = {}
            pair = {}
            for name in ("original", "blend", "outcomes"):
                if k is None:
                    r = blank
                else:
                    a = model.act(k, name)
                    ak = tuple(a)
                    if ak not in seen:
                        r = episode(arena, shot, goal=a)[0]
                        if (
                            r["legal"]
                            and r != episode(arena, shot, choose_goal=lambda v: model.act(v, name))[0]
                        ):
                            raise ValueError("Independent recorded shot failed replay")
                        seen[ak] = r
                    r = seen[ak]
                pair[name] = r
            rows.append(dict(group=group, id=shot["id"], scored=pair))
    finally:
        arena.close()
    _write(OUT / "growing-confirmation.json.gz", dict(seed=591301, rows=rows))
    for group in ("nearby", "unrestricted"):
        print(
            group,
            {k: sum(r["scored"][k]["legal"] for r in rows if r["group"] == group) for k in pair},
            flush=True,
        )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    run(p.parse_args().feeds)
