"""Estimate long-run rally value by linking observed approach states across actual scored movements."""

import argparse
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
from scipy.cluster.vq import kmeans2
from scipy.spatial.distance import cdist

from . import tt46_continuous_season as season
from . import tt49_sustain_value as value
from .tt4_transfer_trial import _hash, _read, _write

ROOT = Path("artifacts/tt49_viability")
MODEL = ROOT / "transition.json"


def observed_state(notes):
    if len(notes) < 2:
        return None
    a, b = notes[:2]
    if b["tick"] - a["tick"] > 12 or a["tick"] >= b["tick"]:
        return None
    p, q = np.asarray(a["p"]), np.asarray(b["p"])
    vel = (q - p) * 60 / (b["tick"] - a["tick"])
    return np.r_[q, vel, float(b["bounced"])]


def collect(files, held):
    points = {}
    coverage = {}
    for path in files:
        d = _read(path)
        for card, r in zip(d["cards"], d["rows"], strict=True):
            if (
                card["id"] in held
                or card.get("profile") != "reach"
                or not r.get("searches")
                or not r.get("placements")
            ):
                continue
            offers = [o for o in r["placements"]["delivered"] if o.get("near_bounce")]
            attempts = []
            for o in offers:
                start = o["far_contact"]["step"]
                stop = (o.get("near_contact") or r["events"][-1])["step"]
                visits = [n for n in r["searches"] if start < n["tick"] <= stop]
                state = observed_state(visits)
                accepted = [
                    n
                    for n in visits
                    if n.get("target_errors") and n.get("contact_delay_s") is not None
                ]
                cmd = None
                if accepted and visits[-1]["tick"] - accepted[-1]["tick"] <= 5:
                    cmd = np.array(accepted[-1]["target_errors"])
                    if len(cmd) != 7:
                        cmd = None
                attempts.append((state, cmd))
            for i, ((state, cmd), o) in enumerate(zip(attempts, offers, strict=True)):
                if state is None:
                    continue
                identity = (
                    card["id"],
                    *np.round(
                        o["far_contact"]["ball"]
                        + o["far_contact"]["incoming"]
                        + state.tolist(),
                        5,
                    ),
                )
                coverage.setdefault(identity, [state, []])[1].append(
                    bool(o["near_landed"])
                )
                if cmd is None or (
                    i + 1 == len(offers)
                    and not any(e["kind"] == "point" for e in r["events"])
                ):
                    continue
                reached = bool(o["near_landed"])
                nxt = attempts[i + 1][0] if reached and i + 1 < len(offers) else None
                scores = [float(o2["near_landed"]) for o2 in offers[i : i + 4]]
                empirical = sum((0.91**j) * v for j, v in enumerate(scores))
                key = (*identity, *np.round(cmd, 6))
                row = points.setdefault(key, [state, cmd, [], [], [], card["id"]])
                row[2].append(reached)
                row[3].append(nxt)
                row[4].append(empirical)
    return list(points.values()), list(coverage.items())


def assign(x, centers, mean, scale):
    return cdist((np.atleast_2d(x) - mean) / scale, centers, "sqeuclidean").argmin(
        axis=1
    )


def groups(x, n, seed):
    mu, sd = x.mean(0), np.maximum(x.std(0), 0.035)
    center, _ = kmeans2((x - mu) / sd, n, minit="++", seed=seed, iter=25)
    return center, mu, sd


def solve(rows, coverage):
    x, a, _, _, _, _ = zip(*rows, strict=True)
    state = groups(np.stack(x), 14, 49941)
    action = groups(np.stack(a), 12, 49952)
    s = assign(x, *state)
    u = assign(a, *action)
    ns, na = len(state[0]), len(action[0])
    n = np.zeros((ns, na))
    own = np.zeros_like(n)
    transition = np.zeros((ns, na, ns))
    for i, (_, _, won, next_states, *_) in enumerate(rows):
        n[s[i], u[i]] += 1
        own[s[i], u[i]] += np.mean(won)
        good = [v for v in next_states if v is not None]
        if good:
            ix = assign(good, *state)
            transition[s[i], u[i]] += np.bincount(ix, minlength=ns) / len(next_states)
    opportunity = np.ones(ns)
    counted = np.zeros(ns)
    successes = np.zeros(ns)
    for _, (key, reward) in coverage:
        i = assign(key, *state)[0]
        counted[i] += 1
        successes[i] += np.mean(reward)
    # Some incoming balls never produce a viable modeled movement. Account for
    # their real loss rate before planning further into a hypothetical rally.
    conditional = (own.sum(axis=1) + 2) / (n.sum(axis=1) + 3)
    opportunity = np.minimum(1.0, ((successes + 3) / (counted + 4)) / conditional)
    # Shrink sparse transitions to measured state and action marginals.
    gs = (transition.sum(axis=(0, 1)) + 0.1) / (n.sum() + 0.1 * ns)
    s_prior = (transition.sum(axis=1) + 8 * gs) / (n.sum(axis=1)[:, None] + 8)
    a_prior = (transition.sum(axis=0) + 8 * gs) / (n.sum(axis=0)[:, None] + 8)
    probs = (transition + 6 * s_prior[:, None] + 6 * a_prior[None]) / (
        n[:, :, None] + 12.0
    )
    immediate = (own + 6 * conditional[:, None]) / (n + 6)
    freq = (n + 1) / (n.sum(axis=1, keepdims=True) + na)
    v = np.zeros(ns)
    for _ in range(80):
        q = immediate + 0.91 * (probs @ v)
        eligible = n >= 5
        soft = (
            freq
            * np.exp(np.clip(1.7 * (q - q.max(axis=1, keepdims=True)), -8, 0))
            * eligible
        )
        weighted = np.divide(
            (soft * q).sum(1), soft.sum(1), out=(freq * q).sum(1), where=soft.sum(1) > 0
        )
        v = opportunity * (0.65 * weighted + 0.35 * (freq * q).sum(1))
    return dict(
        state=[p.tolist() for p in state],
        action=[p.tolist() for p in action],
        q=q.tolist(),
        n=n.tolist(),
        value=v.tolist(),
        opportunity=opportunity.tolist(),
    )


class Markov:
    def __init__(self, d):
        self.state = tuple(map(np.asarray, d["state"]))
        self.action = tuple(map(np.asarray, d["action"]))
        self.q = np.array(d["q"])
        self.n = np.array(d["n"])

    def get(self, state, x):
        s = assign(state, *self.state)[0]
        centers, mu, scale = self.action
        ds = cdist((np.atleast_2d(x) - mu) / scale, centers, "sqeuclidean")
        weight = np.exp(-0.55 * (ds - ds.min(axis=1, keepdims=True))) * (
            self.n[s] / (self.n[s] + 7)
        )
        return weight @ self.q[s] / (1e-8 + weight.sum(1)), np.min(ds, axis=1) < 9


def fit():
    src = _read(value.CHECKPOINT)
    files = [Path(s) for s in src["input_files"]]
    if any(_hash(f) != src["input_files"][str(f)] for f in files):
        raise ValueError("Prior data changed")
    held = {
        c["id"] for deck in ("assessment", "practice") for c in _read(season.PLAN)[deck]
    }
    rows, coverage = collect(files, held)
    releases = np.array([r[-1] for r in rows])
    results = []
    for fold in np.array_split(
        np.random.default_rng(49771).permutation(np.unique(releases)), 4
    ):
        infer = Markov(
            solve(
                [r for r in rows if r[-1] not in fold],
                [r for r in coverage if r[0][0] not in fold],
            )
        )
        actual = []
        pred = []
        conditional = []
        for r in rows:
            if r[-1] in fold:
                actual.append(np.mean(r[4]))
                pred.append(infer.get(r[0], r[1])[0][0])
                conditional.append(infer.q[assign(r[0], *infer.state)[0]].mean())
        actual = np.asarray(actual)

        def spread(p):
            parts = np.array_split(np.argsort(p), 5)
            return [float(actual[ix].mean()) for ix in parts]

        results.append(
            dict(
                n=len(actual),
                state_only=spread(conditional),
                state_and_action=spread(pred),
            )
        )
    result = solve(rows, coverage)
    result.update(
        source=_hash(value.CHECKPOINT),
        examples=len(rows),
        observed_offers=len(coverage),
        folds=results,
        release_ids=np.unique(releases).tolist(),
    )
    ROOT.mkdir(exist_ok=True, parents=True)
    _write(MODEL, result)


class Player(season.Player):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.model = Markov(_read(MODEL))

    def begin(self):
        super().begin()
        self.visits = []

    def search(self, tick, x, q, ref, base):
        if self.visits and tick - self.visits[-1]["tick"] > 12:
            self.visits = []
        self.visits.append(
            dict(tick=tick, p=x[:3].tolist(), bounced=bool(self.engine.bounced))
        )
        self.approach = observed_state(self.visits)
        super().search(tick, x, q, ref, base)

    def robust(self, goals, rank, nominal):
        self.in_rank = True
        try:
            votes, quality = self.uncertainty(goals, nominal)
        finally:
            self.in_rank = False
        x = np.array([self.metrics[np.asarray(g).tobytes()] for g in goals])
        risk, support = self.coverage.risk(x)
        pick, _ = value.fresh.shot.choice(votes, quality, risk, support)
        if self.approach is not None and votes[pick] > 0:
            q, covered = self.model.get(self.approach, x)
            gain = q - q[pick] - 0.12 * np.maximum(0.0, quality - quality[pick])
            allow = (
                (votes >= votes[pick])
                & covered
                & (quality <= quality[pick] + 0.55)
                & (gain > 0.09)
            )
            if covered[pick] and allow.any():
                pick = int(np.argmax(np.where(allow, gain, -np.inf)))
                self.diag["transition_value_edits"] += 1
        return int(rank[pick]), int(votes[pick]), float(quality[pick])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("fit", "live"))
    p.add_argument("out", type=Path, nargs="?")
    p.add_argument("--n", type=int, default=24)
    p.add_argument("--feeds", type=Path, default=Path("/tmp/tt25-rallies.json"))
    p.set_defaults(
        deck="practice", mode="live", carry=season.CARRY, cap=240, resume=False
    )
    a = p.parse_args()
    if a.action == "fit":
        fit()
    else:
        if set(_read(MODEL)["release_ids"]) & {
            c["id"] for c in _read(season.PLAN)[a.deck]
        }:
            raise ValueError("Training and debut overlap")
        with patch.object(season, "Player", Player):
            season.run(a)
        d = _read(a.out)
        d["transitions"] = _hash(MODEL)
        _write(a.out, d)
