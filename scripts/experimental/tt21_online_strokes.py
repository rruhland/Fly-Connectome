"""Grow and score an SO-101 stroke repertoire online from one camera and point rewards."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt8_service_arena import ServiceScore
from .tt20_camera_carry import OUTPUT as PREVIOUS
from .tt20_camera_carry import SETTINGS, Station, summarize, verify

OUTPUT = PREVIOUS.parent / "tt21"
TRAIN = 89173722
GAMES = (89305127, 74419621, 50811532, 107499310)
FACTORY = PREVIOUS / "curriculum-factory.json.gz"
STANDARD_PICK = 3  # Highest point yield of four serves on the prior outside-camera held set.
SCALE = np.array([0.12, 0.18, 0.18, 0.22, 0.17, 0.13, 0.23, 0.23, 0.24, 0.18, 0.9, 0.55, 1.2])


class OnlineBank:
    """Rank local motor variants by public score near the current optical toss context."""

    def __init__(self, programs, *, state=None, learning=True, novel=True, rng=None):
        self.programs = list(map(list, programs)) if state is None else [p.copy() for p in state["programs"]]
        self.rows = [] if state is None else [r.copy() for r in state["rows"]]
        self.parents = list(range(len(programs))) if state is None else state["parents"].copy()
        self.learning, self.novel = learning, novel
        self.rng = np.random.default_rng(28170529) if rng is None else rng
        if state is not None and "rng_state" in state:
            self.rng.bit_generator.state = state["rng_state"]
        self.pending = None
        self.selected = STANDARD_PICK

    def state(self):
        return {
            "programs": self.programs,
            "rows": self.rows,
            "parents": self.parents,
            "rng_state": self.rng.bit_generator.state,
        }

    def values(self, cue):
        n = len(self.programs)
        counts = np.zeros(n)
        points = np.zeros(n)
        if self.rows:
            rows = np.asarray(self.rows)
            distance = (rows[:, :2] - cue) / np.array([0.047, 0.045])
            weight = 0.4 + 0.6 * np.exp(-0.5 * np.sum(distance * distance, axis=1))
            arm = rows[:, 2].astype(int)
            counts = np.bincount(arm, weights=weight, minlength=n)
            points = np.bincount(arm, weights=weight * rows[:, 3], minlength=n)
        # A live trial outweighs the weak neutral prior for each candidate.
        prior = np.full(n, 0.5)
        prior[STANDARD_PICK] = 0.55
        return (prior + points) / (1.0 + counts), counts

    def choose(self, cue, _):
        cue = np.asarray(cue)
        values, counts = self.values(cue)
        if self.learning:
            if self.novel and len(self.rows) >= 12 and len(self.rows) % 6 == 0:
                parent = int(np.argmax(values))
                candidate = np.array(self.programs[parent]) + SCALE * self.rng.normal(size=13)
                candidate[10:12] = np.clip(candidate[10:12], [7.0, 2.0], [21.0, 7.0])
                self.programs.append(candidate.tolist())
                self.parents.append(parent)
                choice = len(self.programs) - 1
            else:
                bonus = 0.56 * np.sqrt(np.log(2 + len(self.rows)) / (1 + counts))
                choice = int(np.argmax(values + bonus + self.rng.uniform(0, 0.007, len(values))))
        else:
            choice = int(np.argmax(values))
        self.selected = choice
        self.pending = [float(cue[0]), float(cue[1]), choice]
        return self.programs[choice]

    def finish(self, score):
        if self.pending is None:
            raise ValueError("serve outcome without an opening choice")
        if self.learning:
            self.rows.append([*self.pending, 0.5 * (1 + score)])
        self.pending = None


def player(eye, strong, bank):
    station = Station(eye, strong=strong)
    station.player.serves = bank
    return station


def point(station, bank, seed, role):
    row = station.point(seed, role, 0)
    if role == 1:
        row["context"] = bank.pending[:2]
        row["candidate"] = bank.selected
        bank.finish(row["reward"])
    else:
        row["candidate"] = None
    return row


def train(eye, strong, count, seed=TRAIN):
    base = _read(FACTORY)["programs"]
    bank = OnlineBank(base)
    station = player(eye, strong, bank)
    rows = []
    started = perf_counter()
    try:
        for seed0 in np.random.default_rng(seed).integers(2**31, size=count):
            rows.append(point(station, bank, int(seed0), 1))
            if len(rows) % 8 == 0:
                print(len(rows), summarize(rows[-8:]), flush=True)
    finally:
        station.close()
    out = OUTPUT / f"online-strokes-{eye}-{'strong' if strong else 'standard'}-{seed}-practice.json.gz"
    data = {
        "eye": eye,
        "arena": SETTINGS,
        "strong": strong,
        "points": rows,
        "bank": bank.state(),
        "seed": seed,
        "factory_sha256": _hash(FACTORY),
        "summary": summarize(rows),
        "elapsed_s": perf_counter() - started,
    }
    _write(out, data)
    return out


def game(eye, strong, source, condition, game_seed):
    checkpoint = None if source is None or condition == "fixed" else _read(source)
    state = None if checkpoint is None else checkpoint["bank"]
    if checkpoint is not None and checkpoint.get("factory_sha256", _hash(FACTORY)) != _hash(FACTORY):
        raise ValueError("source checkpoint uses a different motor library")
    if condition == "shuffled":
        state = {**state, "rows": [r.copy() for r in state["rows"]]}
        results = np.random.default_rng(22041998).permutation([r[3] for r in state["rows"]]).tolist()
        for r, y in zip(state["rows"], results, strict=True):
            r[3] = y
    if condition == "fixed":
        state = None
    bank = OnlineBank(_read(FACTORY)["programs"], state=state, learning=condition in ("continue", "cold"))
    station = player(eye, strong, bank)
    rows = []
    score = ServiceScore(first_server=-1)
    try:
        for seed in np.random.default_rng(game_seed).integers(2**31, size=80):
            r = point(station, bank, int(seed), score.server)
            rows.append(r)
            score.record(r["outcome"])
            if score.finished:
                break
        if not score.finished:
            raise RuntimeError("game cap reached")
    finally:
        station.close()
    data = {
        "eye": eye,
        "arena": SETTINGS,
        "strong": strong,
        "points": rows,
        "seed": game_seed,
        "source": source.name if state is not None else None,
        "source_sha256": _hash(source) if state is not None else None,
        "factory_sha256": _hash(FACTORY),
        "condition": condition,
        "score": list(score.score),
        "summary": summarize(rows),
        "source_points": 0 if state is None else len(state["rows"]),
        "bank": bank.state(),
    }
    data["replayed_points"] = verify(data)
    path = (
        OUTPUT / f"online-strokes-{eye}-{'strong' if strong else 'standard'}-{condition}-{game_seed}.json.gz"
    )
    _write(path, data)
    print(eye, condition, game_seed, data["score"], data["summary"], flush=True)
    return path


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eye", default="left", choices=("left", "wrist"))
    p.add_argument("--strong", action="store_true")
    p.add_argument("--train", type=int)
    p.add_argument("--source", type=Path)
    p.add_argument(
        "--conditions",
        nargs="+",
        choices=("fixed", "learned", "shuffled", "continue", "cold"),
        default=("fixed", "learned"),
    )
    p.add_argument("--seeds", nargs="+", type=int, default=GAMES[:2])
    a = p.parse_args()
    if a.train:
        train(a.eye, a.strong, a.train)
    else:
        for condition in a.conditions:
            previous = None if condition == "cold" else a.source
            if condition in ("learned", "shuffled", "continue") and previous is None:
                p.error("a pretrained condition requires --source")
            for seed in a.seeds:
                saved = game(a.eye, a.strong, previous, condition, seed)
                if condition in ("cold", "continue"):
                    previous = saved
