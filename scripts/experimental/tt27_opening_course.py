"""Evaluate cross-environment visual feedback on new perturbations and original recorded balls."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt26_grip_course import ReferenceReceiver, load_match
from .tt26_receiving_speed import draw
from .tt27_flight_feedback import OUT, SOURCE, Player, load_bank, play


class Physics:
    def __init__(self, arena):
        self.arena = arena
        table = arena.model.geom("table")
        self.contact = table.solref.copy()
        self.gravity = arena.model.opt.gravity.copy()

    def set(self, surface, gravity):
        self.arena.model.geom("table").solref[:] = self.contact * np.array([1.0, surface])
        self.arena.model.opt.gravity[:] = self.gravity * gravity


POLICIES = {
    "frozen": (0.0, False),
    "tracking": (1.0, False),
    "moderate": (0.45, False),
    "switch": (0.75, True),
}
WORLDS = {"standard": (1.0, 1.0), "soft": (2.5, 1.0), "heavy": (1.3, 1.12)}


def group_job(args):
    path, world, group = args
    lookup, _ = load_match(path)
    actor = Player()
    donors = [r["id"] for r in load_bank()["rows"]]
    rng = np.random.default_rng(174841)
    if group == "beginners":
        shots = [draw(lookup, donors, rng, t) for t in range(40)]
    else:
        roster = _read(SOURCE / "fresh-recordings.json.gz")["records"][0]["rows"]
        shots = [lookup[r["id"]] for r in roster if r["group"] == group][:48]
    arena = ReferenceReceiver()
    physics = Physics(arena)
    physics.set(*WORLDS[world])
    rows = []
    try:
        for shot in shots:
            scored = {}
            for label, (gain, switch) in POLICIES.items():
                r = play(arena, shot, actor, gain=gain, reselect=switch)
                scored[label] = r
            rows.append(dict(id=shot["id"], results=scored))
    finally:
        arena.close()
    print(world, group, {k: sum(r["results"][k]["legal"] for r in rows) for k in POLICIES}, flush=True)
    return dict(world=world, group=group, rows=rows)


def run(path, groups):
    jobs = [(path, k, g) for k in WORLDS for g in groups]
    with ProcessPoolExecutor(max_workers=2, mp_context=get_context("spawn")) as pool:
        result = list(pool.map(group_job, jobs))
    _write(OUT / ("opening-" + "-".join(groups) + ".json.gz"), dict(jobs=result, worlds=WORLDS))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    p.add_argument("--groups", nargs="+", default=["beginners"])
    a = p.parse_args()
    run(a.feeds, a.groups)
