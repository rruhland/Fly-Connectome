"""Learn which physical stroke to issue from scored SO-101 experience, without candidate contact simulation."""

import argparse
from itertools import product
from pathlib import Path
from time import process_time

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_factory import CONTACT_OFFSET, moving_joint_rate
from .tt19_flight import optical_start
from .tt19_shoot import HORIZON
from .tt22_rally import LEADS, PACES, PITCHES, RallyPlayer, face, incoming_crossings, rates_for
from .tt22_rally_trial import Station
from .tt22_stroke_memory import context


def learn(row, law):
    samples = dict(row["optical"])
    notes = row["shots"]
    records = []
    for i, note in enumerate(notes):
        stop = notes[i + 1]["tick"] if i + 1 < len(notes) else len(row["actions"]) + 1
        label = any(
            note["tick"] < e["step"] <= stop and e["kind"] == "legal_return" and e["side"] == 1
            for e in row["events"]
        )
        history = [(k, samples.get(k)) for k in range(max(0, note["tick"] - 6), note["tick"] + 1)]
        estimate = optical_start(history, law.gravity)
        if estimate:
            cue = context(list(incoming_crossings(law, *estimate)))
            if cue is not None:
                records.append(
                    dict(
                        cue=cue.tolist(),
                        recipe=[note[k] for k in ("depth", "pitch", "pace", "lead")],
                        returned=label,
                    )
                )
    return records


class RecipePlayer(RallyPlayer):
    def __init__(self, law, records):
        super().__init__(law)
        self.records = records
        self.rng = np.random.default_rng(0)

    def search(self, estimate, observation, target):
        paths = list(incoming_crossings(self.law, *estimate))
        cue = context(paths)
        if cue is None:
            return None
        grid = list(product([x[1][1] for x in paths], PITCHES, PACES, LEADS))
        if self.records:
            xx = np.array([r["cue"] for r in self.records])
            yy = np.array([r["returned"] for r in self.records])
            actions = np.array([r["recipe"] for r in self.records])
            w = np.exp(-0.5 * np.sum(((xx - cue) / [1.1, 1.8, 2.4, 1.5, 2.0, 1.8]) ** 2, axis=1))
            difference = (np.asarray(grid)[:, None] - actions) / [0.035, 0.15, 0.015, 0.85]
            weight = w * np.exp(-0.5 * np.sum(difference * difference, axis=2))
            value = (weight @ yy + 0.10) / (weight.sum(axis=1) + 0.40)
            order = np.argsort(-(value + self.rng.random(len(value)) * 0.0001))
        else:
            order = self.rng.permutation(len(grid))
        for j in order:
            depth, pitch, pace, lead = grid[j]
            eta, center, speed = min(paths, key=lambda x: abs(x[1][1] - depth))
            yaw = self.rally_yaw(center, speed)
            normal = face(pitch, yaw)
            anchor = center - CONTACT_OFFSET * normal
            pose = self.factory.solve(anchor, normal, target)
            actual, facing, _, _ = self.factory.pose(pose)
            if np.linalg.norm(actual - anchor) > 0.057 or facing @ normal < 0.77:
                continue
            rate = moving_joint_rate(self.factory, pose, pace)
            length = int(np.clip(np.ceil(eta) + 3, 3, HORIZON))
            phase = np.clip(np.arange(length) - eta + lead, -3.1, 3.4)
            goals = pose + phase[:, None] * rate
            tape = rates_for(goals[None], target, self.factory.low, self.factory.high)[0]
            return tape, dict(
                pitch=pitch,
                depth=depth,
                pace=pace,
                lead=lead,
                eta=eta,
                anchor=center.tolist(),
                prediction={"eta": eta},
                experience=len(self.records),
            )
        return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--practice", type=Path, nargs="*", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=3169881)
    p.add_argument("--count", type=int, default=10)
    p.add_argument("--online", action="store_true")
    args = p.parse_args()
    station = Station(0.30, 0.82, "anchor")
    law = station.player.law
    station.player.close()
    records = []
    seen = set()
    for path in args.practice:
        data = _read(path)
        if data["privileged"]:
            raise ValueError("training requires optical practice")
        for row in data.get("rows", data.get("practice", [])):
            if row["seed"] not in seen:
                records.extend(learn(row, law))
                seen.add(row["seed"])
    station.player = RecipePlayer(law, records)
    data = dict(
        seed=args.seed,
        initial_records=list(records),
        practice=sorted(seen),
        online=args.online,
        rows=[],
        rival="anchor",
        spread=0.30,
        pace=0.82,
        privileged=False,
    )
    try:
        for seed in np.random.default_rng(args.seed).integers(2**31, size=args.count):
            if int(seed) in seen:
                raise ValueError("evaluation overlaps practice")
            station.player.rng = np.random.default_rng(int(seed))
            start = process_time()
            row = station.point(int(seed))
            row["cpu_seconds"] = process_time() - start
            data["rows"].append(row)
            if args.online:
                records.extend(learn(row, law))
            data["records"] = records
            _write(args.output, data)
            print(int(seed), row["longest"], len(records), flush=True)
    finally:
        station.close()


if __name__ == "__main__":
    main()
