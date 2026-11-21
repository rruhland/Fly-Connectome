"""Learn a small visual stroke proposal library from scored SO-101 practice."""

import argparse
from itertools import product
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_factory import CONTACT_OFFSET, moving_joint_rate
from .tt19_flight import optical_start
from .tt19_shoot import HORIZON
from .tt22_rally import DEPTHS, PITCHES, PACES, LEADS, RallyPlayer, face, incoming_crossings, rates_for
from .tt22_rally_trial import Station


def context(crossings):
    if not crossings:
        return None
    eta, xyz, velocity = min(crossings, key=lambda a: abs(a[1][1] + 0.92))
    return np.r_[xyz[0] / 0.08, xyz[2] / 0.08, velocity / [0.008, 0.018, 0.013], eta / 8.0]


def examples(row, law):
    samples = dict(row["optical"])
    notes = row["shots"]
    results = []
    for index, note in enumerate(notes):
        tick = note["tick"]
        stop = notes[index + 1]["tick"] if index + 1 < len(notes) else len(row["actions"]) + 1
        good = any(
            tick < e["step"] <= stop and e["kind"] == "legal_return" and e["side"] == 1 for e in row["events"]
        )
        if not good:
            continue
        history = [(i, samples.get(i)) for i in range(max(0, tick - 6), tick + 1)]
        est = optical_start(history, law.gravity)
        if est is None:
            continue
        state = context(list(incoming_crossings(law, *est)))
        if state is None:
            continue
        results.append(
            dict(context=state.tolist(), recipe=[note[k] for k in ("depth", "pitch", "pace", "lead")])
        )
    return results


class MemoryPlayer(RallyPlayer):
    def __init__(self, law, records, budget):
        super().__init__(law)
        self.records, self.budget = records, budget
        combos = list(product(DEPTHS, PITCHES, PACES, LEADS))
        self.uniform = [combos[i] for i in np.random.default_rng(34802).permutation(len(combos))]

    def goal_tapes(self, estimate, target):
        crossings = list(incoming_crossings(self.law, *estimate))
        cue = context(crossings)
        if cue is None:
            return np.empty((0, HORIZON, 5)), []
        ordered = sorted(self.records, key=lambda row: np.linalg.norm(cue - row["context"]))
        proposals = [tuple(row["recipe"]) for row in ordered] + self.uniform
        goals, parameters, seen, poses = [], [], set(), {}
        timeline = np.arange(HORIZON)
        for depth, pitch, pace, lead in proposals:
            eta, center, speed = min(crossings, key=lambda item: abs(item[1][1] - depth))
            if abs(center[1] - depth) > 0.001:
                continue
            key = (depth, pitch, pace, lead)
            if key in seen:
                continue
            seen.add(key)
            yaw = float(np.clip(-7.0 * speed[0] - 0.45 * center[0], -0.42, 0.42))
            if (depth, pitch) not in poses:
                normal = face(pitch, yaw)
                anchor = center - CONTACT_OFFSET * normal
                pose = self.factory.solve(anchor, normal, target)
                recovered, facing, _, _ = self.factory.pose(pose)
                error = np.linalg.norm(recovered - anchor)
                poses[depth, pitch] = (pose, error <= 0.057 and facing @ normal >= 0.77, float(error))
            pose, ok, error = poses[depth, pitch]
            if not ok:
                continue
            rate = moving_joint_rate(self.factory, pose, pace)
            phase = np.clip(timeline - eta + lead, -3.1, 3.4)
            goals.append(pose + phase[:, None] * rate)
            parameters.append(
                dict(depth=depth, pitch=pitch, pace=pace, lead=lead, eta=eta, yaw=yaw, alignment=error)
            )
            if len(goals) >= self.budget:
                break
        if not goals:
            return np.empty((0, HORIZON, 5)), []
        return rates_for(np.asarray(goals), target, self.factory.low, self.factory.high), parameters


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--practice", type=Path, nargs="*", default=[])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seed", type=int, default=8351422)
    p.add_argument("--budget", type=int, default=12)
    p.add_argument("--count", type=int, default=5)
    p.add_argument("--online", action="store_true")
    args = p.parse_args()
    station = Station(0.30, 0.82, "anchor")
    law = station.player.law
    station.player.close()
    records = []
    prior_seeds = set()
    for path in args.practice:
        data = _read(path)
        if data.get("privileged", True):
            raise ValueError("use optical practice only")
        for row in data["rows"]:
            if row["seed"] not in prior_seeds:
                records.extend(examples(row, law))
                prior_seeds.add(row["seed"])
    station.player = MemoryPlayer(law, records, args.budget)
    report = dict(
        initial_records=records.copy(),
        prior_seeds=sorted(prior_seeds),
        online=args.online,
        candidate_budget=args.budget,
        seed=args.seed,
        rows=[],
    )
    try:
        for seed in np.random.default_rng(args.seed).integers(2**31, size=args.count):
            if seed in prior_seeds:
                raise ValueError("practice/evaluation overlap")
            start = perf_counter()
            row = station.point(int(seed))
            row["seconds"] = perf_counter() - start
            report["rows"].append(row)
            if args.online:
                records.extend(examples(row, law))
            report["final_records"] = records
            _write(args.output, report)
            print(int(seed), row["longest"], len(row["shots"]), round(row["seconds"], 2), flush=True)
    finally:
        station.close()


if __name__ == "__main__":
    main()
