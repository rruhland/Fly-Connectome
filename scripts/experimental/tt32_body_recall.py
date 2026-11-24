"""Use rewarded joint configurations to propose new contacts after ordinary stroke search fails."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from . import tt29_action_memory as motor
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import desired_shot
from .tt31_curriculum import play_card
from .tt31_ready import centered
from .tt31_recovery import Recovery


class BodyRecall(motor.RecallPlanner):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache = None
        self.trials = []

    def body_samples(self):
        # The joint positions and rates come from the learner's own rewarded motor tapes.
        if self.cache is not None:
            return self.cache
        samples = []
        dt = self.arena.control_dt
        for example in self.memory.examples:
            j = int(
                np.clip(
                    round(example.time_to_contact / dt) - 1, 1, len(example.targets) - 2
                )
            )
            for shift in (-2, 0, 2):
                i = int(np.clip(j + shift, 1, len(example.targets) - 2))
                joints = example.targets[i, :5]
                speed = (example.targets[i + 1, :5] - example.targets[i - 1, :5]) / (
                    2 * dt
                )
                position, normal = self.ik.pose(joints)
                samples.append((position, normal, joints, speed))
        self.cache = (
            tuple(np.stack(column) for column in zip(*samples)) if samples else ()
        )
        return self.cache

    def plan(self):
        tape = super().plan()
        if tape is not None:
            return tape
        a = self.arena
        path = self.flight.path(a.ball, a.velocity, a.spin, 1, a._received_bounce)
        region = path[
            (path[:, 2] >= -1.52)
            & (path[:, 2] <= -1.13)
            & (path[:, 3] > 0.82)
            & (path[:, 3] < 1.36)
            & (path[:, 0] > 0.045)
        ]
        samples = self.body_samples()
        if len(region) == 0 or not samples:
            return None
        positions, normals, qs, speeds = samples
        dt = a.control_dt
        start = a.data.ctrl[self.axes]
        times = np.arange(1, 1 + int(np.ceil((region[-1, 0] + 0.65) / dt))) * dt
        windows = region[
            np.unique(np.linspace(0, len(region) - 1, min(6, len(region))).astype(int))
        ]
        goals = []
        targets = np.array([0.0, 1.32, 1.08, 3.6])
        for w in windows:
            t, pos, incoming = w[0], w[1:4], w[4:7]
            normal = desired_shot(pos, incoming, 1, (0.0, 0.94), 5.4)
            distance = (
                np.linalg.norm(positions - pos, axis=1)
                + 0.15 * np.linalg.norm(normals - normal, axis=1)
                + 0.03 * np.max(abs(qs - start[:5]), axis=1)
            )
            selected = []
            for i in np.argsort(distance):
                if all(np.max(abs(qs[i] - qs[j])) > 0.28 for j in selected):
                    selected.append(i)
                if len(selected) == 3:
                    break
            for i in selected:
                for tilt in (-0.08, 0.12):
                    face = normal + [0.0, 0.0, tilt]
                    face /= np.linalg.norm(face)
                    q, (at, n, jac, ang) = self.ik.fit(pos - 0.025 * face, face, qs[i])
                    if np.linalg.norm(at - pos) > 0.093 or n @ face < 0.78:
                        continue
                    direction = np.linalg.lstsq(
                        np.vstack((jac, 0.042 * ang, 0.013 * np.eye(5))),
                        np.r_[face, np.zeros(8)],
                        rcond=None,
                    )[0]
                    for speed, lead in [
                        (speeds[i], 0.02),
                        (0.95 * direction, 0.05),
                        (np.zeros(5), 0.0),
                    ]:
                        slope = np.clip(speed, -self.rates[:5], self.rates[:5])
                        body = (
                            q + np.clip(times - t + lead, -0.13, 0.09)[:, None] * slope
                        )
                        goals.append(
                            np.column_stack((body, np.full(len(times), start[5])))
                        )
        if not goals:
            return None
        rates, scores = self.predict.trajectories(
            np.stack(goals), targets, region[-1, 0] + 0.08
        )
        valid = np.flatnonzero(np.isfinite(scores[:, 0]))
        self.trials.append([a.steps, len(goals), len(valid)])
        if not len(valid):
            return None
        best = valid[np.argmin(scores[valid, 0])]
        self.fallback += 1
        return rates[best]


def run(feeds, reference, checkpoint, output, indices):
    source = _read(reference)
    if source["checkpoint_sha256"] != _hash(checkpoint):
        raise ValueError("Baseline checkpoint differs")
    result = {
        "ready": source["ready"],
        "opening_side": 1,
        "cards": [],
        "rows": [],
        "reference_sha256": _hash(reference),
    }
    for index in indices:
        card = source["cards"][index]
        objects = []

        def create(*args, instances=objects, **kwargs):
            model = BodyRecall(*args, **kwargs)
            instances.append(model)
            return model

        with (
            centered(source["ready"]),
            patch.object(motor, "RecallPlanner", create),
            patch.object(motor, "RallyObjective", Recovery),
            TemporaryDirectory() as tmp,
        ):
            row, _ = play_card(
                feeds,
                card,
                checkpoint,
                checkpoint,
                Path(tmp) / "result.json.gz",
                practice=False,
                cap=16,
            )
        row["body_search"] = objects[0].trials
        result["cards"].append(card)
        result["rows"].append(row)
        _write(output, result)
        print(card["id"], row["score"]["learner"], row["body_search"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("feeds", "reference", "checkpoint", "output"):
        parser.add_argument(key, type=Path)
    parser.add_argument("indices", nargs="+", type=int)
    args = parser.parse_args()
    run(args.feeds, args.reference, args.checkpoint, args.output, args.indices)
