"""Find an optically redundant action channel and learn its utility from point scores."""

import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import numpy as np
from scipy.linalg import hadamard

from tt3_arena import heldout_cases
from tt3_learning import play_point
from tt3_numeric_transfer import VisualPlayer, PortableMemory
from tt3_visual import RoleBinder, RoleModel
from tt3_wrist import WristArena


class WristPlayer:
    """An unlabeled channel with little centroid motion can still affect outcomes."""

    def __init__(self, roles, memory, seed, policy="learn"):
        self.policy = policy
        self.player = VisualPlayer(
            roles, memory, seed=seed, forecast=True, recall=True, learn_roles=True
        )
        self.calibration = RoleBinder(roles)
        # Opposite, zero-sum actions return the body to its starting configuration.
        plus = hadamard(4).astype(float)
        self.probes = np.repeat(np.concatenate((plus, -plus, -plus, plus)), 2, axis=0)
        self.null = None
        self.centers = []
        self.actions = []
        self.speeds = []
        self.rng = np.random.default_rng(seed)
        self.candidates = np.array([0.0, -3.0, 3.0, -1.4, 1.4, -5.0, 5.0])
        self.values = np.zeros(len(self.candidates))
        self.exposures = np.zeros(len(self.candidates))
        self.outer_points = 0

    def reset(self, option=None):
        self.player.reset()
        self.calibration.reset()
        self.previous = None
        self.paths = {}
        self.actions = []
        self.redundant_motion = 0.0
        self.speed_trace = []
        self.reference = None
        self.outbound = False
        self.measuring = False
        self.influenced = False
        self.total_reward = 0.0
        if option is not None:
            self.option = int(option)
        elif self.policy == "learn":
            # Exploration only when previous returns supply no reliable preference.
            score = self.values / (self.exposures + 1.0) + 0.75 / np.sqrt(
                self.exposures + 1.0
            )
            self.option = int(
                self.rng.choice(np.flatnonzero(np.isclose(score, score.max())))
            )
        else:
            self.option = 0

    def _identify(self, images):
        bound = self.calibration.update(images, self.previous)
        for track in bound.tracks:
            if track.observed:
                self.paths.setdefault(track.id, []).append(
                    (len(self.actions), np.array(track.position))
                )
        if len(self.actions) >= len(self.probes):
            track = bound.controlled
            if track is None:
                # Labels come only from the visual action model on later frames.
                return np.zeros(4)
            series = self.paths[track.id]
            rows = [
                (self.actions[j - 1], b - a)
                for (i, a), (j, b) in zip(series, series[1:])
                if j == i + 1
            ]
            if len(rows) >= 12:
                acts, displacements = map(np.array, zip(*rows))
                fitted = np.linalg.lstsq(acts, displacements, rcond=0.02)[0]
                self.null = int(np.argmin(np.linalg.norm(fitted, axis=1)))
                self.centers = [i for i in range(4) if i != self.null]
                self.player.reset()
                return np.zeros(4)
        return self.probes[len(self.actions) % len(self.probes)].copy()

    def _weak_rate(self):
        flight = self.player.flight
        normal = self.player.normal
        if (
            normal is None
            or self.player.ready is None
            or not hasattr(flight, "tail")
            or len(flight.tail) < 4
        ):
            return 0.0
        recent = np.array(flight.tail)[-4:]
        gap = (recent[-1] - self.player.ready) @ normal
        closing = -np.sign(gap) * ((recent[-1] - recent[0]) @ normal) / 3
        if closing > 0.0025 and not self.outbound:
            self.speed_trace.append(closing)
            if len(self.speed_trace) >= 5:
                self.reference = float(np.median(self.speed_trace))
        if closing < -0.004 and self.reference is not None:
            self.outbound = True
        if self.outbound and closing > 0.003 and self.reference is not None:
            self.measuring = True
            # Only observed motion speed and source-learned identity define this context.
            self.relative_speed = np.clip(closing / self.reference - 0.85, 0.12, 1.0)
        aim = 0.0
        if self.measuring:
            aim = self.candidates[self.option] * self.relative_speed
            self.influenced = True
        # The internal coordinate is the integral of its own bounded requests.
        command = np.clip(aim * 3.8 - self.redundant_motion, -1.0, 1.0)
        return float(command)

    def act(self, images):
        if self.null is None:
            a = self._identify(images)
            self.actions.append(a.copy())
        else:
            a = np.zeros(4)
            a[self.centers] = self.player.act(images)
            a[self.null] = self._weak_rate()
            self.redundant_motion += a[self.null]
        self.previous = a
        return a

    def observe(self, reward, *, terminal_frames=None):
        self.total_reward += reward
        if self.null is not None:
            self.player.observe(reward, terminal_frames=terminal_frames)

    def finish(self):
        if self.influenced and self.policy == "learn":
            self.exposures[self.option] += 1.0
            self.values[self.option] += self.total_reward
        self.outer_points += 1
        r = self.player.finish()
        r.update(
            aux_option=int(self.option),
            null_channel=self.null,
            adjusted=self.influenced,
            public_reward=self.total_reward,
        )
        return r


def one(seed, count, role, source, mode):
    p = WristPlayer(
        RoleModel(**role), PortableMemory.restore(source), seed, policy=mode
    )
    cases = heldout_cases(seed, count)
    rows = []
    for case in cases:
        arena = WristArena(case.seed, view_seed=cases[0].view_seed, wiring_seed=seed)
        r = play_point(arena, p)
        row = {
            k: r[k]
            for k in (
                "legal_returns",
                "contacts",
                "rival_returns",
                "points_for",
                "public_reward",
                "aux_option",
                "null_channel",
                "adjusted",
            )
        }
        row["frames"] = arena.frames
        # Hidden wiring enters only the evaluation receipt.
        row["identified"] = p.null == int(arena.order[-1])
        rows.append(row)
    return dict(points=rows, score=p.values.tolist(), counts=p.exposures.tolist())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roles", type=Path)
    parser.add_argument("source", type=Path)
    parser.add_argument("out", type=Path)
    parser.add_argument("--seed", type=int, default=26783406)
    parser.add_argument("--matches", type=int, default=2)
    parser.add_argument("--points", type=int, default=36)
    args = parser.parse_args()
    roles = json.loads(args.roles.read_text())["role_model"]
    source = json.loads(args.source.read_text())
    results = []
    for i in range(args.matches):
        m = {}
        for mode in ("learn", "locked"):
            m[mode] = one(args.seed + i, args.points, roles, source, mode)
            rec = m[mode]["points"]
            print(
                i,
                mode,
                "legal",
                sum(x["legal_returns"] > 0 for x in rec),
                "repeat",
                sum(max(0, x["legal_returns"] - 1) for x in rec),
                "rival",
                sum(x["rival_returns"] for x in rec),
                "wins",
                sum(x["points_for"] for x in rec),
                "identified",
                sum(x["identified"] for x in rec),
                flush=True,
            )
        results.append(m)
        with gzip.open(args.out, "wt") as f:
            json.dump(dict(seed=args.seed, matches=results), f, separators=(",", ":"))
