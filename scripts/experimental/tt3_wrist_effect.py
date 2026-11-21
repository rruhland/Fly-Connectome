"""Learn how an optically redundant actuator changes the motion of another track."""

import argparse
from collections import deque
import gzip
import json
from pathlib import Path

import numpy as np

from tt3_arena import heldout_cases
from tt3_learning import play_point
from tt3_numeric_transfer import PortableMemory
from tt3_visual import RoleModel
from tt3_wrist import WristArena
from tt3_wrist_learning import WristPlayer


def slope(points):
    t = np.arange(len(points), dtype=float)
    t -= t.mean()
    return t @ np.array(points) / (t @ t)


class OpticalEffectPlayer(WristPlayer):
    def __init__(self, *args, effect=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.effect = effect
        self.examples = []
        self.rewarded = []
        self.fits = 0

    def reset(self, option=None):
        super().reset(option)
        self.window = deque(maxlen=14)
        self.last_event = -100
        self.waiting = None
        self.last_near = None
        self.direction = None
        self.initial_approach = True
        # Excite a visually redundant channel during initial identification.
        self.intervention = (0.0, -2.8, 2.8, 1.5, -1.5)[self.outer_points % 5]
        self.fit_count = 0
        self.aim = 0.0

    def _events(self, point, controlled, direction):
        self.window.append((point.copy(), controlled.copy(), self.redundant_motion))
        if len(self.window) < 11:
            return
        pts = np.array([p for p, _, _ in self.window])
        actors = np.array([p for _, p, _ in self.window])
        coord = np.array([v for _, _, v in self.window])
        mid = len(pts) - 6
        vin = slope(pts[mid - 5 : mid - 1]) * 60
        vout = slope(pts[mid + 1 : mid + 5]) * 60
        station = np.linalg.norm(pts[mid] - actors[mid])
        if self.player.frame - self.last_event < 11:
            return
        if vin @ direction < -0.24 and vout @ direction > 0.24 and station < 0.093:
            vbody = slope(actors[mid - 3 : mid + 2]) * 60
            self.examples.append((vin, vbody, coord[mid], vout))
            self.last_near = len(self.examples) - 1
            self.last_event = self.player.frame
            self.initial_approach = False
        elif vin @ direction > 0.24 and vout @ direction < -0.24 and station > 0.19:
            if self.last_near is not None:
                self.rewarded.append(self.last_near)
                self.last_near = None
            self.last_event = self.player.frame

    def _target(self, vin, vbody):
        selected = self.examples[-100:]
        if len(selected) < 10 or len(self.rewarded) < 3:
            return None
        x = np.array([[1.0, *a, *b, q / 4.0] for a, b, q, _ in selected])
        y = np.stack([c for _, _, _, c in selected])
        if np.std(x[:, -1]) < 0.14:
            return None
        distance = np.linalg.norm(x[:, 1:4] - vin, axis=1)
        w = 1.0 / (1.0 + distance**2)
        ridge = np.array([0.03, 0.1, 0.1, 0.1, 0.5, 0.5, 0.5, 0.10])
        model = np.linalg.solve(
            x.T @ (w[:, None] * x) + np.diag(ridge), x.T @ (w[:, None] * y)
        )
        successful = [self.examples[i] for i in self.rewarded[-50:]]
        desired = np.median([e[3] for e in successful], axis=0)
        neutral = np.array([1.0, *vin, *vbody, 0.0]) @ model
        answer = 4 * (desired - neutral) @ model[-1] / (model[-1] @ model[-1] + 0.05)
        self.fits += 1
        self.fit_count += 1
        return float(np.clip(answer, -7.5, 7.5))

    def _weak_rate(self):
        bank = self.player.flight
        normal = self.player.normal
        if (
            normal is None
            or self.player.ready is None
            or not hasattr(bank, "tail")
            or len(bank.tail) < 4
        ):
            return float(np.clip(-self.redundant_motion, -1, 1))
        position = bank.tail[-1]
        controlled = self.player.history[-1].positions.get(self.player.identity)
        if controlled is None:
            return 0.0
        speed = slope(list(bank.tail)[-4:])
        gap = (position - self.player.ready) @ normal
        if self.direction is None:
            if abs(gap) < 0.07 or (speed @ normal) * gap >= 0:
                return 0.0
            self.direction = np.sign(gap) * normal
        self._events(position, controlled, self.direction)
        closing = -speed @ self.direction
        distance = (position - controlled) @ self.direction
        if self.initial_approach:
            self.aim = (
                self.intervention
                if (len(self.examples) < 20 or self.outer_points % 7 == 0)
                else 0.0
            )
        elif closing > 0.003 and 0.015 < distance < 0.16:
            self.aim = self.intervention
            if self.effect and len(self.window) >= 4:
                command = self._target(
                    60 * speed, 60 * slope([a for _, a, _ in self.window][-4:])
                )
                if command is not None:
                    self.aim = command
        return float(np.clip(self.aim - self.redundant_motion, -1, 1))

    def finish(self):
        if self.last_near is not None and self.total_reward > 0:
            self.rewarded.append(self.last_near)
        return dict(
            **super().finish(),
            example_count=len(self.examples),
            positive_count=len(self.rewarded),
            fits=self.fit_count,
        )


def course(seed, count, roles, memory, enabled):
    cases = heldout_cases(seed, count)
    learner = OpticalEffectPlayer(
        RoleModel(**roles),
        PortableMemory.restore(memory),
        seed,
        policy="locked",
        effect=enabled,
    )
    points = []
    for c in cases:
        g = WristArena(c.seed, view_seed=cases[0].view_seed, wiring_seed=seed)
        r = play_point(g, learner)
        points.append(
            {
                k: r[k]
                for k in (
                    "legal_returns",
                    "rival_returns",
                    "contacts",
                    "points_for",
                    "fits",
                    "example_count",
                    "positive_count",
                )
            }
        )
    return points


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("roles", type=Path)
    p.add_argument("numeric", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, default=92400311)
    p.add_argument("--matches", type=int, default=2)
    p.add_argument("--points", type=int, default=48)
    a = p.parse_args()
    roles = json.loads(a.roles.read_text())["role_model"]
    source = json.loads(a.numeric.read_text())
    runs = []
    for j in range(a.matches):
        m = {}
        for label in ["predict", "explore"]:
            rs = course(a.seed + j, a.points, roles, source, label == "predict")
            m[label] = rs
            print(
                j,
                label,
                "legal",
                sum(x["legal_returns"] > 0 for x in rs),
                "later",
                sum(max(0, x["legal_returns"] - 1) for x in rs),
                "rival",
                sum(x["rival_returns"] for x in rs),
                "wins",
                sum(x["points_for"] for x in rs),
                "updates",
                rs[-1]["example_count"],
                rs[-1]["positive_count"],
                sum(x["fits"] for x in rs),
                flush=True,
            )
        runs.append(m)
        with gzip.open(a.out, "wt") as f:
            json.dump(dict(seed=a.seed, runs=runs), f, separators=(",", ":"))
