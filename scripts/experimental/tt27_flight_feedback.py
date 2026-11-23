"""Retarget successful receiving actions from ordinary binocular ball observations."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_camera import PerspectiveCamera, triangulate
from .tt4_transfer_trial import ROOT, _read, _write
from .tt19_view_trial import CAMERAS
from .tt24_rally_policy import ViewHistory
from .tt24_rally_trial import PublicBody, STEP_SIZE
from .tt25_pose_demonstrations import OBSERVE
from .tt26_grip_course import ReferenceReceiver, load_match
from .tt26_optical_transfer import decoder

OUT = ROOT / "artifacts/tt27"
SOURCE = ROOT / "artifacts/tt26"
PRIOR = ROOT / "artifacts/tt25"


class Eyes:
    def __init__(self, arena):
        self.view = ViewHistory()
        self.cameras = tuple(
            PerspectiveCamera(eye, aim, arena.width, arena.height, fov)
            for _, eye, aim, fov in (CAMERAS[n] for n in ("left", "right"))
        )
        self.scale = arena.width / 480

    def observe(self, arena, body):
        seen = self.view.observe(arena.render_cameras(), arena.joints, body.target)
        px = [h[-1] for h in self.view.views]
        if any(p is None for p in px):
            return seen[:14].copy(), None
        value = triangulate(self.cameras, *[(p + 0.5) * self.scale - 0.5 for p in px])
        if value is None or not (-2.25 < value[1] < 2.75 and -0.15 < value[2] < 1.45):
            return seen[:14].copy(), None
        return seen[:14].copy(), np.array(value)


def derivative(rig, goal):
    def transform(q):
        p, r = rig.links.read(q[None, :])
        return np.r_[p[0] + r[0] @ rig.offset[3:], r[0] @ rig.normal]

    perturb = 0.001 * np.eye(5)
    jac = np.stack([(transform(goal + d) - transform(goal - d)) / 0.002 for d in perturb], axis=1)
    weight = np.diag([1.0, 1.0, 1.0, 0.15, 0.15, 0.15])
    design = weight @ jac
    return np.linalg.solve(design.T @ design + 0.0007 * np.eye(5), design.T @ weight)


def command(goal, body):
    u = np.zeros(6)
    if goal is not None:
        u[:5] = np.clip((goal - body.target[:5]) / STEP_SIZE[:5], -1, 1)
    return u


def load_bank():
    return _read(OUT / "demonstrations.json.gz")


def capture(path):
    lookup, _ = load_match(path)
    with np.load(SOURCE / "match-broader.npz") as saved:
        keys, goals, scale, extra = [saved[k] for k in ("keys", "goals", "scale", "added_ids")]
    ids = _read(PRIOR / "recorded-nearby-lessons.json.gz")["donors"] + extra.tolist()
    arena = ReferenceReceiver()
    rig = decoder()
    rows = []
    try:
        for sid, key, goal in zip(ids, keys, goals, strict=True):
            arena.launch(lookup[sid])
            eye, body = Eyes(arena), PublicBody(arena)
            trace = []
            for t in range(arena.max_steps):
                k, p = eye.observe(arena, body)
                if t == OBSERVE and not np.allclose(k, key):
                    raise ValueError("Training signature changed")
                if p is not None:
                    trace.append([t, *p])
                u = command(goal if t >= OBSERVE else None, body)
                _, _, done, cap, _ = arena.step(u, render=False)
                body.issued(u)
                if done or cap or arena.legal_returns:
                    break
            if not arena.legal_returns:
                raise ValueError("Original teacher failed to reproduce")
            hit = next(e for e in arena.events if e["kind"] == "contact" and e.get("side") == 1)
            cloud = [r for r in trace if r[0] < hit["step"]]
            if len(cloud) < 7:
                raise ValueError("Insufficient visible approach")
            rows.append(
                dict(
                    id=sid,
                    key=key.tolist(),
                    goal=goal.tolist(),
                    trace=cloud,
                    jacobian=derivative(rig, goal).tolist(),
                )
            )
        _write(OUT / "demonstrations.json.gz", dict(rows=rows, scale=scale.tolist()))
    finally:
        arena.close()


class Template:
    def __init__(self, row):
        self.goal = np.array(row["goal"])
        self.key = np.array(row["key"])
        self.matrix = np.array(row["jacobian"])
        trace = np.array(row["trace"])
        # Monotone inbound samples only. A missed camera frame has no entry.
        good = np.r_[True, np.diff(np.minimum.accumulate(trace[:, 2])) < -0.01]
        self.trace = trace[good, 1:]
        self.y = self.trace[::-1, 1].copy()
        self.final_y = self.y[0]

    def difference(self, point):
        x, y, z = point
        if not self.final_y - 0.06 < y < self.y[-1] + 0.03:
            return None
        return np.array(
            [x - np.interp(y, self.y, self.trace[::-1, 0]), z - np.interp(y, self.y, self.trace[::-1, 2])]
        )


class Player:
    def __init__(self):
        archive = load_bank()
        self.templates = [Template(r) for r in archive["rows"]]
        self.keys = np.stack([t.key for t in self.templates])
        self.scale = np.array(archive["scale"])

    def choose(self, k):
        return int(np.argmin(np.linalg.norm((self.keys - k) / self.scale, axis=1)))


def play(
    arena,
    shot,
    player,
    *,
    gain=1.0,
    reselect=False,
):
    arena.launch(shot)
    body = PublicBody(arena)
    eye = Eyes(arena)
    track = []
    residuals = []
    template = None
    index = None
    goal = None
    actions = []
    encoders = []
    for t in range(arena.max_steps):
        key, pos = eye.observe(arena, body)
        if pos is not None:
            track.append((t, pos))
        if t == OBSERVE:
            index = player.choose(key)
            template = player.templates[index]
            goal = template.goal.copy()
        if template is not None and pos is not None:
            # Re-evaluate which stored inbound trajectory the ball follows.
            if reselect and t in (8, 12) and len(track) >= 3:
                other = []
                for j, s in enumerate(player.templates):
                    rr = [s.difference(v) for _, v in track[-4:]]
                    if all(v is not None for v in rr):
                        dist = np.linalg.norm(rr[-1]) + 1.5 * np.linalg.norm(rr[-1] - rr[0])
                        other.append((dist, j))
                if other:
                    _, index = min(other)
                    template = player.templates[index]
                    residuals = []
            d = template.difference(pos)
            if d is not None:
                residuals.append((pos[1], d.copy()))
                # Early height differences can invert across a table bounce.
                # Limit extrapolation and keep the original pose regularizer.
                current = d.copy()
                if len(residuals) >= 4 and pos[1] < -0.8:
                    prev_y, prev_d = residuals[-4]
                    if prev_y - pos[1] > 0.12:
                        slope = (d - prev_d) / (pos[1] - prev_y)
                        current += np.clip(slope * (template.final_y - pos[1]), -0.09, 0.09)
                fade = np.clip((0.4 - pos[1]) / 0.9, 0.0, 1.0)
                displacement = np.clip(gain * fade * current, -0.13, 0.13)
                delta = template.matrix @ np.array([displacement[0], 0.0, displacement[1], 0.0, 0.0, 0.0])
                goal = template.goal + np.clip(delta, -0.30, 0.30)
        u = command(goal, body)
        encoders.append(arena.joints.tolist())
        _, _, done, cap, _ = arena.step(u, render=False)
        body.issued(u)
        eye.view.issued(u)
        actions.append(u.tolist())
        if done or cap or arena.legal_returns:
            break
    q = np.array(encoders)[:, :5]
    return dict(
        legal=arena.legal_returns,
        contact=arena.contacts,
        steps=len(actions),
        index=index,
        path_rad=float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()),
        delta_rate=float(np.mean(np.linalg.norm(np.diff(actions, axis=0), axis=1)))
        if len(actions) > 1
        else 0.0,
        reason=arena.reason,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    capture(p.parse_args().feeds)
