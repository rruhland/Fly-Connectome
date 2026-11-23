"""Update the next motor command from a stereo observation of one's own miss."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt24_rally_trial import PublicBody
from .tt25_pose_demonstrations import episode
from .tt26_grip_course import ReferenceReceiver, load_match
from .tt26_optical_transfer import decoder
from .tt26_receiving_speed import draw
from .tt27_flight_feedback import OUT, SOURCE, Eyes, Player, command, load_bank
from .tt27_mount_position import paddle_origin
from .tt27_opening_course import Physics


class OpticalError:
    def __init__(self, arena):
        self.rig = decoder()
        self.offset = paddle_origin(arena) - self.center(arena.joints)
        self.aim = None

    def center(self, joints):
        p, r = self.rig.links.read(joints[None, :5])
        return p[0] + r[0] @ self.rig.offset[3:]

    @staticmethod
    def crossing(relative):
        for a, b in zip(relative, relative[1:]):
            if a[0] + 3 < b[0] or a[2] < 0 or b[2] > 0 or a[2] - b[2] < 0.01:
                continue
            v = a[1:] + (b[1:] - a[1:]) * a[2] / (a[2] - b[2])
            return v[[0, 2]]
        if len(relative) >= 2:
            a, b = relative[-2:]
            if 0 < b[2] < 0.20 and a[2] > b[2] + 0.025 and b[0] - a[0] <= 2:
                return (b[1:] + (b[1:] - a[1:]) * b[2] / (a[2] - b[2]))[[0, 2]]
        return None


def play(arena, shot, player, eye, remember=None):
    arena.launch(shot)
    vision = Eyes(arena)
    body = PublicBody(arena)
    goal = None
    rel = []
    time = []
    q = []
    idx = None
    key = None
    correction = np.zeros(2)
    for tick in range(arena.max_steps):
        k, p = vision.observe(arena, body)
        if tick == 3:
            key = k
            idx = player.choose(k)
            if remember is not None:
                correction = remember.query(k)
            template = player.templates[idx]
            goal = template.goal + np.clip(
                template.matrix @ np.r_[correction[0], 0.0, correction[1], 0.0, 0.0, 0.0], -0.7, 0.7
            )
        if p is not None and p[1] < -0.75 and (not time or p[1] < time[-1] - 0.018):
            time.append(p[1])
            rel.append(np.r_[tick, p - eye.center(arena.joints) - eye.offset])
        u = command(goal, body)
        q.append(arena.joints.tolist())
        _, _, done, cap, _ = arena.step(u, render=False)
        body.issued(u)
        vision.view.issued(u)
        if done or cap or arena.legal_returns:
            break
    miss = eye.crossing(rel)
    row = dict(
        legal=arena.legal_returns,
        contact=arena.contacts,
        choice=idx,
        key=key.tolist() if key is not None else None,
        correction=correction.tolist(),
        observed=miss.tolist() if miss is not None else None,
        path=float(np.linalg.norm(np.diff(np.array(q)[:, :5], axis=0), axis=1).sum()),
        frames=len(q),
    )
    if remember is not None and key is not None:
        remember.update(key, correction, miss, eye.aim[idx], row["legal"], row["contact"])
    return row


class Feedback:
    def __init__(self, scale):
        self.scale = scale
        self.history = []

    def query(self, key):
        for old, action in reversed(self.history):
            if np.linalg.norm((key - old) / self.scale) < 1.15:
                return action.copy()
        return np.zeros(2)

    def update(self, key, current, observed, desired, legal, contact):
        new = current.copy()
        if not legal and observed is not None:
            delta = observed - desired
            gain = 0.48 if contact else 0.77
            new += gain * np.clip(delta, -0.105, 0.105)
        self.history.append((key, np.clip(new, -0.165, 0.165)))


def run(path):
    lookup, _ = load_match(path)
    player = Player()
    arena = ReferenceReceiver()
    arena.reset(seed=501, render=False)
    error = OpticalError(arena)
    phy = Physics(arena)
    refs = [r["id"] for r in load_bank()["rows"]]
    try:
        source = [play(arena, lookup[s], player, error) for s in refs]
        observed = [s["observed"] for s in source if s["observed"] is not None]
        fallback = np.median(observed, axis=0)
        error.aim = np.array([s["observed"] if s["observed"] is not None else fallback for s in source])
        raw = [
            r["id"]
            for r in _read(SOURCE / "fresh-recordings.json.gz")["records"][0]["rows"]
            if r["group"] == "nearby"
        ]
        rng = np.random.default_rng(661014)
        blocks = []
        # Two distinct families of incoming balls: independent recorded donors and
        # a wider, developmentally easier, continuous perturbation of a known donor.
        for cohort in ("recorded", "nearby"):
            carry = Feedback(player.scale)
            selected = rng.choice(raw if cohort == "recorded" else refs, 8, replace=False)
            for index, sid in enumerate(selected):
                s = [1.0, 1.5, 0.72, 2.1][index % 4]
                phy.set(s, 1.0)
                local = Feedback(player.scale)
                rows = []
                for t in range(7):
                    shot = draw(lookup, [int(sid)], rng, t)
                    if cohort == "nearby":
                        shot["vel_x"] += 0.16 if index % 2 else -0.16
                    # Frozen branch only needs its camera signature and motor posture.
                    frozen, _ = episode(
                        arena, shot, choose_goal=lambda k: player.templates[player.choose(k)].goal
                    )
                    old = play(arena, shot, player, error, carry)
                    if (
                        not local.history
                        and np.linalg.norm(old["correction"]) == 0
                        and old["key"] is not None
                    ):
                        fresh = old.copy()
                        local.update(
                            np.array(old["key"]),
                            np.array(old["correction"]),
                            None if old["observed"] is None else np.array(old["observed"]),
                            error.aim[old["choice"]],
                            old["legal"],
                            old["contact"],
                        )
                    else:
                        fresh = play(arena, shot, player, error, local)
                    rows.append(dict(t=t, frozen=frozen, carried=old, reset=fresh))
                b = dict(cohort=cohort, donor=int(sid), damping=s, rows=rows)
                blocks.append(b)
                print(
                    cohort,
                    sid,
                    {k: sum(r[k]["legal"] for r in rows) for k in ("frozen", "carried", "reset")},
                    "measured",
                    sum(r["carried"]["observed"] is not None for r in rows),
                    flush=True,
                )
                _write(OUT / "miss-learning.json.gz", dict(seed=661014, source=source, blocks=blocks))
    finally:
        arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("feeds", type=Path)
    run(p.parse_args().feeds)
