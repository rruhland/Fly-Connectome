"""Teach a second receive after the physical far arm replies on a full-size table."""

import argparse
import json
from pathlib import Path

import numpy as np

from .tt4_arena import TABLE_HEIGHT
from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt24_rally_policy import ViewHistory
from .tt24_rally_trial import PublicBody, STEP_SIZE
from .tt25_pose_demonstrations import incoming, poses, suggest
from .tt25_recorded_flight import OUTPUT
from .tt25_reference_receiver import ReferenceReceiver


def feed(event):
    out = {}
    for p, name in [("pos", "ball"), ("vel", "velocity"), ("w_vel", "spin")]:
        for i, a in enumerate("xyz"):
            out[p + "_" + a] = event[name][i]
    out["pos_z"] -= TABLE_HEIGHT
    return out


def execute(arena, shot, row, switch, goal, collect=False, through=None, samples=None):
    arena.launch(shot)
    body = PublicBody(arena)
    eye = ViewHistory()
    key = None
    actions = []
    for tick in range(arena.max_steps):
        far = (
            row["rival_actions"][tick]
            if tick < len(row["rival_actions"])
            else np.zeros(len(arena._arm_ids[-1][2]))
        )
        arena._aim = arena._command_to = lambda *args, far=far: np.asarray(far)
        if collect or samples is not None:
            obs = eye.observe(arena.render_cameras(), arena.joints, body.target)
        if collect and tick == switch:
            key = obs.copy()
        if tick < switch:
            cmd = np.array(row["actions"][tick])
        else:
            cmd = np.zeros(arena.action_size)
            target = goal
            if through is not None:
                impact_tick, velocity, lead = through
                phase = np.clip((tick - impact_tick + lead) / 30.0, -0.14, 0.18)
                target = goal + phase * velocity
            cmd[:5] = np.clip((target - body.target[:5]) / STEP_SIZE[:5], -1, 1)
        if samples is not None:
            samples.append((obs.copy(), cmd.copy()))
        _, _, done, cap, _ = arena.step(cmd, render=False)
        body.issued(cmd)
        eye.issued(cmd)
        actions.append(cmd.tolist())
        if done or cap or arena.legal_returns >= 2:
            break
    return dict(
        near=arena.legal_returns,
        contacts=arena.contacts,
        switched=None,
        frames=len(actions),
        events=digest_json(arena.events),
    ), key


def run(path):
    shots = {r["id"]: r for r in json.loads(path.read_bytes())}
    training = _read(OUTPUT / "recorded-live-development.json.gz")["cases"]
    coeff = json.loads((OUTPUT / "recorded-flight.json").read_text())["held_scores"]["match_fit"][
        "coefficients"
    ]
    arena = ReferenceReceiver()
    arena.max_steps = 450
    keys, goals, practice, cases = [], [], [], []
    try:
        bank = poses(arena)
        for row in training:
            if row["replies"] == 0:
                continue
            e = next(e for e in row["events"] if e.get("collider") == "rival_racket")
            # This timestamp is a training scaffold. Frozen deployment below
            # switches based on visual plus own-body observations only.
            tick = e["step"]
            start = np.asarray(row["actions"][:tick])[:, :5].sum(axis=0) * STEP_SIZE[:5]
            from .tt19_so101 import SO101_HOME

            options = suggest(
                bank,
                incoming(feed(e), coeff),
                start_angles=np.array(SO101_HOME[:5]) + start,
                observation_delay=0.0,
            )
            trials = []
            for goal, loss in options:
                stat, _ = execute(arena, shots[row["id"]], row, tick, goal)
                trials.append(dict(loss=loss, goal=goal.tolist(), **stat))
                if stat["near"] >= 2:
                    again, key = execute(arena, shots[row["id"]], row, tick, goal, collect=True)
                    if again != stat:
                        raise ValueError("Second-ball demonstration did not replay")
                    keys.append(key)
                    goals.append(goal)
                    break
            practice.append(dict(id=row["id"], trigger=tick, trials=trials))
            print(
                "second", row["id"], len(options), max((s["near"] for s in trials), default=0), flush=True
            )
        _write(
            OUTPUT / "recorded-second-ball.json.gz",
            dict(practice=practice, demonstrated=len(keys), cases=cases),
        )
    finally:
        arena.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deepmind", type=Path)
    run(p.parse_args().deepmind)
