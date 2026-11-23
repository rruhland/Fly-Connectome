"""Learn direct motor rates for a second receive from successful physical experience."""

import argparse
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt19_view_full_game import digest_json
from .tt22_opponent import install_rally_opponent
from .tt22_opponent_trial import FarTape
from .tt24_rally_policy import ViewHistory
from .tt24_rally_trial import COACHED, PublicBody, STEP_SIZE
from .tt25_recorded_flight import OUTPUT
from .tt25_reference_receiver import ReferenceReceiver
from .tt25_second_ball import execute


def collect(shots):
    prep = {r["id"]: r for r in _read(OUTPUT / "recorded-second-ball.json.gz")["practice"]}
    points = {r["id"]: r for r in _read(OUTPUT / "recorded-live-development.json.gz")["cases"]}
    practice = _read(OUTPUT / "recorded-through-contact.json.gz")["cases"]
    arena = ReferenceReceiver()
    arena.max_steps = 450
    starts, negatives, states, rates, donors = [], [], [], [], []
    try:
        for case in practice:
            best = next((a for a in case["attempts"] if a["near"] >= 2), None)
            if best is None:
                continue
            sid = case["id"]
            switch = prep[sid]["trigger"]
            data = []
            row, _ = execute(
                arena,
                shots[sid],
                points[sid],
                switch,
                np.array(best["goal"]),
                samples=data,
                through=(best["impact"], np.array(best["rate"]), best["lead"]),
            )
            if row["events"] != best["events"]:
                raise ValueError("Physical demonstration did not reproduce")
            starts.extend(s for s, _ in data[switch : switch + 2])
            negatives.extend(s for s, _ in data[: switch - 3])
            for s, q in data[switch:]:
                states.append(s)
                rates.append(q)
            donors.append(sid)
    finally:
        arena.close()
    return [np.stack(a) for a in (starts, negatives, states, rates)], donors


def play(arena, shot, memory, sensor_scale, first, tape):
    if memory is not None:
        triggers, negatives, states, rates = memory
    keys, goals, scales = first
    arena.launch(shot)
    tape.steps.clear()
    body, eye = PublicBody(arena), ViewHistory()
    initial = None
    activated = None
    actions = []
    for tick in range(arena.max_steps):
        obs = eye.observe(arena.render_cameras(), arena.joints, body.target)
        if tick == 3:
            initial = goals[np.argmin(np.linalg.norm((keys - obs[:14]) / scales, axis=1))]
        if memory is not None and activated is None and initial is not None:
            pos = np.min(np.linalg.norm((triggers - obs) / sensor_scale, axis=1))
            neg = np.min(np.linalg.norm((negatives - obs) / sensor_scale, axis=1))
            if pos < neg and pos < 3.0:
                activated = tick
        cmd = np.zeros(arena.action_size)
        if activated is not None:
            distance = np.linalg.norm((states - obs) / sensor_scale, axis=1)
            cmd = rates[np.argmin(distance)]
        elif initial is not None:
            cmd[:5] = np.clip((initial - body.target[:5]) / STEP_SIZE[:5], -1, 1)
        _, _, done, cap, _ = arena.step(cmd, render=False)
        body.issued(cmd)
        eye.issued(cmd)
        actions.append(cmd.tolist())
        if done or cap:
            break
    return dict(
        id=shot["id"],
        near=arena.legal_returns,
        far=max(0, arena.rival_returns - 1),
        activated=activated,
        actions=actions,
        rival_actions=tape.tape(len(actions)),
        events=arena.events.copy(),
        outcome=arena.point_result(),
    )


def run(path):
    shots = {r["id"]: r for r in json.loads(path.read_bytes())}
    memory, donors = collect(shots)
    with np.load(OUTPUT / "recorded-pose-imitation.npz", allow_pickle=False) as f:
        first = [f[k] for k in ("image_keys", "joint_goals", "scale")]
    scale = np.r_[first[2], np.full(12, 0.5), np.full(6, 0.12), np.full(6, 0.6)]
    np.savez_compressed(
        OUTPUT / "recorded-second-pupil.npz",
        triggers=memory[0],
        negatives=memory[1],
        states=memory[2],
        rates=memory[3],
        scale=scale,
    )
    arena = ReferenceReceiver(stationary_opponent=False)
    check = ReferenceReceiver(stationary_opponent=False)
    arena.max_steps = check.max_steps = 450
    profile = _read(COACHED)["metadata"]["opening"]["far_profile"]
    install_rally_opponent(arena, target_profile=profile)
    tape = FarTape(arena)
    held = _read(OUTPUT / "recorded-live-opponent.json.gz")["conditional_roster"]
    cases = []
    try:
        for sid in donors + held:
            row = play(arena, shots[sid], memory, scale, first, tape)
            check.launch(shots[sid])
            for near, far in zip(row["actions"], row["rival_actions"], strict=True):
                check._aim = check._command_to = lambda *args, far=far: np.asarray(far)
                _, _, done, cap, _ = check.step(near, render=False)
            if not (done or cap) or digest_json(check.events) != digest_json(row["events"]):
                raise ValueError("Physical referee changed on replay")
            cases.append(row)
            print(sid, sid in donors, row["near"], row["far"], row["activated"], flush=True)
        _write(OUTPUT / "recorded-second-pupil.json.gz", dict(donors=donors, held=held, cases=cases))
    finally:
        arena.close()
        check.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("deepmind", type=Path)
    run(p.parse_args().deepmind)
