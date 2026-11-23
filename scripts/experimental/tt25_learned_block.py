"""Fit a fixed receiving posture using only scored physical shots, as a passive control."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.stats import qmc

from .tt4_transfer_trial import _write
from .tt19_so101 import RATES, SO101_HOME
from .tt19_view_full_game import digest_json
from .tt25_recorded_flight import OUTPUT
from .tt25_reference_receiver import ReferenceReceiver


def prepare(arena, goal):
    arena.reset(seed=912, server=-1, render=False)
    # Move into the ready pose before any incoming observation. The archived
    # snapshot amortizes identical pre-positioning over paired independent shots.
    arena.ball[:] = [0., 4., 12.]
    arena.velocity[:] = 0.
    arena.data.qvel[arena._ball_dof+3:arena._ball_dof+6] = 0.
    dt = arena.model.opt.timestep
    axes = arena._arm_ids[1][2]
    for _ in range(round(1.2/dt)):
        arena.data.ctrl[axes] += np.clip(goal-arena.data.ctrl[axes], -np.asarray(RATES)*dt, np.asarray(RATES)*dt)
        mujoco.mj_step(arena.model, arena.data)
    return arena.data.qpos.copy(), arena.data.qvel.copy(), arena.data.ctrl.copy()


def point(arena, shot, ready, audit=False):
    arena.launch(shot)
    position, velocity, control = ready
    bq, bv = arena._ball_joint, arena._ball_dof
    for arr, const, start, count in [(arena.data.qpos, position, bq, 7), (arena.data.qvel, velocity, bv, 6)]:
        arr[:start] = const[:start]
        arr[start+count:] = const[start+count:]
    arena.data.ctrl[:] = control
    mujoco.mj_forward(arena.model, arena.data)
    zero = np.zeros(arena.action_size)
    for _ in range(arena.max_steps):
        _, _, done, cap, _ = arena.step(zero, render=False)
        if arena.legal_returns or done or cap:
            break
    row = dict(id=shot["id"], contact=arena.contacts, legal=arena.legal_returns, incoming=arena.rival_returns)
    if audit:
        row.update(steps=arena.steps, reason=arena.reason, events=digest_json(arena.events))
    return row


def run(path):
    import hashlib
    raw = path.read_bytes()
    source = json.loads((OUTPUT / "recorded-secondary-reference.json").read_text())
    if hashlib.sha256(raw).hexdigest() != source["sha256"]:
        raise ValueError("Unexpected recorded shots")
    shots = {r["id"]: r for r in json.loads(raw)}
    ordered = json.loads((OUTPUT/"recorded-reach.json").read_text())["cases"]
    train = [r["id"] for r in ordered[32:128] if r["near_blade"]][:12]
    test = [r["id"] for r in ordered[128:160]]
    if len(train) != 12 or len(test) != 32 or set(train) & set(test):
        raise ValueError("Invalid practice/evaluation separation")
    arena = ReferenceReceiver()
    try:
        home = np.array(SO101_HOME)
        axes = arena._arm_ids[1][2]
        # Symmetric, task-independent joint perturbations, fixed before scoring.
        draws = 2*qmc.Sobol(5, scramble=True, seed=15657).random_base2(5)-1
        poses = [home]
        for offsets in draws:
            goal = home.copy()
            goal[:5] += .9*offsets
            poses.append(np.clip(goal, arena._joint_low[axes], arena._joint_high[axes]))
        ready = [prepare(arena, q) for q in poses]
        practice = []
        for i, pose in enumerate(ready):
            rows = [point(arena, shots[s], pose) for s in train]
            practice.append(rows)
            print("pose", i, sum(r["legal"] for r in rows), sum(r["contact"] for r in rows), flush=True)
        def rank(i):
            return (sum(r["legal"] for r in practice[i]), sum(r["contact"] > 0 for r in practice[i]), -i)
        best = max(range(len(poses)), key=rank)
        conditions = {}
        for mode, i in [("learned_ready",best), ("default_ready",0)]:
            results = [point(arena, shots[s], ready[i], audit=True) for s in test]
            again = [point(arena, shots[s], ready[i], audit=True) for s in test]
            if results != again:
                raise ValueError("Passive control changed during independent replay")
            conditions[mode] = results
        _write(OUTPUT/"recorded-learned-block.json.gz", dict(source=source, train=train, held=test,
               proposals=np.asarray(poses).tolist(), selected=best, selected_goal=poses[best].tolist(),
               practice=practice, conditions=conditions, verified=2*len(test)))
        print("held",best,{k:sum(r["legal"] for r in v) for k,v in conditions.items()},flush=True)
    finally:
        arena.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("deepmind", type=Path)
    run(parser.parse_args().deepmind)
