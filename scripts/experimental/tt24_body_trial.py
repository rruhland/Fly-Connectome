"""Continuous five-axis proprioceptive control on a changing SO-101 plant.

The simulator owns the transmission, inertial changes and servo targets. The
controller sees encoders, own commands and an encoder-space task goal. This
is a generic reaching test, independent of the bounce and rally scorekeepers.
"""
import argparse
import json
import gzip
from pathlib import Path

import mujoco
import numpy as np

from .tt20_curriculum import model_arena
from .tt24_effects import MotorEffects, excitation, transmission


class Body:
    def __init__(self):
        self.arena = model_arena()
        self.arena.reset(seed=44017, render=False)
        model, data = self.arena.model, self.arena.data
        self.dt = self.arena.control_dt
        self.substeps = round(self.dt / model.opt.timestep)
        self.qindex, _, axes, _, _ = self.arena._arm_ids[1]
        self.axes = axes[:5]
        self.qindex = self.qindex[:5]
        self.increments = self.arena._joint_increments[self.axes]
        self.low, self.high = model.actuator_ctrlrange[self.axes].T.copy()
        self.home = data.qpos[self.qindex].copy()
        self.part = [model.body(name).id for name in ('learner_elbow', 'learner_wrist_camera')]
        self.scratch = mujoco.MjData(model)
        self.mass = model.body_mass[self.part].copy()
        self.inertia = model.body_inertia[self.part].copy()
        self.variant(np.eye(5), (1., 1.), 0)

    def reset(self):
        self.variant(np.eye(5), (1., 1.), 0)
        self.arena.reset(seed=44017, render=False)

    def variant(self, transmission, payload, delay):
        self.transmission, self.delay = transmission, delay
        self.pending = [np.zeros(5) for _ in range(delay)]
        model = self.arena.model
        model.body_mass[self.part] = self.mass * payload
        model.body_inertia[self.part] = self.inertia * np.asarray(payload)[:, None]

        mujoco.mj_setConst(model, self.scratch)
        mujoco.mj_forward(model, self.arena.data)

    def observe(self):
        return self.arena.data.qpos[self.qindex].copy()

    def advance(self, action, ticks):
        model, data = self.arena.model, self.arena.data
        for _ in range(ticks):
            self.pending.append(np.clip(self.transmission @ action, -1., 1.))
            delta = self.increments * self.pending.pop(0)
            for _ in range(self.substeps):
                data.ctrl[self.axes] = np.clip(data.ctrl[self.axes] + delta, self.low, self.high)
                mujoco.mj_step(model, data)
        return self.observe()


def model_errors(bank, transitions):
    return [np.mean([np.mean((m.predict(u, d) - y)**2) for u, d, y in transitions])**.5 for m in bank]


def trial(plant, seed, adaptive, *, proactive=False, reverse=False, bounded=False):
    rng = np.random.default_rng(seed)
    variants = [transmission(rng), transmission(rng)]
    plant.reset()
    block = 3

    def move(a):
        plant.advance(a, block)
        return plant.advance(np.zeros(5), block)

    effects = MotorEffects(5, blocks=24)
    prev = np.zeros(5)
    obs = plant.observe()
    for probe in excitation(rng, 2):
        after = move(probe)
        effects.update(probe, after - obs, prev)
        prev, obs = after - obs, after
    rows = []
    explore = []
    probing_evidence = []
    undo = []
    bank = [effects]
    recent = []
    probes_rng = np.random.default_rng(seed + 99991)
    surprising = 0
    for stage, (mapping, payload, delay) in enumerate(zip(
            [np.eye(5), *variants, np.eye(5)],
            [(1., 1.), (1.35, 2.5), (.8, 4.), (1., 1.)], [0, 1, 1, 0])):
        plant.variant(mapping, payload, delay)
        for tick in range(48):
            if tick % 4 == 0:
                goal = np.clip(plant.home + rng.uniform(-.30, .30, 5), plant.low + .025, plant.high - .025)
            # Draw the same probe and goal streams for both treatments.
            probe = rng.uniform(-1., 1., 5)
            desired = .65 * (goal - obs)
            calibrating = bool(explore)
            recalled = False
            reversing = bool(undo)
            if reversing:
                action = undo.pop(0)
            elif calibrating:
                action = explore.pop(0)
            else:
                action = np.clip(effects.solve(desired, prev), -.52, .52) + .015 * probe
            if bounded:
                action *= min(1., .40 / max(1e-6, np.linalg.norm(action)))
            after = move(action)
            if reversing:
                pass
            elif calibrating:
                effects.update(action, after - obs, prev)
                if proactive:
                    probing_evidence.append((action, prev, after - obs))
                    if len(probing_evidence) >= 4:
                        errors = model_errors(bank, probing_evidence)
                        if min(errors) < .035:
                            effects = bank[int(np.argmin(errors))]
                            recalled = True
                            explore.clear()
                if not explore:
                    if not recalled:
                        bank.append(effects)
                    recent.clear()
            else:
                mismatch = np.linalg.norm(effects.predict(action, prev) - (after - obs)) / np.sqrt(5)
                recent.append((action, prev, after - obs))
                recent = recent[-3:]
                threshold = (.025 + .25*np.sqrt(np.mean((after - obs)**2))) if proactive else .095
                surprising = surprising + 1 if mismatch > threshold else 0
                if adaptive and surprising >= (1 if reverse else 2):
                    if reverse:
                        undo = [-action.copy()]
                    errors = model_errors(bank, recent)
                    if min(errors) < (.030 if proactive else .052):
                        effects = bank[int(np.argmin(errors))]
                        recalled = True
                    else:
                        effects = MotorEffects(5, blocks=24)
                        explore = excitation(probes_rng)
                        probing_evidence = []
                    recent.clear()
                    surprising = 0
            prev, obs = after - obs, after
            rows.append(dict(stage=stage, block=tick, error=float(np.linalg.norm(goal - after) / np.sqrt(5)),
                             probe=calibrating and not reversing, reverse=reversing, recalled=recalled, clip=float(np.mean(np.abs(action) >= .52))))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', type=int, default=12)
    parser.add_argument('--proactive', action='store_true')
    parser.add_argument('--reverse', action='store_true')
    parser.add_argument('--bounded', action='store_true')
    parser.add_argument('--first', type=int, default=431)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    plant = Body()
    try:
        rows = {str(s): {mode: trial(plant, s, mode == 'adaptive', proactive=args.proactive, reverse=args.reverse, bounded=args.bounded) for mode in ('adaptive', 'frozen')}
                for s in range(args.first, args.first + args.seeds)}
        record = dict(proactive=args.proactive, reverse=args.reverse, bounded=args.bounded, dt=plant.dt, block_frames=6, nominal_probe_blocks=20, rows=rows)
        with (gzip.open(args.out, 'wt') if args.out.suffix == '.gz' else args.out.open('w')) as file:
            json.dump(record, file, separators=(',', ':'))
    finally:
        plant.arena.close()


if __name__ == '__main__':
    main()
