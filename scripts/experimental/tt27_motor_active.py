"""Identify hidden command routing with a short, balanced encoder probe."""

import argparse
from dataclasses import asdict
from itertools import permutations
from pathlib import Path

import mujoco
import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_so101 import AXES
from .tt24_rally_trial import PublicBody
from .tt25_pose_demonstrations import OBSERVE
from .tt25_reference_receiver import ReferenceReceiver
from .tt26_bounce_effects import CHECKPOINT, JointEffects
from .tt27_motor_interface import AXIS_COUNT, Candidates, IDENTITY, Interface, course, describe_fit, play, select
from .tt27_world_camera import CameraMounts, FEEDS, POSES


OUT = Path("artifacts/tt27/motor-active.json.gz")
SEEDS = (400132193, 40179057)
SETTINGS = 6
TICKS = 10
PULSE = .38
LOW = (.16, -.21, .27, -.33, .39)
MID = (-.33, .39, .21, -.16, -.27)
HIGH = (.27, .16, -.39, -.21, .33)
HOLD = np.zeros((4, AXIS_COUNT))
PACKET = np.vstack((LOW, -np.asarray(LOW), MID, -np.asarray(MID), HIGH, -np.asarray(HIGH), HOLD))
MARGIN = 9.
PRIOR_COST = 2 * np.log(119.)


def encoder(arena, dofs):
    return np.r_[arena.joints[:AXIS_COUNT], .02 * arena.data.qvel[dofs]]


def collect(arena, dofs, requests, actual, seed):
    """Prepare the articulated arm while the stationary receiver has no feed."""
    arena.reset(seed=seed, server=-1, render=False)
    arena._serving = False
    arena.ball[:] = (0., 1.45, 2.75)
    arena.velocity[:] = 0.
    mujoco.mj_forward(arena.model, arena.data)
    states = [encoder(arena, dofs)]
    previous = np.zeros(AXIS_COUNT)
    delivered = []
    for request in requests:
        applied = actual.apply(previous if actual.delay else request)
        _, _, done, cap, _ = arena.step(np.r_[applied, 0.], render=False)
        if done or cap:
            raise ValueError("A preparation interval ended before the diagnostic finished")
        previous = request
        delivered.append(applied)
        states.append(encoder(arena, dofs))
    return np.asarray(states), np.asarray(delivered)


class PulseModel:
    """A local command impulse response acquired on the native interface."""

    def __init__(self, arena, dofs):
        blank = np.zeros((TICKS, AXIS_COUNT))
        self.rest, _ = collect(arena, dofs, blank, IDENTITY, 1504161)
        impulse_traces = []
        for axis in range(AXIS_COUNT):
            stream = blank.copy()
            stream[0, axis] = PULSE
            plus, _ = collect(arena, dofs, stream, IDENTITY, 1504161)
            minus, _ = collect(arena, dofs, -stream, IDENTITY, 1504161)
            impulse_traces.append((plus[1:] - minus[1:]) / (2 * PULSE))
        self.kernel = np.stack(impulse_traces, axis=1)
        rng = np.random.default_rng(20409323)
        residuals = []
        for _ in range(6):
            pulse = PACKET[:, rng.permutation(AXIS_COUNT)] * rng.choice((-1, 1), size=AXIS_COUNT)
            observed, _ = collect(arena, dofs, pulse, IDENTITY, 1504161)
            residuals.append(observed[1:] - self.predict(pulse[:, None, :])[:, 0])
        rms = np.sqrt(np.mean(np.asarray(residuals)**2, axis=(0, 1)))
        self.noise = np.maximum(rms * 2., .0016)
        self.receipt = dict(rest_ticks=TICKS, symmetric_pulse_trials=2 * AXIS_COUNT,
                            source_validation_trials=len(residuals), total_ticks=TICKS * (1 + 2 * AXIS_COUNT + len(residuals)),
                            single_axis_source_peak=PULSE, residual_rms=rms.tolist(), noise=self.noise.tolist(),
                            native_kernel=self.kernel.tolist())

    def predict(self, effective):
        states = []
        for tick in range(len(effective)):
            motion = sum(effective[tick - past] @ self.kernel[past] for past in range(tick + 1))
            states.append(self.rest[tick + 1] + motion)
        return np.asarray(states)

    def loss(self, requests, states, candidates):
        requests = np.vstack((np.zeros(AXIS_COUNT), requests))
        ticks = np.arange(1, len(requests))[:, None, None]
        effective = candidates.signs * requests[ticks - candidates.delays[None, :, None],
                                                candidates.permutations[None, :, :]]
        residual = (self.predict(effective) - states[1:, None]) / self.noise
        return np.cumsum(np.minimum(residual**2, 64.).sum(axis=2), axis=0)


def stable_choice(loss, candidates, initial, prior_order=None):
    prior = 0. if prior_order is None else PRIOR_COST * np.all(candidates.permutations == prior_order, axis=1)
    accepted = initial
    previous = None
    first_switch = None
    history = []
    for index, evidence in enumerate(loss):
        scores = evidence - prior
        a, b = np.argpartition(scores, 1)[:2]
        margin = scores[b] - scores[a]
        candidate = candidates.maps[a]
        if candidate == previous and margin >= MARGIN:
            if first_switch is None:
                first_switch = index + 1
            accepted = candidate
        previous = candidate if margin >= MARGIN else None
        history.append(dict(best=asdict(candidate), accepted=asdict(accepted), margin=float(margin)))
    return accepted, dict(confirm_tick=first_switch, history=history)


def calibrate_passive(arena, dofs, previous, candidates):
    traces = []
    for old in previous[:16]:
        tape = np.asarray(old["results"]["bounce"]["actions"])[:, :AXIS_COUNT]
        trace = play(arena, old["shot"], None, IDENTITY, IDENTITY, dofs,
                     budget=min(TICKS - OBSERVE, len(tape) - OBSERVE), tape=tape)
        traces.append(trace)
    return candidates.learn_clock_correction(traces)


def install_course(rng):
    # Half of the transitions change only signs and latency relative to their
    # preceding permutation. The others replace that permutation as well.
    population = np.asarray(list(permutations(range(AXIS_COUNT))))
    orders = population[rng.choice(len(population), size=3, replace=False)]
    delays = (0, 1, 0, 1, 0, 1)
    settings = []
    for n, (order, delay) in enumerate(zip(np.repeat(orders, 2, axis=0), delays, strict=True)):
        polarity = tuple(rng.choice((-1, 1), size=AXIS_COUNT).tolist())
        routing = Interface(tuple(order.tolist()), polarity, delay)
        x = (.018 + .006 * (n % 4)) * (1 if n % 2 else -1)
        speed = (.97, 1.02, .98, 1.03, .96, 1.04)[n]
        settings.append((routing, x, speed))
    return settings


def totals(rows):
    modes = rows[0]["returns"]
    values = {name: dict(legal=sum(r["returns"][name]["legal"] for r in rows),
                         contacts=sum(r["returns"][name]["contact"] for r in rows)) for name in modes}
    for name in modes:
        paired = np.asarray([r["returns"][name]["legal"] - r["returns"]["passive"]["legal"] for r in rows])
        values[name]["wins_losses_vs_passive"] = [int(np.sum(paired > 0)), int(np.sum(paired < 0))]
    return values


def run(path, per_setting):
    history = [r for r in _read(FEEDS)["cases"] if r["angle"] < 0]
    donors = {r["shot"]["id"]: r["shot"] for r in history}
    with np.load(POSES, allow_pickle=False) as data:
        keys, goals, scale = (data[k] for k in ("keys", "goals", "scale"))
    arena = ReferenceReceiver()
    mounts = CameraMounts(arena)
    dofs = [arena.model.joint("learner_" + axis).dofadr[0] for axis in AXES[:AXIS_COUNT]]
    native = PublicBody(arena)
    candidates = Candidates(JointEffects.load(), native.lower[:AXIS_COUNT], native.upper[:AXIS_COUNT], arena.control_dt)
    receipt = dict(schema="tt27-motor-active-1", source_fit_sha256=_hash(CHECKPOINT),
                   source_game_sha256=_hash(FEEDS), receiving_pose_sha256=_hash(POSES),
                   seeds=SEEDS, prep_ticks=TICKS, request_peak=float(np.max(np.abs(PACKET))),
                   active_request_ticks=int(np.count_nonzero(np.linalg.norm(PACKET, axis=1))),
                   hypothesis_count=len(candidates.maps), acceptance_margin=MARGIN,
                   receiver_clock_s=arena.control_dt, settings_per_seed=SETTINGS,
                   recentering="each assessed point launches from the engine's ordinary home state",
                   same_budget_passive="ten ordinary ticks including the three before the receiver sends a nonzero stroke",
                   streams=[])
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        impulse = PulseModel(arena, dofs)
        receipt["source_impulse_fit"] = impulse.receipt
        receipt["passive_source"] = calibrate_passive(arena, dofs, history, candidates)
        for seed in SEEDS:
            stream = dict(seed=seed, settings=[])
            receipt["streams"].append(stream)
            rng = np.random.default_rng(seed)
            carried = IDENTITY
            for index, (actual, shift, speed) in enumerate(install_course(rng)):
                adaptation, *fresh = course(rng, donors, per_setting + 1, shift, speed)
                probe_seed = int(rng.integers(2**31))
                states, delivered = collect(arena, dofs, PACKET, actual, probe_seed)
                errors = impulse.loss(PACKET, states, candidates)
                probed, active_trace = stable_choice(errors, candidates, carried, carried.permutation)
                reset, reset_trace = stable_choice(errors, candidates, IDENTITY)
                index0 = select(arena, mounts, adaptation, keys, scale)
                ordinary = play(arena, adaptation, goals[index0], actual, carried, dofs, budget=TICKS - OBSERVE)
                passive_loss = candidates.loss(ordinary)
                passive, passive_trace = stable_choice(passive_loss, candidates, carried, carried.permutation)
                encoders = np.asarray(ordinary["encoders"])
                passive_states = np.vstack((np.r_[encoders[0, 0], .02 * encoders[0, 1]],
                                            np.c_[encoders[:, 2], .02 * encoders[:, 3]]))
                impulse_passive_loss = impulse.loss(ordinary["requests"], passive_states, candidates)
                passive_pulse, passive_pulse_trace = stable_choice(impulse_passive_loss, candidates,
                                                                  carried, carried.permutation)
                modes = dict(active=probed, active_reset=reset, passive=passive, passive_same_model=passive_pulse,
                             frozen=carried, factory=IDENTITY, expert=actual)
                rows = []
                for shot in fresh:
                    pick = select(arena, mounts, shot, keys, scale)
                    results, cached = {}, {}
                    for name, candidate in modes.items():
                        if candidate not in cached:
                            scored = play(arena, shot, goals[pick], actual, candidate, dofs)
                            check = play(arena, shot, goals[pick], actual, candidate, dofs, tape=scored["requests"])
                            if check != scored:
                                raise ValueError("A physical command replay changed the scored outcome")
                            cached[candidate] = scored
                        results[name] = cached[candidate]
                    rows.append(dict(shot=shot, goal=pick, returns=results, verified=len(cached)))
                displacement = np.abs(states[:, :AXIS_COUNT] - states[0, :AXIS_COUNT])
                stream["settings"].append(dict(
                    number=index, actual=asdict(actual), shift_m=shift, forward_speed_factor=speed,
                    prior_accepted=asdict(carried), estimates={name: describe_fit(value, actual) for name, value in modes.items()},
                    probe=dict(seed=probe_seed, issued=PACKET.tolist(), executed=delivered.tolist(), states=states.tolist(),
                               peak_angle_excursion_rad=float(displacement.max()),
                               final_angle_error_rad=float(displacement[-1].max())),
                    active_trace=active_trace, reset_trace=reset_trace, passive_trace=passive_trace,
                    passive_same_model_trace=passive_pulse_trace, ordinary=ordinary, rows=rows, summary=totals(rows),
                ))
                carried = probed
                all_rows = [row for s in receipt["streams"] for case in s["settings"] for row in case["rows"]]
                receipt["summary"] = totals(all_rows)
                _write(path, receipt)
                print(seed, index, "fits", {name: value == actual for name, value in modes.items()},
                      "confirm", [active_trace["confirm_tick"], reset_trace["confirm_tick"], passive_trace["confirm_tick"]],
                      "legal", {name: v["legal"] for name, v in totals(rows).items()}, flush=True)
    finally:
        arena.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--per-setting", type=int, default=8)
    args = parser.parse_args()
    if args.per_setting <= 0:
        parser.error("--per-setting must be positive")
    run(args.output, args.per_setting)
