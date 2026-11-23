"""Restore receiving by matching isolated physical motor effects after a remap."""

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_so101 import AXES
from .tt24_rally_trial import PublicBody
from .tt25_pose_demonstrations import OBSERVE
from .tt25_reference_receiver import ReferenceReceiver
from .tt26_bounce_effects import CHECKPOINT, JointEffects
from .tt27_motor_active import calibrate_passive, collect, install_course, stable_choice, totals
from .tt27_motor_interface import AXIS_COUNT, Candidates, IDENTITY, Interface, course, describe_fit, play, select
from .tt27_world_camera import CameraMounts, FEEDS, POSES


OUT = Path("artifacts/tt27/motor-active-serial.json.gz")
SEEDS = (53017817, 8913323)
TICKS_PER_PORT = 6
PULSE = .32
NOMINAL_SIGMA = .0016


def packet(port):
    values = np.zeros((TICKS_PER_PORT, AXIS_COUNT))
    values[0, port] = PULSE
    values[1, port] = -PULSE
    return values


class NativeTemplates:
    """Measure nonlinear isolated motor responses, without reading actuator mappings."""

    def __init__(self, arena, dofs):
        self.responses = np.empty((AXIS_COUNT, 2, 2, TICKS_PER_PORT, 2 * AXIS_COUNT))
        for axis in range(AXIS_COUNT):
            for flip in range(2):
                for delay in range(2):
                    body = Interface(IDENTITY.permutation, IDENTITY.signs, delay)
                    requests = packet(axis) * (-1 if flip else 1)
                    trace, _ = collect(arena, dofs, requests, body, 41986931)
                    self.responses[axis, flip, delay] = trace[1:] - trace[0]
        self.receipt = dict(trials=AXIS_COUNT * 4, ticks=AXIS_COUNT * 4 * TICKS_PER_PORT,
                            responses=self.responses.tolist(),
                            encoders="joint angles in radians and joint velocities scaled by 0.02 seconds")

    def loss(self, traces, candidates):
        joint_of_port = np.argsort(candidates.permutations, axis=1)
        all_rows = []
        accumulated = np.zeros(len(candidates.maps))
        for port, observed in enumerate(traces):
            native = joint_of_port[:, port]
            flipped = candidates.signs[np.arange(len(native)), native] < 0
            response = self.responses[native, flipped.astype(int), candidates.delays]
            relative = observed[1:] - observed[0]
            loss = np.minimum(((response - relative) / NOMINAL_SIGMA)**2, 64.)
            partial = np.cumsum(loss.sum(axis=2), axis=1).T + accumulated
            all_rows.extend(partial)
            accumulated = partial[-1]
        return np.asarray(all_rows)


def passive_block(arena, mounts, shots, pose_memory, actual, carry, dofs, candidates):
    keys, goals, scale = pose_memory
    observations, all_rows = [], []
    accumulated = np.zeros(len(candidates.maps))
    for shot in shots:
        position = select(arena, mounts, shot, keys, scale)
        trace = play(arena, shot, goals[position], actual, carry, dofs, budget=TICKS_PER_PORT - OBSERVE)
        observations.append(trace)
        partial = candidates.loss(trace) + accumulated
        all_rows.extend(partial)
        accumulated = partial[-1]
    inferred, context = stable_choice(all_rows, candidates, carry, carry.permutation)
    return inferred, dict(inference=context, observations=observations)


def run(path, per_setting):
    history = [r for r in _read(FEEDS)["cases"] if r["angle"] < 0]
    donors = {r["shot"]["id"]: r["shot"] for r in history}
    with np.load(POSES, allow_pickle=False) as data:
        pose_memory = tuple(data[k] for k in ("keys", "goals", "scale"))
    keys, goals, scale = pose_memory
    arena = ReferenceReceiver()
    mounts = CameraMounts(arena)
    dofs = [arena.model.joint("learner_" + axis).dofadr[0] for axis in AXES[:AXIS_COUNT]]
    native = PublicBody(arena)
    candidates = Candidates(JointEffects.load(), native.lower[:AXIS_COUNT], native.upper[:AXIS_COUNT], arena.control_dt)
    receipt = dict(schema="tt27-motor-active-serial-1", source_fit_sha256=_hash(CHECKPOINT),
                   source_game_sha256=_hash(FEEDS), receiving_pose_sha256=_hash(POSES),
                   seeds=SEEDS, nominal_angle_and_velocity_noise=NOMINAL_SIGMA,
                   prep_intervals=AXIS_COUNT, prep_ticks=AXIS_COUNT * TICKS_PER_PORT,
                   active_request_ticks=2 * AXIS_COUNT, peak_request=PULSE, receiver_clock_s=arena.control_dt,
                   restarts_each_setting=AXIS_COUNT,
                   reset_cost="MuJoCo restores ordinary home pose before every packet. Physical reset time is unmeasured.",
                   hypotheses=len(candidates.maps), streams=[])
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        templates = NativeTemplates(arena, dofs)
        receipt["native_templates"] = templates.receipt
        receipt["passive_source"] = calibrate_passive(arena, dofs, history, candidates)
        for seed in SEEDS:
            stream = dict(seed=seed, settings=[])
            receipt["streams"].append(stream)
            rng = np.random.default_rng(seed)
            carried = passive_carry = IDENTITY
            for index, (actual, shift, speed) in enumerate(install_course(rng)):
                shots = course(rng, donors, per_setting + AXIS_COUNT, shift, speed)
                training, fresh = shots[:AXIS_COUNT], shots[AXIS_COUNT:]
                probe_seed = int(rng.integers(2**31))
                traces, probes = [], []
                for port in range(AXIS_COUNT):
                    requests = packet(port)
                    states, applied = collect(arena, dofs, requests, actual, probe_seed)
                    traces.append(states)
                    difference = np.abs(states[:, :AXIS_COUNT] - states[0, :AXIS_COUNT])
                    probes.append(dict(port=port, requests=requests.tolist(), executed=applied.tolist(),
                                       states=states.tolist(), peak_excursion_rad=float(difference.max()),
                                       final_angle_error_rad=float(difference[-1].max())))
                evidence = templates.loss(traces, candidates)
                active, active_trace = stable_choice(evidence, candidates, carried, carried.permutation)
                reset, reset_trace = stable_choice(evidence, candidates, IDENTITY)
                passive, passive_trace = passive_block(arena, mounts, training, pose_memory, actual,
                                                       passive_carry, dofs, candidates)
                passive_warm, warm_trace = passive_block(arena, mounts, training, pose_memory, actual,
                                                         carried, dofs, candidates)
                modes = dict(active=active, active_reset=reset, passive=passive,
                             passive_warm=passive_warm, frozen=carried, factory=IDENTITY, expert=actual)
                rows = []
                for shot in fresh:
                    pick = select(arena, mounts, shot, keys, scale)
                    results, cached = {}, {}
                    for name, candidate in modes.items():
                        if candidate not in cached:
                            scored = play(arena, shot, goals[pick], actual, candidate, dofs)
                            replay = play(arena, shot, goals[pick], actual, candidate, dofs, tape=scored["requests"])
                            if scored != replay:
                                raise ValueError("A physical command replay changed the scored outcome")
                            cached[candidate] = scored
                        results[name] = cached[candidate]
                    rows.append(dict(shot=shot, goal=pick, returns=results, verified=len(cached)))
                settings = dict(number=index, actual=asdict(actual), shift_m=shift, forward_speed_factor=speed,
                                prior_accepted=asdict(carried), probe_seed=probe_seed, probes=probes,
                                estimates={name: describe_fit(value, actual) for name, value in modes.items()},
                                active_trace=active_trace, reset_trace=reset_trace, passive_trace=passive_trace,
                                passive_warm_trace=warm_trace, adaptation_feeds=training, rows=rows,
                                summary=totals(rows))
                stream["settings"].append(settings)
                carried, passive_carry = active, passive
                all_rows = [row for s in receipt["streams"] for case in s["settings"] for row in case["rows"]]
                receipt["summary"] = totals(all_rows)
                _write(path, receipt)
                print(seed, index, "fits", {name: value == actual for name, value in modes.items()},
                      "confirm", [active_trace["confirm_tick"], reset_trace["confirm_tick"],
                                  passive_trace["inference"]["confirm_tick"]],
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
