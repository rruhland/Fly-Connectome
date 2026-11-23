"""Reuse inferred relative changes to shorten a continuous motor diagnostic."""

import argparse
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_so101 import AXES
from .tt27_motor_active import encoder
from .tt27_motor_active_continuous import ContinuousReceiver, PACKET_TICKS, PULSE, WARMUP, receive
from .tt27_motor_interface import AXIS_COUNT, IDENTITY, Candidates, Interface, course
from .tt27_world_camera import FEEDS, POSES


OUT = Path("artifacts/tt27/motor-recurrence.json.gz")
SEEDS = (4012649, 8175539)
SOURCE_SEED = 4037015
AUDIT_PORTS = 3
SIGMA = .0018
MARGIN = 9.
ROSTER = 6
OPERATIONS = (
    Interface((1, 0, 2, 3, 4), (-1, 1, -1, 1, 1), 1),
    Interface((0, 1, 3, 2, 4), (1, -1, 1, 1, -1), 0),
    Interface((4, 1, 2, 3, 0), (1, 1, 1, -1, 1), 1),
)


def compose(old, change):
    return Interface(tuple(change.permutation[p] for p in old.permutation),
                     tuple(s * change.signs[p] for p, s in zip(old.permutation, old.signs, strict=True)),
                     old.delay ^ change.delay)


def relative(old, new):
    permutation, signs = [0] * AXIS_COUNT, [0] * AXIS_COUNT
    for i, x in enumerate(old.permutation):
        permutation[x] = new.permutation[i]
        signs[x] = new.signs[i] * old.signs[i]
    return Interface(tuple(permutation), tuple(signs), old.delay ^ new.delay)


def signature(wiring, port):
    axis = wiring.permutation.index(port)
    return axis, int(wiring.signs[axis] < 0), wiring.delay


class Pulse:
    """Hold the incoming ball and move the same arm throughout the diagnostic."""

    def __init__(self, arena, dofs, wiring, seed):
        self.arena, self.dofs, self.wiring = arena, dofs, wiring
        self.previous = np.zeros(AXIS_COUNT)
        arena.hold(seed)
        self.advance(np.zeros((WARMUP, AXIS_COUNT)))

    def advance(self, commands):
        samples = [encoder(self.arena, self.dofs)]
        for request in commands:
            actual = self.wiring.apply(self.previous if self.wiring.delay else request)
            _, _, done, cap, _ = self.arena.step(np.r_[actual, 0.], render=False)
            if done or cap or not self.arena.grasped:
                raise ValueError("Diagnostic contact or terminal event")
            self.previous = request
            samples.append(encoder(self.arena, self.dofs))
        return np.asarray(samples)

    def observe(self, port, sign=1):
        commands = np.zeros((PACKET_TICKS, AXIS_COUNT))
        commands[:2, port] = np.array((PULSE, -PULSE)) * sign
        return self.advance(commands)


class Templates:
    def __init__(self, arena, dofs):
        self.labels = tuple(product(range(AXIS_COUNT), range(2), range(2)))
        self.templates = []
        for axis, negative, delay in self.labels:
            native = Interface(IDENTITY.permutation, IDENTITY.signs, delay)
            observed = Pulse(arena, dofs, native, SOURCE_SEED).observe(axis, (-1)**negative)
            self.templates.append(observed[1:] - observed[0])
        self.templates = np.asarray(self.templates)

    def measure(self, trace):
        gap = (self.templates - (trace[1:] - trace[0])) / SIGMA
        evidence = np.cumsum(np.square(gap).sum(axis=2), axis=1)
        scores = evidence[:, -1]
        order = np.argsort(scores)
        best, second = order[:2]
        stable = best == np.argmin(evidence[:, -2])
        margin = scores[second] - scores[best]
        if not stable or margin < MARGIN:
            raise ValueError("An isolated encoder signature was inconclusive")
        return self.labels[best], float(margin)


def distinguish(unobserved, hypotheses):
    """Prefer a cable whose responses split the currently plausible changes."""
    def alternatives(port):
        return len({signature(hypothesis, port) for hypothesis in hypotheses})
    return max(unobserved, key=alternatives)


def identify(arena, dofs, bank, candidates, actual, seed, *, prior=()):
    active = Pulse(arena, dofs, actual, seed)
    remaining = list(range(AXIS_COUNT))
    full, family = candidates[:], list(prior)
    rows, contradicted = [], False
    while remaining:
        port = distinguish(remaining, family) if family else remaining[0]
        remaining.remove(port)
        trace = active.observe(port)
        label, margin = bank.measure(trace)
        full = [guess for guess in full if signature(guess, port) == label]
        matched = [guess for guess in family if signature(guess, port) == label]
        contradicted |= bool(family and not matched)
        family = matched
        rows.append(dict(port=port, signature=label, margin=margin, family_left=len(family),
                         full_left=len(full), encoder=trace.tolist()))
        if len(family) == 1 and len(rows) >= AUDIT_PORTS:
            return family[0], dict(probes=rows, contradicted=contradicted, route="recurrent")
    if len(full) != 1:
        raise ValueError("Full port acquisition left an ambiguous wiring")
    return full[0], dict(probes=rows, contradicted=contradicted, route="exhaustive")


def make_schedule(rng):
    old = IDENTITY
    used = {old}
    source = []
    for op in OPERATIONS:
        old = compose(old, op)
        source.append(old)
        used.add(old)
    surprise = int(rng.integers(2, ROSTER - 1))
    held = []
    for i in range(ROSTER):
        if i == surprise:
            while True:
                trial = Interface(tuple(rng.permutation(AXIS_COUNT).tolist()),
                                  tuple(rng.choice((-1, 1), AXIS_COUNT).tolist()), int(rng.integers(2)))
                if trial not in used and relative(old, trial) not in OPERATIONS:
                    break
        else:
            available = [compose(old, op) for op in OPERATIONS if compose(old, op) not in used]
            trial = available[int(rng.integers(len(available)))]
        held.append((trial, i == surprise))
        used.add(trial)
        old = trial
    return source, held


def score(arena, dofs, actual, map_estimate, probe, shot, memory, seed, *, capture=False):
    def replay(tape=None):
        setup = Pulse(arena, dofs, actual, seed)
        for item in probe["probes"]:
            observed = setup.observe(item["port"])
            if not np.array_equal(observed, item["encoder"]):
                raise ValueError("Preparation did not replay before the real feed")
        duration = WARMUP + PACKET_TICKS * len(probe["probes"])
        return receive(arena, shot, actual, map_estimate, memory, tape, preparation_ticks=duration,
                       encoder_dofs=dofs if capture else None)
    result = replay()
    verification = replay(result["requests"])
    if any(result[key] != verification[key] for key in result if key != "goal_index"):
        raise ValueError("Receiving after the shortened diagnostic did not replay")
    return result


def run(path, per_change):
    given = [r for r in _read(FEEDS)["cases"] if r["angle"] < 0]
    donors = {r["shot"]["id"]: r["shot"] for r in given}
    with np.load(POSES, allow_pickle=False) as data:
        memory = tuple(data[name] for name in ("keys", "goals", "scale"))
    arena = ContinuousReceiver()
    dofs = [arena.model.joint("learner_" + axis).dofadr[0] for axis in AXES[:AXIS_COUNT]]
    candidates = Candidates(None, None, None, arena.control_dt).maps
    receipt = dict(schema="tt27-motor-recurrence-1", source_sha256=_hash(FEEDS), pose_sha256=_hash(POSES),
                   seeds=SEEDS, n_templates=20, template_ticks=20 * (WARMUP + PACKET_TICKS),
                   warmup_ticks=WARMUP, pulse_move_ticks=2, packet_settle_ticks=PACKET_TICKS - 2,
                   max_request=PULSE, tick_s=arena.control_dt, audit_ports=AUDIT_PORTS,
                   candidate_count=len(candidates), streams=[])
    try:
        bank = Templates(arena, dofs)
        for seed in SEEDS:
            rng = np.random.default_rng(seed)
            source, held = make_schedule(rng)
            block = dict(seed=seed, source=[], held=[])
            receipt["streams"].append(block)
            previous = IDENTITY
            remembered = []
            for actual in source:
                measured, preparation = identify(arena, dofs, bank, candidates, actual, SOURCE_SEED)
                remembered.append(relative(previous, measured))
                block["source"].append(dict(actual=asdict(actual), inferred=asdict(measured),
                                            observed_relative=asdict(remembered[-1]), **preparation))
                previous = measured
            for sequence, (actual, surprise) in enumerate(held):
                suggested = list(dict.fromkeys(compose(previous, op) for op in remembered))
                mode, active = identify(arena, dofs, bank, candidates, actual, SOURCE_SEED, prior=suggested)
                blank, reset = identify(arena, dofs, bank, candidates, actual, SOURCE_SEED)
                shots = course(rng, donors, per_change, .024 * (-1)**sequence, (1.03, .97)[sequence % 2])
                rows = [dict(shot=shot, active=score(arena, dofs, actual, mode, active, shot, memory, SOURCE_SEED),
                             reset=score(arena, dofs, actual, blank, reset, shot, memory, SOURCE_SEED))
                        for shot in shots]
                block["held"].append(dict(index=sequence, actual=asdict(actual), unexpected=surprise,
                                          carried=asdict(mode), reset=asdict(blank), active_exact=mode == actual,
                                          reset_exact=blank == actual, prior=[asdict(r) for r in suggested],
                                          active_probe=active, reset_probe=reset, rows=rows))
                op = relative(previous, mode)
                if op not in remembered:
                    remembered.append(op)
                previous = mode
                print(seed, sequence, surprise, mode == actual, blank == actual, len(active["probes"]),
                      len(reset["probes"]), [sum(r[name]["legal"] for r in rows) for name in ("active", "reset")],
                      flush=True)
                _write(path, receipt)
    finally:
        arena.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--per-change", type=int, default=3)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    run(args.output, args.per_change)
