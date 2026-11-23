"""Collect matched SO-101 action-effect experience from physical ball drops."""

from dataclasses import asdict, dataclass, replace
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _write
from .tt24_bounce import BounceArena, _sample_drop
from .tt24_bounce_ident import CleanState
from .tt24_carried_wrist import commands
from .tt24_donor_assess import CHECKPOINT as INITIAL_WEIGHTS
from .tt26_bounce_effects import CHECKPOINT, CONTROL_INTERVAL, JointEffects, ORDINARY, OUTPUT, STEP


DATA = OUTPUT / 'bounce-encoder-practice.npz'
RECEIPT = OUTPUT / 'bounce-effects.json.gz'
FIELDS = ('q', 'qvel', 'target', 'action', 'next_q', 'next_qvel')
PRACTICE_SEED = 261700400
HELD_SEED = 261780400
CAP = 240


@dataclass(frozen=True)
class Variation:
    gravity: float
    drag: float
    damping: float
    drive: tuple[float, ...]
    lateral_impulse: tuple[float, float]
    height_scale: float


VARIATIONS = (
    Variation(.92, .88, -2., (1., .93, .95, 1., 1.04), (.052, -.033), .93),
    Variation(1., 1.12, 2., (.94, 1.02, 1., .90, 1.), (-.048, .038), 1.13),
    Variation(1.07, 1.03, 4., (1.04, 1., .94, 1.03, .91), (.012, .064), 1.03),
)


class VariedBounce(BounceArena):
    """Keep the stock SO-101 and change physical drop and servo conditions."""

    def __init__(self, variation, **kwargs):
        self.variation = variation
        super().__init__(**kwargs)
        self._model.opt.gravity[2] *= variation.gravity
        drive = np.r_[variation.drive, 1.]
        self._standard_gain *= drive
        self._standard_bias *= drive[:, None]

    def sample_drop(self, seed, split, regime):
        drop = _sample_drop(seed, split, regime)
        rng = np.random.default_rng(seed ^ 28915743)
        width = np.array((.19, .13, .16, .14, .10))
        pose = np.asarray(drop.pose) + np.r_[rng.uniform(-width, width), 0.]
        offset = np.asarray(drop.offset)
        offset[2] *= self.variation.height_scale
        velocity = np.asarray(drop.velocity)
        velocity[:2] += rng.uniform(.4, 1.3) * np.asarray(self.variation.lateral_impulse)
        return replace(drop, pose=tuple(pose), offset=tuple(offset), velocity=tuple(velocity),
                       drag=drop.drag * self.variation.drag, damping=drop.damping + self.variation.damping)


def initial_weights():
    with np.load(INITIAL_WEIGHTS, allow_pickle=False) as data:
        return data['weights']


def encoder_rollout(arena, seed, weights, *, split, explore=True):
    """Run a real release and retain issued commands and sampled encoders."""
    obs = arena.reset(seed=seed, split=split, render=False)
    rng = np.random.default_rng(seed ^ 10232729)
    observer = CleanState(arena)
    initial_target = np.asarray(arena._drop.pose[:5])
    target = initial_target.copy()
    action = np.zeros(5)
    tape, transitions = [], []
    for tick in range(arena.max_steps):
        if tick % 2 == 0:
            action = commands(observer.observe(obs), weights, target - initial_target)
            if explore:
                dither = rng.uniform(-.22, .22, 5)
                if rng.random() < .20:
                    dither[rng.integers(5)] *= 2.
                action = np.clip(action + dither, -.77, .77)
        prior_time = float(arena._data.time)
        before = (obs['joints'], obs['joint_velocity'], target, action)
        obs, _, done = arena.step(action, render=False)
        tape.append(action.tolist())
        if np.isclose(arena._data.time - prior_time, CONTROL_INTERVAL, atol=1.e-8):
            transitions.append((*before, obs['joints'], obs['joint_velocity']))
        target = np.clip(target + STEP * action, arena._control_low[:5], arena._control_high[:5])
        if done:
            break
    return np.asarray(transitions, dtype=float).reshape(-1, len(FIELDS), 5), arena.judge_metrics(), tape


def collect(weights, split, seed_base, releases):
    groups = {}
    for mode in ('varied', 'ordinary'):
        records = []
        for profile, variation in enumerate(VARIATIONS):
            factory = VariedBounce if mode == 'varied' else BounceArena
            args = (variation,) if mode == 'varied' else ()
            arena = factory(*args, view='none', max_steps=CAP)
            try:
                for seed in range(seed_base + 100 * profile, seed_base + 100 * profile + releases):
                    samples, metrics, tape = encoder_rollout(arena, seed, weights, split=split)
                    records.append((samples, dict(profile=profile, metrics=metrics, actions=tape)))
            finally:
                arena.close()
        groups[mode] = records
    return groups


def record_arrays(groups):
    return (np.concatenate([samples for samples, _ in groups]),
            np.repeat([row['metrics']['seed'] for _, row in groups], [len(samples) for samples, _ in groups]),
            np.concatenate([np.arange(len(samples)) for samples, _ in groups]))


def errors(observed, predicted):
    residual = 1000. * (observed - predicted)
    return dict(rms_mrad=float(np.sqrt(np.mean(residual ** 2))),
                axis_rms_mrad=np.sqrt(np.mean(residual ** 2, axis=0)).tolist())


def score(model, records):
    data, seeds, ticks = record_arrays(records)
    q, qvel, target, action, next_q, _ = np.moveaxis(data, 1, 0)
    next_indices, = np.where((seeds[1:] == seeds[:-1]) & (ticks[:-1] % 2 == 0))
    selected = (q[next_indices], qvel[next_indices], target[next_indices], action[next_indices])
    return dict(
        one_tick=errors(next_q - q, model.predict(q, qvel, target, action)),
        two_ticks=errors(next_q[next_indices + 1] - selected[0], model.predict(*selected, steps=2)),
        no_new_action=errors(next_q - q, model.predict(q, qvel, target, np.zeros_like(action))),
        ideal_joint_rate=errors(next_q - q, STEP * action),
        hold_encoder_velocity=errors(next_q - q, CONTROL_INTERVAL * qvel),
        assessed_steps=len(q), two_tick_windows=len(next_indices),
    )


def summarize(records):
    results = [row['metrics'] for _, row in records]
    return dict(drops=len(results), encoder_transitions=sum(len(samples) for samples, _ in records),
                physical_steps=sum(row['steps'] for row in results), scored_rebounds=sum(row['rebounds'] for row in results),
                at_least_two=sum(row['longest_consecutive_rebounds'] >= 2 for row in results),
                longest=max(row['longest_consecutive_rebounds'] for row in results))


def run():
    began = perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    weights = initial_weights()
    practice = collect(weights, 'practice', PRACTICE_SEED, releases=14)
    arrays = {mode: record_arrays(records) for mode, records in practice.items()}
    budget = min(len(data) for data, _, _ in arrays.values())
    rng = np.random.default_rng(28559001)
    payload, models = {}, {}
    for mode, (data, seeds, ticks) in arrays.items():
        chosen = np.sort(rng.choice(len(data), size=budget, replace=False))
        models[mode] = JointEffects.fit(*np.moveaxis(data[chosen], 1, 0))
        payload.update({f'{mode}_{name}': data[:, i] for i, name in enumerate(FIELDS)})
        payload.update({f'{mode}_seeds': seeds, f'{mode}_ticks': ticks, f'{mode}_fit_rows': chosen})
    np.savez_compressed(DATA, **payload)
    for mode, path in (('varied', CHECKPOINT), ('ordinary', ORDINARY)):
        models[mode].save(path, mode=mode, sample_count=budget, practice_hash=_hash(DATA))
    print('Frozen bounce effects', budget, {mode: summarize(records) for mode, records in practice.items()}, flush=True)
    held = collect(weights, 'held', HELD_SEED, releases=6)
    report = dict(schema='tt26-bounce-effects-receipt-1', input_weights_sha256=_hash(INITIAL_WEIGHTS),
                  source_observations='metric ball and blade for inherited bounce control, five joint encoders for effects fit',
                  varied_conditions=[asdict(variation) for variation in VARIATIONS],
                  action_dither='uniform +/-0.22 with occasional larger single-axis perturbation, held for 2 ticks',
                  fit_samples_each=budget, raw_practice_sha256=_hash(DATA),
                  frozen_before_held=True, hold_jaw=True,
                  model_sha256={p.name: _hash(p) for p in (CHECKPOINT, ORDINARY)},
                  practice={mode: summarize(records) for mode, records in practice.items()},
                  held={mode: summarize(records) for mode, records in held.items()},
                  held_prediction={mode: {name: score(JointEffects.load(path), data) for name, path in
                                          (('varied', CHECKPOINT), ('ordinary', ORDINARY))}
                                   for mode, data in held.items()},
                  practice_releases={mode: [row for _, row in data] for mode, data in practice.items()},
                  held_releases={mode: [row for _, row in data] for mode, data in held.items()},
                  elapsed_s=perf_counter() - began)
    _write(RECEIPT, report)
    print('Held bounce prediction', report['held_prediction'], flush=True)


if __name__ == '__main__':
    run()
