"""Blend two frozen readouts using source-only encoder distributions at release."""

import argparse
from hashlib import sha256

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_donor_assess import ShiftedBounce
from .tt24_donor_trial import DESIGNS, FUSED_OUTPUT, ROOT, DonorBounce, Native, _seeds, load_public
from .tt24_posture import WidePosture
from .tt24_tilted_start_trial import (CHECKPOINT, CPP, ORIGINAL, OUTPUT, SOURCE, WideDonor, WideShifted,
                                     conclusions, matched, saved_original)
from .tt4_transfer_trial import _read, _write


GATE = ROOT / 'tt24_tilted_start_context.npz'
RESULT = ROOT / 'tt24_tilted_start_context_held.json.gz'
SAMPLES = 64
CANDIDATES = 12


def orientation(decoder, observation):
    """Estimate orientation from public vendor kinematics and encoder angles."""
    return float(np.linalg.norm(decoder.body(observation['joints'])[15:17]))


def fit_gate():
    decoder = load_public(FUSED_OUTPUT).decoder
    tilt = []
    seed_sets = _seeds(920641722, len(DESIGNS), 2*SAMPLES)
    for kind, factory in enumerate((DonorBounce, WideDonor)):
        samples = []
        for body, seeds in zip(DESIGNS, seed_sets, strict=True):
            arena = factory(body)
            try:
                for seed in seeds[kind*SAMPLES:(kind+1)*SAMPLES]:
                    obs = arena.reset(seed=seed, split='held', regime='changed', render=False)
                    samples.append(orientation(decoder, obs))
            finally:
                arena.close()
        tilt.append(samples)
    low = max(tilt[0])
    high = float(np.quantile(tilt[1], .25))
    if high <= low:
        raise ValueError('the source orientation distributions do not support a separation')
    pending = GATE.with_name(GATE.stem + '.tmp.npz')
    np.savez_compressed(pending, ordinary=tilt[0], wide=tilt[1], low=low, high=high,
                        source_encoder_seeds=seed_sets)
    pending.replace(GATE)
    return low, high


def blend(value, lower, upper):
    amount = float(np.clip((value-lower)/(upper-lower), 0., 1.))
    return amount * amount * (3. - 2.*amount)


def independent_seed_sets():
    prior = _read(OUTPUT)
    source = _read(SOURCE)
    seen = {row['seed'] for group in prior['targets'].values() for row in group['cases']}
    for group in prior['held_donors']:
        seen.update(group['seeds'])
    for key in ('validation_wide_seeds', 'validation_ordinary_seeds'):
        seen.update(seed for row in source[key] for seed in row)
    seen.update(seed for step in source['updates'] for row in step['practice_seeds'] for seed in row)
    seen.update(seed for row in _seeds(920641722, len(DESIGNS), 2*SAMPLES) for seed in row)
    fresh = _seeds(1064110273, 4, CANDIDATES)
    if seen.intersection(seed for row in fresh for seed in row):
        raise ValueError('new held release overlaps a used start')
    return fresh


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    source = _read(SOURCE)
    if source['checkpoint_sha256'] != sha256(CHECKPOINT.read_bytes()).hexdigest():
        raise ValueError('broad-trained actor changed after selection')
    if source['original_sha256'] != sha256(ORIGINAL.read_bytes()).hexdigest():
        raise ValueError('original actor changed after selection')
    lower, upper = fit_gate()
    decoder = load_public(FUSED_OUTPUT).decoder
    with np.load(CHECKPOINT, allow_pickle=False) as fit:
        wide_policy = np.ascontiguousarray(fit['warm'])
    original = saved_original()
    report = dict(source_gate_sha256=sha256(GATE.read_bytes()).hexdigest(), novel_releases=True,
                  no_target_outcomes_used_for_gate=True, gate_lower=lower, gate_upper=upper,
                  gate_fit_samples_per_distribution=len(DESIGNS)*SAMPLES, held={})
    _write(RESULT, report)
    native = Native(CPP)
    specs = (('wide_stock', lambda: WidePosture(view='wrist', width=192, height=144), 'standard'),
             ('ordinary_stock', lambda: BounceArena(view='wrist', width=192, height=144), 'standard'),
             ('wide_altered', WideShifted, 'changed'),
             ('ordinary_altered', ShiftedBounce, 'changed'))
    try:
        for (label, factory, regime), starts in zip(specs, independent_seed_sets(), strict=True):
            arena = factory()
            cases = []
            try:
                for seed in starts:
                    obs = arena.reset(seed=seed, split='held', regime=regime, render=False)
                    initial = orientation(decoder, obs)
                    amount = blend(initial, lower, upper)
                    original_run = matched(arena, native, original, seed, regime, 'wrist')
                    wide_run = matched(arena, native, wide_policy, seed, regime, 'wrist')
                    if amount == 0.:
                        context_run = original_run
                    elif amount == 1.:
                        context_run = wide_run
                    else:
                        policy = original + amount * (wide_policy-original)
                        context_run = matched(arena, native, policy, seed, regime, 'wrist')
                    cases.append(dict(seed=seed, estimated_horizontal_normal_length=initial, blend=amount,
                                      conditions=dict(context=context_run, original=original_run, wide=wide_run)))
                    print(label, seed, round(amount, 2),
                          [r['metrics']['longest_consecutive_rebounds'] for r in (context_run, original_run, wide_run)],
                          flush=True)
                report['held'][label] = dict(cases=cases, summary=conclusions(cases),
                                             distinct_tapes_verified=2*len(cases)+sum(0 < r['blend'] < 1 for r in cases))
                _write(RESULT, report)
            finally:
                arena.close()
    finally:
        native.working.cleanup()


if __name__ == '__main__':
    main()
