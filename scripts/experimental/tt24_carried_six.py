"""Carry the reward-trained donor head through archived six-hinge mappings."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from .tt19_view_full_game import digest_json
from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BounceArena
from .tt24_bounce_ident import _INC
from .tt24_bounce_trial import verify_replay
from .tt24_donor_assess import CHECKPOINT, TRAINING
from .tt24_donor_trial import ROOT, Native
from .tt24_launch_context import draw
from .tt24_public_body import FUSED_OUTPUT, load_public
from .tt24_redundant import RedundantBounce
from .tt24_redundant_jam import LAYOUT
from .tt24_redundant_trial import grounding


TAG = 'carried-six'
PLAN = ROOT / f'{TAG}-plan.json'
OUTPUT = ROOT / f'{TAG}.json.gz'
ORIGINAL_CALIBRATION = ROOT / 'redundant-game.json.gz'
JAM_CALIBRATION = ROOT / 'redundant-jam.json.gz'
SEED = 1182837761
COUNT, FRAMES = 8, 420
CONDITIONS = dict(stock=('stock', 'identity'), intact_padded=('normal', 'padded'),
                  intact_learned=('normal', 'normal'), jammed_padded=('jammed', 'padded'),
                  jammed_learned=('jammed', 'jammed'), both_jammed=('both_jammed', 'both_jammed'))


def archive_map():
    stock = _read(ORIGINAL_CALIBRATION)
    faults = _read(JAM_CALIBRATION)
    if faults['source_sha256'] != _hash(ORIGINAL_CALIBRATION):
        raise ValueError('the jam probes use a different frozen stock optical map')
    data = lambda row: {k: np.array(row[k]) for k in ('encoder', 'command', 'actuator')}
    source = data(stock['calibration']['stock'])
    matrices = dict(identity=(np.eye(5).tolist(), np.eye(5).tolist()),
                    padded=(np.eye(6, 5).tolist(), np.eye(5, 6).tolist()))
    for variant in LAYOUT:
        saved = faults['grounding'][variant]
        checked = grounding(source, data(faults['calibration'][variant]))
        if any(not np.allclose(checked[key], saved[key], rtol=1e-10, atol=1e-12) for key in ('virtual', 'transmit')):
            raise ValueError(f'archived calibration no longer reproduces {variant}')
        matrices[variant] = saved['virtual'], saved['transmit']
    return matrices, stock, faults


def register():
    maps, stock, faults = archive_map()
    training = _read(TRAINING)
    known = set(stock['launch_seeds']) | set(faults['held_seeds']) | {stock['seed'], faults['seed']}
    known.update(seed for group in training['dev_seeds'] for seed in group)
    known.update(seed for step in training['updates'] for group in step['training_seeds'] for seed in group)
    known.update(row['seed'] for group in training['held_donor'] for row in group)
    known.update(row['seed'] for row in training['held_stock'])
    if _hash(CHECKPOINT) != training['frozen_sha256'] or not training['donor_gate_passed']:
        raise ValueError('the carried motor must be the qualified reward-trained head')
    seeds = (np.random.default_rng(SEED).choice(2 ** 31 - 2, COUNT, replace=False) + 1).tolist()
    if known.intersection(seeds):
        raise ValueError('held starts must be fresh across the calibrations and donor training')
    plan = dict(schema='tt24-carried-six-1', seed=SEED, launches=seeds, horizon=FRAMES, regime='standard',
                conditions={name: list(value) for name, value in CONDITIONS.items()},
                maps={name: list(value) for name, value in maps.items()},
                sources={path.name: _hash(path) for path in (ORIGINAL_CALIBRATION, JAM_CALIBRATION, CHECKPOINT, FUSED_OUTPUT)},
                public_correspondence='five legacy encoder indices and reset angles plus measured virtual contribution of sixth',
                archived_probe_views=2, new_probe_or_ball_practice=0,
                primary='longest physical consecutive rebound streak on every paired start')
    if PLAN.exists():
        if _read(PLAN) != plan:
            raise ValueError('registered six-hinge comparison changed')
    else:
        _write(PLAN, plan)
    return plan


def play(arena, native, weights, virtual, transmit, seed):
    obs = arena.reset(seed=seed, split='held', regime='standard')
    frozen = load_public(FUSED_OUTPUT)
    home = obs['joints'].astype(float)
    source_issued = np.zeros(5)
    actions, encoders, proposals = [], [], []
    total = 0.
    stock = arena.action_size == 5
    padded = np.array_equal(virtual, np.eye(6, 5))
    while True:
        if stock:
            sensed = obs
        elif padded:
            sensed = obs | dict(joints=obs['joints'][:5], joint_velocity=obs['joint_velocity'][:5])
        else:
            displacement = obs['joints'].astype(float) - home
            sensed = obs | dict(joints=home[:5] + displacement @ virtual,
                                joint_velocity=obs['joint_velocity'] @ virtual)
        policy = draw(native, frozen.observe(sensed), weights, source_issued)
        target = policy if stock else policy @ transmit
        saturation = max(1., float(np.max(np.abs(target))))
        rates = target / saturation
        if saturation > 1.:
            source_issued -= _INC * policy * (1. - 1. / saturation)
        obs, reward, done = arena.step(rates)
        total += reward
        actions.append(rates.tolist())
        encoders.append(obs['joints'].tolist())
        proposals.append(policy.tolist())
        if done:
            break
    judge = arena.judge_metrics()
    report = judge | dict(score=total, actions=actions, source_proposals=proposals,
                          starting_encoders=home.tolist(), encoders=encoders)
    verify_replay(arena, report)
    if digest_json(arena.judge_metrics()) != digest_json(judge):
        raise ValueError('physical replay changed the complete judge report')
    return report


def summary(rows):
    streaks = [row['longest_consecutive_rebounds'] for row in rows]
    peak = np.array([np.max(np.abs(np.array(row['encoders']) - row['starting_encoders']), axis=0) for row in rows])
    rates = np.concatenate([row['actions'] for row in rows], axis=0)
    active_six = peak.shape[1] == 6
    return dict(streaks=streaks, mean=float(np.mean(streaks)), total=sum(row['rebounds'] for row in rows),
                longest_sum=sum(streaks), at_least_ten=sum(k >= 10 for k in streaks),
                at_least_twenty=sum(k >= 20 for k in streaks), maximum=max(streaks),
                legacy_pitch_max_rad=float(np.max(peak[:, 3])),
                sixth_median_excursion_rad=float(np.median(peak[:, 5])) if active_six else None,
                sixth_max_excursion_rad=float(np.max(peak[:, 5])) if active_six else None,
                sixth_rms_rate=float(np.sqrt(np.mean(rates[:, 5]**2))) if active_six else None)


def evaluate(task):
    condition, seeds, variant, pair = task
    virtual, transmit = map(np.asarray, pair)
    specs = dict(view='wrist', width=192, height=144, max_steps=FRAMES)
    arena = BounceArena(**specs) if variant == 'stock' else RedundantBounce(**LAYOUT[variant], **specs)
    native = Native()
    try:
        with np.load(CHECKPOINT, allow_pickle=False) as saved:
            weights = np.ascontiguousarray(saved['weights'], dtype=np.float64)
        rows = [play(arena, native, weights, virtual, transmit, seed) for seed in seeds]
        summarized = summary(rows)
        print(condition, summarized, flush=True)
        return condition, dict(summary=summarized, episodes=rows)
    finally:
        arena.close()
        native.working.cleanup()


def comparisons(data):
    def pair(learned, control):
        a, b = (data[name]['summary']['streaks'] for name in (learned, control))
        difference = np.asarray(a) - b
        return dict(differences=difference.tolist(), mean=float(difference.mean()),
                    wins=int(np.sum(difference > 0)), ties=int(np.sum(difference == 0)), losses=int(np.sum(difference < 0)))

    return {control: pair('jammed_learned', control) for control in ('jammed_padded', 'both_jammed', 'intact_learned')}


def run(plan, workers):
    receipt = _read(OUTPUT) if OUTPUT.exists() else dict(plan_sha256=_hash(PLAN), results={})
    if receipt['plan_sha256'] != _hash(PLAN):
        raise ValueError('held six-hinge runs came from a different plan')
    tasks = [(mode, plan['launches'], variant, plan['maps'][key]) for mode, (variant, key) in plan['conditions'].items()
             if mode not in receipt['results']]
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        for mode, results in pool.map(evaluate, tasks):
            receipt['results'][mode] = results
            _write(OUTPUT, receipt)
    receipt['paired_jam'] = comparisons(receipt['results'])
    _write(OUTPUT, receipt)
    print('paired jam differences', receipt['paired_jam'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    plan = register()
    if not args.plan_only:
        run(plan, args.workers)


if __name__ == '__main__':
    main()
