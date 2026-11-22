"""Tune an RGB motor policy from complete native alternate-return episodes."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_direct_receive import previous_seeds
from .tt20_natural import NaturalViewTransport
from .tt22_precision_trial import CONFIG, OUTPUT, PrecisionRallyArena, PrecisionStation


TRAINING_RNG = (348503617, 1991081720)
HELD_RNG = 683045219
ARMS = {'stock': 1.0, 'softer': .72, 'brisker': 1.28}
SPREAD = .30
PACE = .82
DEST = OUTPUT/'tt22_rally_bandit.json.gz'


def schedule():
    training = [np.random.default_rng(seed).integers(2**31, size=2).tolist() for seed in TRAINING_RNG]
    held = np.random.default_rng(HELD_RNG).integers(2**31, size=12).tolist()
    return training, held


def metrics(rows):
    lengths = [r['longest'] for r in rows]
    return dict(points=len(rows), lengths=lengths, total=sum(lengths), longest=max(lengths, default=0),
                ten_or_more=sum(n>=10 for n in lengths), twenty_or_more=sum(n>=20 for n in lengths),
                learner_returns=sum(r['outcome']['legal_returns'] for r in rows),
                rival_returns=sum(r['outcome']['rival_returns'] for r in rows),
                learner_contacts=sum(r['outcome']['contacts'] for r in rows),
                wins=sum(r['reward']>0 for r in rows), frames=sum(len(r['actions']) for r in rows))


def rank(blocks):
    points = [row for block in blocks for row in block['points']]
    total = {arm: sum(p['arms'][arm]['longest'] for p in points) for arm in ARMS}
    # Ties deliberately retain the unmodified policy, then fixed arm order.
    return max(ARMS, key=total.__getitem__), total


def verify(arena, row):
    arena.reset(seed=row['seed'], server=-1, render=False)
    far = row['far_actions']
    arena._aim = lambda *_: far[arena.steps]
    digest = sha256()
    for tick, (near, opponent) in enumerate(zip(row['actions'], far, strict=True)):
        for stream, frame in arena.render_cameras().items():
            digest.update(stream.encode())
            digest.update(frame)
        _, reward, ended, capped, _ = arena.step(near, render=False)
        if bool(ended or capped) != (tick==len(far)-1):
            raise ValueError(f'point ended at a different step for seed {row["seed"]}')
    if (arena.point_result()!=row['outcome'] or reward!=row['reward'] or digest.hexdigest()!=row['rgb']
            or json.dumps(arena.events)!=json.dumps(row['events'])):
        raise ValueError(f'camera or strict physical replay changed for seed {row["seed"]}')


def _record_seed(seed, outcomes, key):
    found = next((row for row in outcomes if row['seed']==seed), None)
    if found is not None:
        return found
    row = dict(seed=seed, **{key:{}})
    outcomes.append(row)
    return row


def run(output):
    train, held = schedule()
    all_seeds = [seed for block in train for seed in block] + held
    hashes = {name: _hash(Path(__file__).with_name(name)) for name in (
        'tt22_rally_bandit.py','tt22_rally.py','tt22_precision.py','tt22_precision_trial.py','tt22_rally_trial.py')}
    known = previous_seeds(OUTPUT)
    data = None
    if output.exists():
        data = _read(output)
        if data['train_starts']!=train or data['held_starts']!=held or data['arms']!=ARMS or data['implementation']!=hashes:
            raise ValueError('existing result uses another schedule or implementation')
        known.difference_update(all_seeds)
    if len(set(all_seeds)) != len(all_seeds) or known.intersection(all_seeds):
        raise ValueError('training and held ordinary physical starts must be fresh and disjoint')
    if data is None:
        data = dict(schema='tt22-whole-rally-pace-search-1', arena=CONFIG, opponent='five-axis physical precision rival',
                    spread=SPREAD, launch_pace=PACE, train_rng=TRAINING_RNG, held_rng=HELD_RNG, arms=ARMS,
                    train_starts=train, held_starts=held, training=[dict(rng=r, points=[]) for r in TRAINING_RNG],
                    held=[], selected=None, learner_inputs=['native fixed stereo RGB','clock','own motor commands'],
                    training_credit='number of consecutive legal alternating returns in the complete point',
                    held_updates=0, implementation=hashes)
    station = PrecisionStation(SPREAD, PACE)
    verification = PrecisionRallyArena(SPREAD, PACE)
    started = perf_counter()
    prior_time = data.get('wall_s',0.)

    def play(seed, arm):
        station.player.pace_scale = ARMS[arm]
        # Each experiment starts with a fresh fixed-camera registration.
        station.optics = NaturalViewTransport('fixed', lens_fov=station.arena.lens_intrinsics(),
                                               law=station.player.law, flight=False, associate=True)
        row = station.point(seed)
        verify(verification, row)
        row.update(pace_scale=ARMS[arm], arm=arm, verified=True)
        return row

    def save():
        data['wall_s'] = prior_time + perf_counter()-started
        data['verified_episodes'] = sum(len(row['arms']) for block in data['training'] for row in block['points'])
        data['verified_episodes'] += sum(len({r['arm'] for r in row['conditions'].values()}) for row in data['held'])
        _write(output,data)

    try:
        for block, seeds in zip(data['training'],train,strict=True):
            for seed in seeds:
                choice = _record_seed(seed, block['points'], 'arms')['arms']
                for name in np.random.default_rng(seed).permutation(tuple(ARMS)):
                    arm = str(name)
                    if arm not in choice:
                        choice[arm] = play(seed,arm)
                        save()
                        print('practice', block['rng'], seed, arm, choice[arm]['longest'], flush=True)
        selected, totals = rank(data['training'])
        if data['selected'] not in (None,selected):
            raise ValueError('a previous selection no longer reproduces')
        data['selected'] = selected
        data['training_totals'] = totals
        data['training_block_totals'] = [rank([block])[1] for block in data['training']]
        data['training_budget'] = {arm:metrics([row['arms'][arm] for b in data['training'] for row in b['points']])
                                   for arm in ARMS}
        save()
        print('frozen selection',selected,totals,flush=True)
        for index,seed in enumerate(held):
            branches = _record_seed(seed,data['held'],'conditions')['conditions']
            for label in (('stock','selected') if index%2==0 else ('selected','stock')):
                if label in branches:
                    continue
                arm = selected if label=='selected' else 'stock'
                if selected=='stock' and branches:
                    branches[label] = next(iter(branches.values()))
                else:
                    branches[label]=play(seed,arm)
                save()
                print('held',seed,label,branches[label]['longest'],flush=True)
    finally:
        verification.close()
        station.close()
    data['summary'] = {name:metrics([row['conditions'][name] for row in data['held']]) for name in ('stock','selected')}
    pairs = [tuple(row['conditions'][name]['longest'] for name in ('stock','selected')) for row in data['held']]
    data['paired'] = dict(gains=sum(b>a for a,b in pairs), losses=sum(a>b for a,b in pairs), ties=sum(a==b for a,b in pairs))
    save()
    print(data['summary'], data['paired'], flush=True)
    return data


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=DEST)
    a=p.parse_args()
    run(a.output)
