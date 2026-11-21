"""Compare score-trained Pong relation weights on fresh SO-101 RGB predictions."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory

import numpy as np
from scipy.stats import rankdata
import torch

from .tt3_pong import source_episodes
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt20_fixture import TT20Arena
from .tt22_relation import BASE, FEATURE_HORIZON, Observation, learn, predict, targets


OUTPUT = ROOT / 'artifacts/tt22'
TT20 = ROOT / 'artifacts/tt20'
TT21 = ROOT / 'artifacts/tt21'
SOURCE_SEED = 1561940299
TARGET_SPLIT = 266405102
SOURCE_SHUFFLE = 772318503
MODES = ('pong', 'score_shuffled', 'relation_shuffled', 'scratch', 'unary')
BUDGETS = (0, 4, 12, 20)


def library(folder):
    code = Path(__file__).with_name('tt3_components.cpp')
    shared = Path(folder) / 'components.so'
    subprocess.run(['g++', '-O3', '-std=c++17', '-shared', '-fPIC', str(code), '-o', str(shared)], check=True)
    return str(shared)


def pong(shared, count):
    rng = np.random.default_rng(SOURCE_SEED)
    seeds = rng.integers(2**31, size=count).tolist()
    scenes, metadata = [], []
    for index, samples in enumerate(source_episodes(seeds, max_ticks=1200, batch_size=16)):
        observer = Observation(shared)
        digest = sha256()
        features = []
        for sample in samples:
            digest.update(sample.frames)
            features.append(observer.update(sample.frames, sample.preceding_action))
        reward = samples[-1].reward
        scenes.append((np.asarray(features, np.float32), float(reward)))
        metadata.append(dict(seed=seeds[index], frames=len(features), reward=reward, rgb_sha256=digest.hexdigest()))
        if (index + 1) % 32 == 0:
            print('Pong', index + 1, flush=True)
    return scenes, metadata


def archived():
    """Return explicit already-scored full points from four disjoint archive cohorts."""
    sources = (
        (TT20 / 'mono-flight-outside.json.gz',
         lambda data: [p['temporal'] for p in data['pairs']['outside']], 'rgb_sha256', True),
        (TT21 / 'demonstration-held.json.gz',
         lambda data: [p['frozen'] for p in data['rounds']], 'rgb_sha256', False),
        (TT20 / 'arc-carry.json.gz', lambda data: [p['optics'] for p in data['held']['outside']], 'rgb', True),
        (TT20 / 'camera-carry-left-game-40582930-0.json.gz',
         lambda data: [p for p in data['points'] if p['role'] == -1], 'rgb_sha256', False),
    )
    taken = set()
    for path, select, digest_field, include_name in sources:
        data = _read(path)
        for row in select(data):
            if row['seed'] in taken:
                raise ValueError('target receipts repeat a point start')
            taken.add(row['seed'])
            yield (path.name, _hash(path), data.get('arena', data.get('settings')), row, digest_field, include_name)


def native_point(arena, shared, row, *, include_name):
    arena.reset(seed=row['seed'], server=-1, render=False)
    observer = Observation(shared)
    digest = sha256()
    features = []
    last = None
    for tick, command in enumerate(row['actions']):
        images = arena.render_cameras()
        if tuple(images) != ('left',):
            raise ValueError('unexpected target stream')
        if include_name:
            digest.update(b'left')
        digest.update(images['left'])
        features.append(observer.update(images['left'], last))
        _, reward, ended, capped, _ = arena.step(command, render=False)
        if bool(ended or capped) != (tick == len(row['actions']) - 1):
            raise ValueError('the archived point ended on a different frame')
        last = command
    if (arena.point_result() != row.get('outcome', row.get('result'))
            or json.dumps(arena.events) != json.dumps(row['events'])):
        raise ValueError(f'physical replay changed for TT point {row["seed"]}')
    return (np.asarray(features, np.float32), float(reward)), digest.hexdigest()


def tt(shared):
    target, metadata = [], []
    arena = settings = None
    try:
        for filename, digest, station, row, rgb_key, include_name in archived():
            if settings != station:
                if arena is not None:
                    arena.close()
                settings = station
                arena = TT20Arena(**settings, view_mode='left')
            episode, rgb = native_point(arena, shared, row, include_name=include_name)
            if rgb != row[rgb_key]:
                raise ValueError(f'native RGB differs from archive at TT point {row["seed"]}')
            source_reward = row.get('reward', row.get('outcome', row.get('result'))['points_for']
                                    - row.get('outcome', row.get('result'))['points_against'])
            if episode[1] != source_reward:
                raise ValueError('scalar target reward changed')
            target.append(episode)
            metadata.append(dict(seed=row['seed'], reward=episode[1], frames=len(episode[0]), rgb_sha256=rgb,
                                 source=filename, source_sha256=digest))
            print('SO-101', len(metadata), row['seed'], len(episode[0]), episode[1], flush=True)
    finally:
        if arena is not None:
            arena.close()
    return target, metadata


def _save(path, episodes):
    np.savez_compressed(path, values=np.concatenate([x for x, _ in episodes]), rewards=[r for _, r in episodes],
                        lengths=[len(x) for x, _ in episodes])


def _load(path):
    with np.load(path) as data:
        scenes = np.split(data['values'], np.cumsum(data['lengths'])[:-1])
        return list(zip(scenes, data['rewards'].tolist(), strict=True))


def data(folder, count):
    with TemporaryDirectory() as temp:
        shared = library(temp)
        pong_obs, pong_records = pong(shared, count)
        _save(folder / 'relation-pong-features.npz', pong_obs)
        _write(folder / 'relation-pong-observations.json.gz', dict(rows=pong_records, seed=SOURCE_SEED))
        target, target_records = tt(shared)
        _save(folder / 'relation-tt-features.npz', target)
        _write(folder / 'relation-tt-observations.json.gz', dict(rows=target_records, native_replays=len(target)))
    return pong_obs, target


def fit_data(points, *, terminal_image=True):
    return tuple(np.concatenate(vals) for vals in zip(
        *(targets(ep, terminal_image=terminal_image)[:3] for ep in points), strict=True))


def calculate(y, probability):
    ranks = rankdata(probability)
    n = y.sum()
    auc = (ranks[y > 0].sum() - n * (n + 1) / 2.) / (n * (len(y) - n)) if 0 < n < len(y) else None
    p = np.clip(probability, 1e-7, 1 - 1e-7)
    return dict(auc=None if auc is None else float(auc), n=len(y), positive=int(n),
                brier=float(np.mean((p-y)**2)), log_loss=float(-np.mean(y*np.log(p) + (1-y)*np.log1p(-p))))


def scores(w, points, outcome, *, terminal_image=True):
    y, probabilities, losses = [], [], []
    for episode in points:
        x, end, win, _ = targets(episode, terminal_image=terminal_image)
        target = end if outcome == 'terminal' else win
        value = predict(w, x[:, :len(w)-1])
        y.extend(target)
        probabilities.extend(value)
        losses.append(calculate(target, value)['brier'])
    return dict(calculate(np.asarray(y), np.asarray(probabilities)), mean_point_brier=float(np.mean(losses)))


def paired_brier(aligned, control, episodes, task):
    errors = []
    for episode in episodes:
        x, end, win, _ = targets(episode, terminal_image=False)
        y = end if task == 'terminal' else win
        a, b = predict(aligned, x[:, :len(aligned)-1]), predict(control, x[:, :len(control)-1])
        errors.append(np.mean((a-y)**2 - (b-y)**2))
    sample = np.asarray(errors)
    rng = np.random.default_rng(191007301)
    estimates = sample[rng.integers(len(sample), size=(5000, len(sample)))].mean(axis=1)
    return dict(mean=float(sample.mean()), interval=np.quantile(estimates, (.025, .975)).tolist())


def source_models(pong_train, pong_held):
    x, end, win = fit_data(pong_train)
    rng = np.random.default_rng(SOURCE_SHUFFLE)
    order = rng.permutation(len(x))
    mismatched = np.column_stack((x[:, :BASE], x[order, BASE:]))
    priors, metrics = {}, {}
    for name, y in (('terminal', end), ('wins', win)):
        options = {'pong': (x, y), 'score_shuffled': (x, y[order]),
                   'relation_shuffled': (mismatched, y), 'unary': (x[:, :BASE], y)}
        models = {key: learn(design, label) for key, (design, label) in options.items()}
        metrics[name] = {mode: scores(model, pong_held, name) for mode, model in models.items()}
        priors[name] = {mode: np.r_[np.zeros(BASE), model[1+BASE:]]
                        for mode, model in models.items() if mode != 'unary'}
    return priors, metrics


def transfer(episodes, destination, priors, sizes=BUDGETS):
    observations = _read(destination / 'relation-tt-observations.json.gz')['rows']
    order = np.random.default_rng(TARGET_SPLIT).permutation(len(episodes))
    practice, held = order[:max(sizes)], order[max(sizes):]
    valid = [episodes[i] for i in held]
    metrics = {}
    for size in sizes:
        models = {}
        if size:
            x, end, win = fit_data([episodes[i] for i in practice[:size]], terminal_image=False)
        for outcome in ('terminal', 'wins'):
            options = {}
            weights = {}
            for mode in MODES:
                prior = priors[outcome].get(mode)
                if not size:
                    w = np.zeros(1+len(episodes[0][0][0]) if mode != 'unary' else 1+BASE)
                    if prior is not None:
                        w[1:] = prior
                else:
                    y = end if outcome == 'terminal' else win
                    design = x[:, :BASE] if mode == 'unary' else x
                    w = learn(design, y, prior=prior)
                weights[mode] = w
                options[mode] = scores(w, valid, outcome, terminal_image=False)
            models[outcome] = dict(options, paired_brier={mode: paired_brier(weights['pong'], weights[mode], valid, outcome)
                                                           for mode in MODES if mode != 'pong'})
        metrics[str(size)] = models
    return dict(seed=TARGET_SPLIT, practice_seeds=[observations[i]['seed'] for i in practice],
                held_seeds=[observations[i]['seed'] for i in held], metrics=metrics)


def run(folder, count, cache):
    torch.set_num_threads(1)
    if cache:
        pong_obs, target = (_load(folder / f'relation-{name}-features.npz') for name in ('pong', 'tt'))
    else:
        pong_obs, target = data(folder, count)
    split = (len(pong_obs) * 3) // 4
    priors, source = source_models(pong_obs[:split], pong_obs[split:])
    report = dict(schema='tt22-relational-forecast-1', source_train=split, source_held=len(pong_obs)-split,
                  source_metrics=source, source_horizon=FEATURE_HORIZON, source_shuffle_seed=SOURCE_SHUFFLE,
                  prior={name: {key: value.tolist() for key, value in rows.items()} for name, rows in priors.items()},
                  target=transfer(target, folder, priors), implementation=_hash(Path(__file__).with_name('tt22_relation.py')))
    _write(folder / 'relation-forecast.json.gz', report)
    print(json.dumps(dict(source=source, target=report['target']['metrics']), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--pong-starts', type=int, default=144)
    parser.add_argument('--cache', action='store_true')
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    run(args.folder, args.pong_starts, args.cache)
