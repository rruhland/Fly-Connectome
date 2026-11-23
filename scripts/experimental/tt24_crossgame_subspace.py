"""Transfer reward-relevant overhead feedback directions into wrist-only solo search."""

import argparse
import ctypes
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_donor_trial import DESIGNS, ROOT, SHAPE, Native, _seeds, _summary
from .tt24_overhead import PartnerWorld
from .tt24_overhead_audit import replay as replay_overhead
from .tt24_public_body import FUSED_OUTPUT, load_public


ARCHIVE = ROOT / 'overhead-source-training.json.gz'
SOURCE_HEAD = ROOT / 'overhead-source.npz'
SOURCE_HELD = ROOT / 'overhead-source-held.json.gz'
OUTPUT = ROOT / 'crossgame-subspace.json.gz'
CHECKPOINT = ROOT / 'crossgame-subspace.npz'
RANK = 6
STREAMS = 3
ROUNDS = (.50, .32)
LEARNERS = ('subspace', 'reward_shuffle', 'joint_shuffle', 'unstructured')
BASELINES = ('source_head', 'hold')


def read(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


def write(path, document):
    pending = path.with_name(path.name + '.tmp')
    with gzip.open(pending, 'wt') as stream:
        json.dump(document, stream, separators=(',', ':'))
    pending.replace(path)


def proposals(archive):
    directions, scores = [], []
    for generation in archive['updates']:
        weights = np.asarray(generation['weights'])[1:]
        rewards = np.asarray(generation['training_results'])[:, 1:, :, 0].mean(axis=(0, 2))
        plus, minus = np.split(weights.reshape(len(weights), -1), 2)
        direction = .5 * (plus - minus)
        direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1e-12)
        directions.append(direction)
        scores.append(rewards - rewards.mean())
    return np.asarray(directions), np.asarray(scores)


def paired_gradient(directions, scores):
    plus, minus = np.split(scores, 2, axis=1)
    scale = np.maximum(.35, scores.std(axis=1))
    return np.einsum('gk,gkd->gd', (plus - minus) / scale[:, None], directions)


def signal_basis(directions, scores, null):
    gradients = paired_gradient(directions, scores)
    covariance = gradients.T @ gradients / len(gradients)
    eigenvalues, vectors = np.linalg.eigh(covariance - null)
    return vectors[:, -RANK:][:, ::-1].T.copy(), eigenvalues[-RANK:][::-1].copy()


def source_bases(archive):
    directions, scores = proposals(archive)
    dimension = directions.shape[-1]
    rng = np.random.default_rng(39897021)
    null = np.zeros((dimension, dimension))
    for _ in range(64):
        draw = np.asarray([rng.permutation(row) for row in scores])
        gradients = paired_gradient(directions, draw)
        null += gradients.T @ gradients / (64 * len(gradients))
    basis, eigenvalues = signal_basis(directions, scores, null)
    choices = []
    for _ in range(STREAMS):
        shuffled_rewards = np.asarray([rng.permutation(row) for row in scores])
        shuffled_basis, shuffled_values = signal_basis(directions, shuffled_rewards, null)
        permutation = np.roll(rng.permutation(SHAPE[0]), 1)
        while (permutation == np.arange(SHAPE[0])).any():
            permutation = rng.permutation(SHAPE[0])
        joint_basis = basis.reshape(RANK, *SHAPE)[:, permutation].reshape(RANK, dimension)
        isotropic, _ = np.linalg.qr(rng.normal(size=(dimension, RANK)))
        conditions = dict(zip(LEARNERS, (basis, shuffled_basis, joint_basis, isotropic.T), strict=True))
        choices.append((conditions, permutation.tolist(), shuffled_values.tolist()))
    scored_drops = sum(np.asarray(row['training_results'])[:, 1:, :, 0].size for row in archive['updates'])
    return choices, dict(generations=len(scores), antithetic_pairs=int(directions.shape[0] * directions.shape[1]),
                         scored_drops=int(scored_drops),
                         eigenvalues=eigenvalues.tolist(), reward_shuffle_null_draws=64)


def actions(arena, native, weights, seed, split, regime, *, tape=None, judge=False):
    # The policy and reward search only consume the public return from reset/step.
    render = tape is None and bool(np.count_nonzero(weights))
    obs = arena.reset(seed=seed, split=split, regime=regime, render=render)
    observer = load_public(FUSED_OUTPUT) if render else None
    weights = np.ascontiguousarray(weights)
    issued, action = np.zeros(5), np.zeros(5)
    record = []
    reward = 0.
    for tick in range(arena.max_steps):
        if observer is not None:
            state = np.ascontiguousarray(observer.observe(obs))
            native.command(state.ctypes.data, weights.ctypes.data, issued.ctypes.data, action.ctypes.data)
        elif tape is not None:
            action = np.asarray(tape[tick])
        obs, increment, done = arena.step(action, render=render)
        record.append(action.tolist())
        reward += increment
        if tape is not None and bool(done) != (tick + 1 == len(tape)):
            raise ValueError(f'solo tape ended on a different frame for seed {seed}')
        if done:
            break
    result = dict(reward=reward, actions=record)
    if judge:
        result['metrics'] = arena.judge_metrics()
    return result


def train(arena, native, basis, starts, latent, residual):
    zero = np.zeros(SHAPE)
    prior_candidates = [zero]
    rounds = []
    for index, scale in enumerate(ROUNDS):
        centre = prior_candidates[-1]
        directions = latent[index] @ basis
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        directions += .15 * residual[index]
        directions *= scale / np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions.reshape(-1, *SHAPE)
        trial = [centre, *(centre + directions), *(centre - directions)]
        grades = [actions(arena, native, weights, starts[index], 'practice', 'standard')['reward']
                  for weights in trial]
        winner = int(np.argmax(grades))
        prior_candidates.append(trial[winner])
        rounds.append(dict(seed=starts[index], parameter_options=np.asarray(trial).tolist(),
                           score=grades, winner=winner))
    # Two new starts challenge the winners independently before any held stock launch.
    validation = [[actions(arena, native, weights, seed, 'practice', 'standard')['reward']
                   for seed in starts[len(ROUNDS):]] for weights in prior_candidates]
    winner = int(np.argmax(np.sum(validation, axis=1)))
    return np.asarray(prior_candidates[winner]), dict(rounds=rounds, validation_seeds=starts[len(ROUNDS):],
                                                       validation=validation, selected=winner)


def stock_held(arena, native, policies, seeds, regime):
    records = []
    for seed in seeds:
        conditions = {}
        for name, weights in policies.items():
            trial = actions(arena, native, weights, seed, 'held', regime, judge=True)
            replay = actions(arena, native, weights, seed, 'held', regime, tape=trial['actions'], judge=True)
            if digest_json(trial) != digest_json(replay):
                raise ValueError(f'exact solo replay failed for {name} at {seed}')
            conditions[name] = trial
        records.append(dict(seed=seed, conditions=conditions))
    return records


def retention(source_head, source_hash, archive):
    opponent = Native(Path(__file__).with_name('tt24_overhead_native.cpp'))
    trace = opponent.so.tt24_overhead_trace
    trace.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] + [ctypes.c_void_p] * 2
    trace.restype = None
    before = read(SOURCE_HELD)['source']
    old_seeds = {seed for donor in before for seed in donor['seeds']}
    probe = _seeds(479920811, len(DESIGNS), 12)
    source_training = {seed for row in archive['updates'] for group in row['training_seeds'] for seed in group}
    source_development = {seed for group in archive['dev_seeds'] for seed in group}
    if (old_seeds | source_training | source_development).intersection(np.ravel(probe)):
        raise ValueError('source retention sets overlap')
    results = []
    try:
        for index, design in enumerate(DESIGNS):
            world = PartnerWorld(opponent, design)
            try:
                options = np.stack((source_head, np.zeros(SHAPE)))
                previous = world.play(options, world.prepare(before[index]['seeds'], 'held'))
                if not np.array_equal(previous, before[index]['scores']):
                    raise ValueError(f'prior ceiling pairs changed for donor {index}')
                seeds = probe[index]
                batch = world.prepare(seeds, 'held')
                held = world.play(options, batch)
                tapes = []
                for trial, seed in enumerate(seeds):
                    state, physics = batch[0][trial], batch[1][trial]
                    tape, stats = np.zeros((300, 5)), np.zeros(4)
                    trace(world.handle, source_head.ctypes.data, state.ctypes.data, physics.ctypes.data,
                          len(tape), stats.ctypes.data, tape.ctypes.data)
                    tape = tape[:int(stats[2])]
                    if not np.array_equal(stats, held[0, trial]):
                        raise ValueError(f'native traced scores changed on donor {index} seed {seed}')
                    try:
                        replayed = replay_overhead(world.arena, seed, tape)
                        replay_error = None
                    except AssertionError as error:
                        replayed = None
                        replay_error = str(error)
                    tapes.append(dict(actions=tape.tolist(), replay_counts=replayed, exact_replay=
                                      np.array_equal(stats, replayed), replay_error=replay_error))
                results.append(dict(donor=index, prior_trials=len(previous[0]), exact_prior=True,
                                    fresh_seeds=seeds, fresh_scores=held.tolist(), trials=tapes))
            finally:
                world.close()
    finally:
        opponent.working.cleanup()
    actual_hash = sha256(SOURCE_HEAD.read_bytes()).hexdigest()
    if actual_hash != source_hash:
        raise ValueError('the original overhead head was modified')
    return dict(source_head_sha256=actual_hash, donors=results)


def recover_policies(report, checkpoint, expected):
    if len(report['streams']) != STREAMS or report['source_archive_sha256'] != expected:
        raise ValueError('resume requires three complete practice streams from this source archive')
    checkpoint_hash = sha256(checkpoint.read_bytes()).hexdigest()
    if report.get('selected_sha256_before_held', checkpoint_hash) != checkpoint_hash:
        raise ValueError('resume checkpoint changed after selection')
    policies = []
    with np.load(checkpoint, allow_pickle=False) as data:
        for index, stream in enumerate(report['streams']):
            restored = {}
            for key in LEARNERS:
                practice = stream['practice'][key]
                selected = practice['selected']
                if selected:
                    winner = practice['rounds'][selected - 1]
                    from_tape = winner['parameter_options'][winner['winner']]
                else:
                    from_tape = np.zeros(SHAPE)
                restored[key] = data[f'{index}_{key}']
                if not np.array_equal(restored[key], from_tape):
                    raise ValueError('resume checkpoint changed the selected solo weights')
            policies.append(restored)
    return policies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--resume', action='store_true', help='resume held tests from completed practice')
    args = parser.parse_args()
    archive = read(ARCHIVE)
    basis_choices, source_stats = source_bases(archive)
    head_sha = sha256(SOURCE_HEAD.read_bytes()).hexdigest()
    if archive['frozen_sha256'] != head_sha:
        raise ValueError('source head and exploration archive came from different runs')
    with np.load(SOURCE_HEAD, allow_pickle=False) as data:
        source_head = np.ascontiguousarray(data['weights'])
    training = _seeds(65846131, STREAMS, 4)
    stock = _seeds(236298137, STREAMS, 8)
    changed = _seeds(385169278, STREAMS, 4)
    all_target_seeds = [seed for cohort in (training, stock, changed) for group in cohort for seed in group]
    if len(set(all_target_seeds)) != len(all_target_seeds):
        raise ValueError('training, validation and held stock drops must be disjoint')
    native = Native()
    arena = BounceArena(view='wrist', width=192, height=144)
    report = dict(source=source_stats, practice_drops_per_condition_stream=20,
                  target_observations='wrist RGB, robot encoders and scalar rebound reward',
                  source_geometry='three physical hinged donors with alternation against a ceiling racket',
                  source_archive_sha256=sha256(ARCHIVE.read_bytes()).hexdigest(),
                  initial_weights='zero for every candidate search', streams=[])
    trained = []
    try:
        if args.resume:
            expected = report['source_archive_sha256']
            report = read(args.output)
            trained = recover_policies(report, args.checkpoint, expected)
        else:
            for index, (choices, permutation, shuffle_eigenvalues) in enumerate(basis_choices):
                rng = np.random.default_rng(6201741 + index)
                latent = rng.normal(size=(len(ROUNDS), 3, RANK))
                residual = rng.normal(size=(len(ROUNDS), 3, int(np.prod(SHAPE))))
                residual /= np.linalg.norm(residual, axis=2, keepdims=True)
                policies, search = {}, {}
                for name, basis in choices.items():
                    policies[name], search[name] = train(arena, native, basis, training[index], latent, residual)
                    print('solo practice', index + 1, name, search[name]['selected'],
                          search[name]['validation'], flush=True)
                trained.append(policies)
                report['streams'].append(dict(index=index, reward_shuffle_eigenvalues=shuffle_eigenvalues,
                                              joint_permutation=permutation, practice=search))
                write(args.output, report)
            np.savez_compressed(args.checkpoint, **{f'{index}_{key}': value
                                                   for index, stream in enumerate(trained)
                                                   for key, value in stream.items()},
                                **{f'{index}_{key}_basis': value
                                   for index, (basis, _, _) in enumerate(basis_choices)
                                   for key, value in basis.items()})
        report['selected_sha256_before_held'] = sha256(args.checkpoint.read_bytes()).hexdigest()
        write(args.output, report)
        for index, policies in enumerate(trained):
            policies.update(zip(BASELINES, (source_head, np.zeros(SHAPE)), strict=True))
            for label, seeds, regime in (('standard', stock[index], 'standard'),
                                         ('changed', changed[index], 'changed')):
                if label in report['streams'][index]:
                    continue
                records = stock_held(arena, native, policies, seeds, regime)
                report['streams'][index][label] = records
                print('held wrist', index + 1, label,
                      {key: row['mean'] for key, row in _summary(records).items()}, flush=True)
                write(args.output, report)
        report['overhead_retention'] = retention(source_head, head_sha, archive)
        report['summary'] = {regime: _summary([record for stream in report['streams']
                                               for record in stream[regime]])
                             for regime in ('standard', 'changed')}
        report['verified_solo_tapes'] = sum(row['drops'] for tables in report['summary'].values()
                                            for row in tables.values())
        report['verified_fresh_source_tapes'] = sum(trial['exact_replay']
                                                     for donor in report['overhead_retention']['donors']
                                                     for trial in donor['trials'])
        write(args.output, report)
    finally:
        arena.close()
        native.working.cleanup()


if __name__ == '__main__':
    main()
