"""Compare carried, fresh and frozen visual motor models on paired reaching courses."""
import argparse
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import monotonic

import numpy as np

from body_effect_learner import BodyLearner
from body_reaching_arena import ReachingArena, sample_case


CONDITIONS = (
    ('carriage', 'arm'),
    ('carriage', 'arm_remapped'),
    ('arm', 'arm_variant'),
    ('arm', 'arm_remapped'),
)
MAX_STEPS = 96


def rng_for(seed, *indices):
    return np.random.default_rng(np.random.SeedSequence([seed, *indices]))


def reach(learner, body, case, *, save_path=False):
    arena = ReachingArena(body, case, max_steps=MAX_STEPS)
    position, target = arena.observation()
    actions = sha256()
    errors = [arena.error]
    learner.reset_episode()
    while not arena.done:
        action = learner.act(position, target)
        next_position, target = arena.step(action)
        learner.observe(position, action, next_position)
        position = next_position
        actions.update(np.asarray(action, dtype='<f8').tobytes())
        errors.append(arena.error)
    row = dict(case=asdict(case), success=bool(arena.success), commands=int(arena.steps),
               error_fraction_at_8=float(errors[min(8, len(errors)-1)] / errors[0]),
               final_error=float(errors[-1]), actions_sha256=actions.hexdigest())
    if save_path:
        row['trajectory'] = np.asarray(arena.trace).tolist()
    return row


def bootstrap(values):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(120799)
    draws = rng.choice(values, size=(4000, len(values)), replace=True).mean(axis=1)
    return np.quantile(draws, [.025, .975]).tolist()


def summarize(streams):
    out = {}
    for src, dst in CONDITIONS:
        key = f'{src}_to_{dst}'
        scenarios = [s['conditions'][key] for s in streams]
        metrics = {}
        for arm in ('carry', 'fresh', 'frozen'):
            firsts = [s[arm][0] for s in scenarios]
            later = [r for s in scenarios for r in s[arm][1:]]
            metrics[arm] = {}
            for phase, rows in [('first', firsts), ('later', later)]:
                metrics[arm][phase] = dict(reaches=len(rows), successes=sum(r['success'] for r in rows),
                    mean_capped_commands=float(np.mean([r['commands'] for r in rows])),
                    mean_error_fraction_at_8=float(np.mean([r['error_fraction_at_8'] for r in rows])))
        gains = np.array([s['fresh'][0]['commands'] - s['carry'][0]['commands'] for s in scenarios])
        metrics['paired_first_commands_saved'] = dict(mean=float(gains.mean()), ci95=bootstrap(gains),
            ahead=int((gains > 0).sum()), tied=int((gains == 0).sum()), behind=int((gains < 0).sum()))
        out[key] = metrics
    return out


def run(seed, n_streams, n_practice, n_target):
    started = monotonic()
    streams = []
    for i in range(n_streams):
        source_rng = rng_for(seed, i, 1)
        target_rng = rng_for(seed, i, 2)
        practice = [sample_case(source_rng) for _ in range(n_practice)]
        test = [sample_case(target_rng) for _ in range(n_target)]
        policy_seed = int(rng_for(seed, i, 3).integers(2**31))
        saved = {}
        stream = dict(index=i, policy_seed=policy_seed, practice={}, conditions={})
        for body in ('carriage', 'arm'):
            model = BodyLearner(seed=policy_seed)
            stream['practice'][body] = [reach(model, body, case) for case in practice]
            saved[body] = model.export_prior()
        stream['source_checkpoints'] = saved
        for source, target in CONDITIONS:
            course = {}
            for arm in ('carry', 'fresh', 'frozen'):
                model = BodyLearner(seed=policy_seed, prior=saved[source] if arm != 'fresh' else None,
                                    updates_enabled=arm != 'frozen')
                course[arm] = [reach(model, target, case, save_path=(i == 0 and j == 0))
                               for j, case in enumerate(test)]
                if arm == 'frozen' and model.export_prior() != saved[source]:
                    raise AssertionError('frozen coefficients changed')
            stream['conditions'][f'{source}_to_{target}'] = course
        streams.append(stream)
    path = Path(__file__).parent
    source = {n: sha256((path/n).read_bytes()).hexdigest() for n in
              ('body_effect_learner.py', 'body_reaching_arena.py', 'run_body_transfer.py')}
    return dict(schema='body-effect-transfer-v1', source_sha256=source, seed=seed, practice=n_practice,
                target=n_target, max_steps=MAX_STEPS, wall_seconds=monotonic()-started,
                summary=summarize(streams), streams=streams)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--streams', type=int, default=24)
    parser.add_argument('--practice', type=int, default=24)
    parser.add_argument('--target', type=int, default=4)
    args = parser.parse_args()
    if min(args.streams, args.practice, args.target - 1) < 1:
        parser.error('use at least one stream and practice reach and two target reaches')
    data = run(args.seed, args.streams, args.practice, args.target)
    with args.output.open('x') as out:
        json.dump(data, out, separators=(',', ':'), allow_nan=False)
    for case, metrics in data['summary'].items():
        print(case, metrics['paired_first_commands_saved'])
