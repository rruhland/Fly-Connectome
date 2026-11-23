"""Compare identically budgeted 3D reward learning from native Pong motion and controls."""

import argparse
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_pong_world import CUT, OUTPUT
from .tt24_world_prior import WEIGHTS
from .tt24_world_prior_train import SOURCE, Worlds, data, save, utility


PLAN = OUTPUT / 'pong-world-plan.json'
RECEIPT = OUTPUT / 'pong-world-learning.json.gz'
MODES = ('pong', 'cold', 'shuffled', 'matched_cold')
BUDGETS = (0, 1, 2, 4, 8, 16)
EARLY = (1, 2, 4, 8)
FROZEN = 8
COHORTS = 24
DIRECTIONS, ELITE, INSTANCES, STRESS = 16, 6, 12, 3
FRAMES = 300
HELD = 128
VARIANCE_SCALE = np.r_[np.full(10, 1.25), np.full(16, .67), np.full(16, 1.)]
SIGMA = .17
RATE = .095
SCHEDULE = 6735971436
BOOTSTRAP_DRAWS = 8192
METRICS = ('mean_rebounds', 'mean_consecutive', 'at_least_ten', 'mean_time_s')


def certify(init):
    if init['held_moving_cues'] < 50:
        raise ValueError('too little actual Pong motion for a transferable initialization')
    pang = init['held']['pong']
    if pang['rms_moving'] >= init['held']['shuffled']['rms_moving']:
        raise ValueError('Pong failed the held native pose prediction against shuffled outcomes')
    observed = np.array([init['initial'][mode] for mode in MODES])
    if np.any(observed[:, :10] != 0.) or np.any(observed[:, 22:] != 0.) or observed.shape[1] != WEIGHTS:
        raise ValueError('only the three observed tangential position terms may be initialized')
    return observed


def register(path=PLAN, schedule=SCHEDULE):
    rng = np.random.default_rng(schedule)
    cohorts = [int(n) for n in rng.integers(1, 2**63, COHORTS, dtype=np.uint64)]
    plan = dict(schema='tt24-pong-world-schedule-1', schedule=schedule, arms=list(MODES), cohorts=cohorts,
                frozen_cohort=cohorts[0], budgets=list(BUDGETS), early=list(EARLY), frozen_at=FROZEN,
                primary='paired mean consecutive rebounds averaged over held profile 0 and 2 at updates 1, 2, 4, 8',
                directions=DIRECTIONS, elite=ELITE, worlds_per_direction=INSTANCES, stress_worlds=STRESS,
                held_per_profile_per_cohort=HELD, frames=FRAMES, sigma=SIGMA, rate=RATE, physics_sha256=_hash(SOURCE),
                common_variance_scale=VARIANCE_SCALE.tolist())
    if path.exists():
        if _read(path) != plan:
            raise ValueError('the registered 3D learning protocol changed')
    else:
        _write(path, plan)
    return plan


def score(x):
    return np.column_stack((x[..., 0].mean(1), x[..., 2].mean(1), np.mean(x[..., 2] >= 10, axis=1),
                            x[..., 1].mean(1) / 60.))


def evaluation(world, initial, held):
    return {name: score(world.play(initial, worlds, profile=profile)).tolist()
            for name, profile, worlds in zip(('nominal', 'stress'), (0, 2), held, strict=True)}


def acquire(world, initial, seed):
    rng = np.random.default_rng(seed)
    held = [data(int(q), HELD) for q in rng.integers(1, 2**63, 2, dtype=np.uint64)]
    weights = initial.copy()
    reports = {'0': evaluation(world, weights, held)}
    exposure = np.zeros((len(MODES), 2), dtype=np.int64)
    archive, learned = [], None
    for index in range(1, BUDGETS[-1] + 1):
        direction = rng.normal(size=(DIRECTIONS, WEIGHTS)) * VARIANCE_SCALE
        perturbation = SIGMA * np.concatenate((direction, -direction))
        population = (weights[:, None] + perturbation).reshape(-1, WEIGHTS)
        cases = rng.integers(1, 2**63, INSTANCES, dtype=np.uint64)
        base = world.play(population, cases[:-STRESS])
        varied = world.play(population, cases[-STRESS:], profile=1)
        results = np.concatenate((base, varied), axis=1).reshape(len(MODES), 2 * DIRECTIONS, INSTANCES, 6)
        updates = []
        for i, batch in enumerate(results):
            positive, negative = utility(batch, FRAMES).reshape(2, DIRECTIONS)
            best = np.argsort(-np.maximum(positive, negative), kind='stable')[:ELITE]
            denominator = max(.15, np.std(np.r_[positive[best], negative[best]]))
            advantage = (positive[best] - negative[best]) / denominator
            delta = np.mean(advantage[:, None] * direction[best], axis=0)
            weights[i] = np.clip(weights[i] + RATE * delta, -5.5, 5.5)
            exposure[i] += np.array((len(batch) * INSTANCES, batch[..., 1].sum()), dtype=np.int64)
            updates.append(float(np.linalg.norm(delta)))
        archive.append(dict(launches=cases.tolist(), gradient_norms=updates, exposure=exposure.tolist()))
        if index in BUDGETS:
            reports[str(index)] = evaluation(world, weights, held)
        if index == FROZEN:
            learned = weights.copy()
    hold_set = {int(q) for group in held for q in group}
    if len(hold_set) != 2 * HELD or hold_set & {q for row in archive for q in row['launches']}:
        raise ValueError('held and fitting worlds overlap')
    return dict(seed=seed, held=[h.tolist() for h in held], reports=reports, updates=archive, deployed=learned.tolist())


def inference(cohorts):
    n = len(cohorts)
    values = np.array([[np.array(list(r['reports'][str(b)].values())) for b in BUDGETS] for r in cohorts])
    # Cohort, budget, profile, arm, metric. All control comparisons preserve pairing.
    profiles = {name: x for name, x in zip(('nominal', 'stress', 'mixed'),
                                          (values[:, :, 0], values[:, :, 1], values.mean(axis=2)), strict=True)}
    rng = np.random.default_rng(1918205702)
    resample = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, n))
    comparison = {}
    early = [BUDGETS.index(update) for update in EARLY]
    for name, field in profiles.items():
        rows = {mode: field[:, :, j, :].mean(axis=0).tolist() for j, mode in enumerate(MODES)}
        differences = {}
        for j, mode in enumerate(MODES[1:], 1):
            advantage = field[:, :, 0, :] - field[:, :, j, :]
            total = advantage[:, early, 1].mean(axis=1)
            differences[mode] = dict(by_budget=advantage.mean(axis=0).tolist(),
                                     consecutive_bootstrap95=np.quantile(advantage[:, :, 1][resample].mean(axis=1),
                                                                         (.025, .975), axis=0).T.tolist(),
                                     early_mean_consecutive=float(total.mean()),
                                     early_bootstrap95=np.quantile(total[resample].mean(axis=1), (.025, .975)).tolist(),
                                     early_better_cohorts=int((total > 0).sum()))
        comparison[name] = dict(arms=rows, pong_minus=differences)
    return comparison


def run(workers, *, plan_path=PLAN, init_path=CUT, output=RECEIPT, schedule=SCHEDULE, prefix='pong-world'):
    began = perf_counter()
    plan = register(plan_path, schedule)
    init = _read(init_path)
    initial = certify(init)
    metadata = dict(plan_sha256=_hash(plan_path), init_sha256=_hash(init_path), modes=list(MODES), metrics=list(METRICS))
    report = _read(output) if output.exists() else dict(metadata=metadata, cohorts=[])
    if report['metadata'] != metadata:
        raise ValueError('saved learning curve uses other frozen motion or physics')
    world = Worlds(workers=workers, frames=FRAMES)
    try:
        for i, seed in enumerate(plan['cohorts']):
            if i < len(report['cohorts']):
                continue
            row = acquire(world, initial, seed)
            if i == 0:
                for mode, coefficients in zip(MODES, row['deployed'], strict=True):
                    save(OUTPUT / f'{prefix}-frozen-{mode}.npz', coefficients)
            report['cohorts'].append(row)
            _write(output, report)
            print('matched 3D acquisition', i + 1, 'of', COHORTS, round(perf_counter() - began, 1), flush=True)
    finally:
        world.close()
    report.update(summary=inference(report['cohorts']), low_budget_policy_sha256={
        mode: _hash(OUTPUT / f'{prefix}-frozen-{mode}.npz') for mode in MODES}, seconds=perf_counter() - began)
    _write(output, report)
    for name, profile in report['summary'].items():
        print(name, {label: (row['early_mean_consecutive'], row['early_bootstrap95'])
                     for label, row in profile['pong_minus'].items()}, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.plan_only:
        register()
    else:
        run(args.workers)


if __name__ == '__main__':
    main()
