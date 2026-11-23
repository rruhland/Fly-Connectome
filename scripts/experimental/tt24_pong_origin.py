"""Learn one common visual and motor origin weight from actual native Pong."""

import argparse
from time import perf_counter

import numpy as np
import torch

from fly_connectome.pong import Physics, Pong

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_donor_trial import ROOT
from .tt24_joint_pong import CAP, HEIGHT, PLAN as PARENT_PLAN, ROWS, STRIDE, WARM, WIDTH, field


TAG = 'tt24_pong_origin'
PLAN = ROOT / f'{TAG}-plan.json'
LOCK = ROOT / f'{TAG}-lock.json.gz'
OUTPUT = ROOT / f'{TAG}.json.gz'
BANK_PLAN = ROOT / f'{TAG}-bank-plan.json'
BANK = ROOT / f'{TAG}-bank.json.gz'
BANK_HELD = ROOT / f'{TAG}-bank-held.json.gz'
CONFIRM_PLAN = ROOT / f'{TAG}-confirmation-plan.json'
CONFIRM_BANK = ROOT / f'{TAG}-confirmation-bank.json.gz'
CONFIRM_HELD = ROOT / f'{TAG}-confirmation-held.json.gz'
SEED = 1819039097
SHUFFLE = 1586409229
BANK_SEED = 508081255
BANK_STREAMS = 6
CONFIRM_SEED = 1388239721
CONFIRM_STREAMS = 12
GRID = np.linspace(0., 1., 11)
COURSES = {'practice': 96, 'shifted': 64, 'ordinary': 48}


def named_roster(rng, count, displaced):
    launches = (rng.choice(2**31 - 2, count, replace=False) + 1).tolist()
    side = rng.permutation(np.tile([-1., 1.], count // 2))
    offsets = side * rng.uniform(.17, .33, count) if displaced else np.zeros(count)
    return dict(seeds=launches, physical_offsets=offsets.tolist())


def register():
    rng = np.random.default_rng(SEED)
    used = set()
    records = {}
    for name, count in COURSES.items():
        records[name] = named_roster(rng, count, name != 'ordinary')
        launches = records[name]['seeds']
        if used.intersection(launches):
            raise ValueError('native origin course seeds overlap')
        used.update(launches)
    old = _read(PARENT_PLAN)
    prior = {q for group in old['warm_train'] for q in group} | set(old['warm_held'])
    prior.update(q for row in old['streams'] for q in row['held_native'])
    prior.update(q for row in old['streams'] for group in row['native'] for q in group)
    if used & prior:
        raise ValueError('native origin course contains an earlier actor example')
    plan = dict(schema='tt24-pong-origin-1', seed=SEED, shuffled_seed=SHUFFLE, rosters=records,
                parents={p.name: _hash(p) for p in (WARM, PARENT_PLAN)}, practice_alphas=GRID.tolist(),
                practice_payoff='native terminal score + .43 times active duration fraction',
                fitting='continuous regularized quadratic argmax on [0,1]', native_cap_frames=CAP,
                projection='current optical launch origin + alpha * (familiar optical origin - current optical launch origin)',
                retention='frozen source actor on shifted and zero-offset actual native serves')
    if PLAN.exists():
        if _read(PLAN) != plan:
            raise ValueError('registered native origin protocol changed')
    else:
        _write(PLAN, plan)
    return plan


def familiar_home():
    prior = _read(PARENT_PLAN)
    anchors = [seed for group in prior['warm_train'] for seed in group]
    frame = Pong(anchors).render(HEIGHT, WIDTH).numpy()
    blade = frame[:, :, 1]
    x = (blade @ ROWS) / blade.sum(axis=1)
    return float(np.median(x)), x.tolist()


@torch.no_grad()
def points(weights, alphas, roster, familiar):
    """Run real native points with the one altered initial condition in physics."""
    seeds, offsets = roster['seeds'], roster['physical_offsets']
    count = len(seeds) * len(alphas)
    blend = np.repeat(np.asarray(alphas, dtype=float), len(seeds))
    game = Pong(seeds * len(alphas), Physics(opponent_speed=.45))
    # Actuate the physical launch displacement on the environment side only.
    game.body.position.add_(torch.tensor(offsets * len(alphas), dtype=game.body.position.dtype))
    live = np.ones(count, dtype=bool)
    old = np.zeros((count, 2))
    velocity = np.zeros_like(old)
    previous_visibility = np.zeros(count, dtype=bool)
    score, hits, age, action, paddle_speed = [np.zeros(count) for _ in range(5)]
    for step in range(CAP):
        if step % STRIDE == 0:
            paddle, seen = field(game.render(HEIGHT, WIDTH).numpy(), old, velocity, previous_visibility)
            if step == 0:
                observed_home = paddle.copy()
                origin = observed_home + blend * (familiar - observed_home)
                previous_paddle = observed_home.copy()
            gap = np.clip((old[:, 1] - paddle) * .76 / .065, -3., 3.)
            tangent = np.clip(velocity[:, 1] * 3. / .38, -3., 3.)
            pose = np.clip((paddle - origin) * .76 / .04, -3., 3.)
            normal = np.clip(velocity[:, 0] * 3. / 1.5, -3., 3.)
            features = np.column_stack((gap, tangent, pose, np.clip(gap * normal, -3., 3.)))
            offset = .36 * np.tanh((features * weights).sum(axis=1))
            paddle_speed = .70 * paddle_speed + .30 * (paddle - previous_paddle) * (120 / STRIDE)
            request = np.clip(-11. * (origin + offset - paddle) + 1.35 * paddle_speed, -1., 1.)
            action[seen] = request[seen]
            previous_paddle = paddle
        result = game.step(torch.from_numpy(action.astype(np.float32)), auto_reset=False)
        scored = result.scores.numpy()
        age += live
        hits += result.hits.numpy() * live
        score += scored * live
        live &= scored == 0.
        if not live.any():
            break
    result = np.column_stack((score, hits, age)).reshape(len(alphas), len(seeds), 3)
    return result, observed_home.reshape(len(alphas), len(seeds))


def fit(payoffs):
    x = GRID - .5
    design = np.column_stack((np.ones(len(x)), x, x**2))
    coefficients = np.linalg.solve(design.T @ design + np.diag([0., .025, .025]), design.T @ payoffs.mean(axis=1))
    if coefficients[2] < 0.:
        alpha = np.clip(.5 - .5 * coefficients[1] / coefficients[2], 0., 1.)
    else:
        alpha = float(coefficients[1] > 0.) if coefficients[1] else .5
    return dict(alpha=float(alpha), quadratic_centered=coefficients.tolist(), real_mean_by_grid=payoffs.mean(axis=1).tolist())


def credit(practice, shuffle_seed):
    payoff = practice[..., 0] + .43 * practice[..., 2] / CAP
    rng = np.random.default_rng(shuffle_seed)
    order = np.stack([rng.permutation(len(GRID)) for _ in range(payoff.shape[1])], axis=1)
    shuffled = np.take_along_axis(payoff, order, axis=0)
    learning = {'native': fit(payoff), 'shuffled': fit(shuffled)}
    values = dict(native=learning['native']['alpha'], shuffled=learning['shuffled']['alpha'], neutral=.5, current=0., familiar=1.)
    return values, learning


def lock(plan):
    if LOCK.exists():
        saved = _read(LOCK)
        if saved['plan'] != _hash(PLAN):
            raise ValueError('old alpha differs from the registered source')
        return saved
    familiar, earlier = familiar_home()
    weights = np.array(_read(WARM)['weights'])
    practice, first = points(weights, GRID, plan['rosters']['practice'], familiar)
    values, learning = credit(practice, SHUFFLE)
    result = dict(plan=_hash(PLAN), familiar_optical_home=familiar, earlier_optical_starts=earlier,
                  selected=values, source_fit=learning, practice=practice.tolist(), optical_launch=first[0].tolist(),
                  stored_weights_parent=_hash(WARM), native_practice_points=int(np.prod(practice.shape[:2])),
                  native_practice_frames=int(practice[..., 2].sum()), arm_labels_or_rewards=0)
    _write(LOCK, result)
    print('LOCKED NATIVE ORIGIN', SEED, _hash(LOCK), values, 'curve', learning['native']['real_mean_by_grid'], flush=True)
    return result


def outcomes(values):
    score, hits, length = values.T
    return dict(points=len(score), returned=int((hits > 0.).sum()), later=int(np.maximum(hits - 1., 0.).sum()),
                own_wins=int(((hits > 0.) & (score > 0.)).sum()), losses=int((score < 0.).sum()),
                frames=int(length.sum()), primary=float(np.mean(score + .43 * length / CAP)))


def course_outcomes(weights, modes, roster, familiar, ordinary):
    actual, starts = points(weights, list(modes.values()), roster, familiar)
    if ordinary and not np.array_equal(actual, np.repeat(actual[0:1], len(actual), axis=0)):
        raise ValueError('unshifted native pool must retain the original policy for every alpha')
    return dict(outcomes={name: outcomes(row) for name, row in zip(modes, actual, strict=True)},
                raw={name: row.tolist() for name, row in zip(modes, actual, strict=True)}, optical_launch=starts[0].tolist())


def held(plan, source, began):
    weights = np.array(_read(WARM)['weights'])
    modes = source['selected']
    result = dict(plan=_hash(PLAN), source_alpha_lock=_hash(LOCK), held={}, wall_seconds=None)
    for course in ('shifted', 'ordinary'):
        end = course_outcomes(weights, modes, plan['rosters'][course], source['familiar_optical_home'], course == 'ordinary')
        result['held'][course] = end
        print(course, end['outcomes'], flush=True)
    result['wall_seconds'] = perf_counter() - began
    _write(OUTPUT, result)


def bank_registration(confirm=False):
    prior = _read(PARENT_PLAN)
    previous = _read(PLAN)
    destination = CONFIRM_PLAN if confirm else BANK_PLAN
    seed, streams = (CONFIRM_SEED, CONFIRM_STREAMS) if confirm else (BANK_SEED, BANK_STREAMS)
    used = {k for row in previous['rosters'].values() for k in row['seeds']}
    used.update(prior['warm_held'])
    used.update(k for group in prior['warm_train'] for k in group)
    used.update(k for trace in prior['streams'] for k in trace['held_native'])
    used.update(k for trace in prior['streams'] for group in trace['native'] for k in group)
    if confirm:
        previous_bank = _read(BANK_PLAN)
        previous_lock = _read(BANK)
        if (previous_lock['plan'] != _hash(BANK_PLAN) or previous_lock['earlier_lock'] != _hash(LOCK)
                or len(previous_lock['replicas']) != BANK_STREAMS):
            raise ValueError('earlier source ties and replication bank must be frozen first')
        used.update(k for group in previous_bank['replicas'] for row in group['rosters'].values() for k in row['seeds'])
    rng = np.random.default_rng(seed)
    groups = []
    for index in range(streams):
        practice_seed, shuffle_seed = (int(q) for q in rng.integers(1, 2**31 - 1, size=2))
        replica = np.random.default_rng(practice_seed)
        rosters = {name: named_roster(replica, count, name != 'ordinary') for name, count in COURSES.items()}
        current = [q for row in rosters.values() for q in row['seeds']]
        if len(current) != len(set(current)) or used.intersection(current):
            raise ValueError('bank replica contains an older native start')
        used.update(current)
        groups.append(dict(index=index, seed=practice_seed, shuffle_seed=shuffle_seed, rosters=rosters))
    bank = dict(schema='tt24-pong-origin-bank-plan-1', seed=seed, first_alpha_lock=_hash(LOCK),
                source_plan=_hash(PLAN), warm_policy=_hash(WARM), replicas=groups,
                practice_alphas=GRID.tolist(), source_objective=previous['practice_payoff'], arm_exposure=0)
    if confirm:
        bank['preceding_bank'] = dict(plan_sha256=_hash(BANK_PLAN), checkpoint_sha256=_hash(BANK))
    if destination.exists():
        if _read(destination) != bank:
            raise ValueError('native origin replication schedule changed')
    else:
        _write(destination, bank)
    return bank


def replication(source, held_too, *, confirm=False):
    plan = bank_registration(confirm)
    path_plan, path_bank, path_held = (CONFIRM_PLAN, CONFIRM_BANK, CONFIRM_HELD) if confirm else (BANK_PLAN, BANK, BANK_HELD)
    plan_hash = _hash(path_plan)
    weights = np.array(_read(WARM)['weights'])
    familiar = source['familiar_optical_home']
    if path_bank.exists():
        receipt = _read(path_bank)
        if receipt['plan'] != plan_hash or len(receipt['replicas']) != len(plan['replicas']):
            raise ValueError('incomplete or wrong alpha lock bank')
    else:
        replicas = []
        for spec in plan['replicas']:
            data, origin = points(weights, GRID, spec['rosters']['practice'], familiar)
            selected, fitted = credit(data, spec['shuffle_seed'])
            replicas.append(dict(index=spec['index'], seed=spec['seed'], shuffle_seed=spec['shuffle_seed'],
                                 selected=selected, fitted=fitted, optical_launch=origin[0].tolist(),
                                 practice=data.tolist(), points=int(np.prod(data.shape[:2])), frames=int(data[..., 2].sum())))
            print('native replication complete', spec['index'], flush=True)
        receipt = dict(plan=plan_hash, warm=_hash(WARM), earlier_lock=_hash(LOCK), arm_exposure=0,
                       familiar_optical_home=familiar, replicas=replicas)
        _write(path_bank, receipt)
    print('LOCKED COMPLETE BANK', _hash(path_bank), [rep['selected'] for rep in receipt['replicas']], flush=True)
    if held_too:
        outcomes_by_replica = []
        for row, spec in zip(receipt['replicas'], plan['replicas'], strict=True):
            metric = {name: course_outcomes(weights, row['selected'], spec['rosters'][name], familiar, name == 'ordinary')
                      for name in ('shifted', 'ordinary')}
            outcomes_by_replica.append(dict(index=spec['index'], held=metric))
        _write(path_held, dict(plan=plan_hash, frozen=_hash(path_bank), replicas=outcomes_by_replica))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock-only', action='store_true')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--replicate', action='store_true')
    group.add_argument('--confirm', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    start = perf_counter()
    plan = register()
    source = lock(plan)
    print('source coefficients', source['selected'], flush=True)
    if args.replicate or args.confirm:
        replication(source, not args.lock_only, confirm=args.confirm)
    elif not args.lock_only:
        held(plan, source, start)


if __name__ == '__main__':
    main()
