"""Learn a local motor-to-achieved-wrist-outcome map from sixteen self trials."""

from collections import deque
from hashlib import sha256
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt24_bounce import BounceArena
from .tt24_bounce_trial import verify_replay
from .tt24_hindsight import OUTPUT, RESULT as NATIVE_AUDIT
from .tt24_motor import MotorPolicy, optical_sightings
from .tt24_recurrence_arm import EMBODIMENT, target_effects


RECEIPT = OUTPUT / 'hindsight-arm.json.gz'
SEED = 1484803258
TRAIN = 8
HELD = 16
PULSE = 11
OBSERVATION = (16, 25)
MODES = ('hindsight', 'reward', 'warm')


def optical_state(rgb):
    ball, blade = optical_sightings(rgb)
    if ball is None or blade is None:
        return None
    offset = (ball.center - blade.center) / blade.half_span
    return np.r_[offset, np.log(ball.radius / blade.half_span)]


def sensory(history):
    finite = [(tick, row) for tick, row in history if row is not None]
    if len(finite) < 3:
        return None
    t, x = (np.asarray(items) for items in zip(*finite, strict=True))
    t = (t - t.mean()) / 7.
    velocity = t @ x / max(1.e-8, t @ t)
    return np.r_[np.median(x, axis=0), velocity]


def drop(arena, seed, stage, body, choose=None, *, inspect=False):
    obs = arena.reset(seed=seed, split=stage)
    initial = obs['joints'].copy()
    anchor = np.asarray(body['weights'])
    actor = MotorPolicy(anchor, target_effects(body, arena.action_size), readout=body['readout'], retinal=body['retinal'])
    commands, witnessed, awards = [], [], []
    past = deque(maxlen=6)
    pulse_at = None
    context = None
    residual = np.zeros(arena.action_size)
    done = False
    h = sha256()
    reward = 0.
    while not done:
        tick = len(commands)
        sampled = tick % body['stride'] == 0
        if sampled and (pulse_at is None or inspect):
            rgb = obs['rgb']
            track = optical_state(rgb)
            if pulse_at is None:
                past.append((tick, track))
                h.update(rgb)
            elif OBSERVATION[0] <= tick - pulse_at <= OBSERVATION[1]:
                witnessed.append((tick, track))
        if awards and pulse_at is None:
            cue = sensory(past)
            pulse_at = tick
            if cue is not None:
                context = np.r_[cue, 4. * (obs['joints'] - initial)]
                if choose is not None:
                    residual = choose(context)
        command = actor.command(obs, sampled=sampled)
        if pulse_at is not None and tick - pulse_at < PULSE:
            command = np.clip(command + residual, -1., 1.)
            # Preserve the executed command for the optical history.
            actor.last_action = command
        commands.append(command.tolist())
        obs, earned, done = arena.step(command, render=(tick + 1) % body['stride'] == 0)
        reward += earned
        if earned:
            awards.append(tick + 1)
    measured = sensory(witnessed)
    result = arena.judge_metrics()
    return result | dict(actions=commands, score=reward, reward_ticks=awards, pulse=pulse_at,
                         pre_pulse_rgb_sha256=h.hexdigest(), context=None if context is None else context.tolist(),
                         residual=residual.tolist(), achieved=None if measured is None else measured.tolist())


def memories(practice):
    rows = [visit for visit in practice if visit['base']['context'] is not None]
    x = np.array([visit['base']['context'] for visit in rows])
    u = np.array([visit['probe']['residual'] for visit in rows])
    public = np.array([visit['probe']['score'] - visit['base']['score'] for visit in rows])
    matched = np.array([all(visit[which]['achieved'] is not None for which in ('base', 'probe')) for visit in rows])
    base = np.array([rows[i]['base']['achieved'] for i in np.flatnonzero(matched)])
    perturbed = np.array([rows[i]['probe']['achieved'] for i in np.flatnonzero(matched)])
    entries = [(i, visit[which]['achieved'], visit[which]['score']) for i, visit in enumerate(rows)
               for which in ('base', 'probe') if visit[which]['achieved'] is not None]
    if len(x) < 4 or len(base) < 4 or len(entries) < 4:
        return None
    top = np.quantile([score for _, _, score in entries], .65)
    goals = [(i, future) for i, future, score in entries if score >= max(top, 2.)]
    if len(goals) < 3:
        return None
    q = np.array([row for _, row in goals])
    sources = np.array([i for i, _ in goals])
    return dict(context=x, context_scale=np.maximum(x.std(axis=0), .18), movement=u, reward=np.tanh(public / 2.),
                matched=matched, base=base, optical_delta=perturbed - base,
                optical_scale=np.maximum(np.r_[base, perturbed].std(axis=0), .22),
                goals=q, goal_sources=sources)


def motor(context, memory, mode):
    if mode == 'warm':
        return np.zeros(memory['movement'].shape[1])
    gap = (memory['context'] - context) / memory['context_scale']
    distance = np.mean(gap**2, axis=1)
    influence = np.exp(-.65 * distance)
    u = memory['movement']
    penalty = .16 * np.eye(u.shape[1])
    if np.max(influence) < .16:
        return np.zeros(u.shape[1])
    reward_gradient = np.linalg.solve(u.T @ (influence[:, None] * u) + penalty,
                                      u.T @ (influence * memory['reward']))
    action = reward_gradient * .43
    if mode == 'hindsight':
        valid = memory['matched']
        support = influence[valid]
        fit = u[valid]
        scale = memory['optical_scale']
        effects = np.linalg.solve(fit.T @ (support[:, None] * fit) + penalty,
                                  fit.T @ (support[:, None] * memory['optical_delta'] / scale))
        endpoint = np.average(memory['base'], axis=0, weights=support)
        goals = (memory['goals'] - endpoint) / scale
        weights = influence[memory['goal_sources']] * np.exp(-.35 * np.sum(goals**2, axis=1))
        if weights.max() > .025:
            desired = np.average(goals, axis=0, weights=weights)
            action = np.linalg.solve(effects @ effects.T + .65 * np.eye(len(u.T)),
                                     effects @ desired + .29 * reward_gradient)
    return action * min(1., .32 / max(.001, np.linalg.norm(action)))


def summarize(held):
    values = {mode: np.array([episode['rebounds'] for episode in rows]) for mode, rows in held.items()}
    rng = np.random.default_rng(91380975)
    bins = rng.integers(0, HELD, size=(4096, HELD))
    comparisons = {}
    for mode in ('reward', 'warm'):
        delta = values['hindsight'] - values[mode]
        comparisons[mode] = dict(mean=float(delta.mean()), higher=int((delta > 0).sum()),
                                 lower=int((delta < 0).sum()), bootstrap95=np.quantile(delta[bins].mean(1), (.025, .975)).tolist())
    return dict(totals={mode: int(y.sum()) for mode, y in values.items()},
                maximum={mode: int(y.max()) for mode, y in values.items()},
                hindsight_minus=comparisons)


def practice_audit(practice):
    changes = np.array([row['probe']['score'] - row['base']['score'] for row in practice])
    complete = [row for row in practice if all(row[arm]['achieved'] is not None for arm in ('base', 'probe'))]
    u = np.array([row['probe']['residual'] for row in complete])
    base = np.array([row['base']['achieved'] for row in complete])
    follow = np.array([row['probe']['achieved'] for row in complete])
    error, persistence = [], []
    for i in range(len(complete) if len(complete) >= 2 else 0):
        others = np.arange(len(complete)) != i
        scale = np.maximum(np.r_[base[others], follow[others]].std(axis=0), .22)
        observed = (follow - base) / scale
        local = u[others]
        gain = np.linalg.solve(local.T @ local + .16 * np.eye(u.shape[1]), local.T @ observed[others])
        error.extend(np.square(observed[i] - u[i] @ gain).tolist())
        persistence.extend(np.square(observed[i]).tolist())
    result = dict(positive_rewards=int((changes > 0.).sum()), paired_reward_deltas=changes.tolist(),
                  optical_pairs=len(complete), observed_motor_rank=int(np.linalg.matrix_rank(u)) if len(u) else 0)
    if error:
        result.update(held_out_effect_rmse=float(np.mean(error)**.5), persistence_rmse=float(np.mean(persistence)**.5))
    return result


def serialize(memory):
    return {name: value.tolist() for name, value in memory.items()}


def run():
    began = perf_counter()
    rng = np.random.default_rng(SEED)
    schedule = rng.choice(2147483646, TRAIN + HELD, replace=False).tolist()
    body = _read(EMBODIMENT)
    known = set(body['practice_seeds']) | {episode['seed'] for trials in body['held'].values() for episode in trials}
    if known.intersection(schedule):
        raise ValueError('practice or held starts overlap prior embodiment exposure')
    metadata = dict(schema='tt24-hindsight-arm-1', embodiment_sha256=_hash(EMBODIMENT),
                    native_audit_sha256=_hash(NATIVE_AUDIT), inherited_attempts=body['total_initial_attempts'],
                    schedule=SEED, practice=schedule[:TRAIN], held=schedule[TRAIN:], pulse=PULSE,
                    observation=OBSERVATION, practice_limit=2 * TRAIN)
    if RECEIPT.exists():
        report = _read(RECEIPT)
        metadata['observation'] = list(OBSERVATION)
        if metadata != report['metadata']:
            raise ValueError('hindsight receipt uses different body or starts')
    else:
        report = dict(metadata=metadata, practice=[], held={mode: [] for mode in MODES})
    arena = BounceArena(view='wrist', width=body['width'], height=body['height'])
    try:
        for i, seed in enumerate(schedule[:TRAIN]):
            direction = rng.normal(size=arena.action_size)
            direction *= .72 / np.linalg.norm(direction)
            if i < len(report['practice']):
                continue
            base, probe = (drop(arena, seed, 'practice', body, choose=choice, inspect=True)
                           for choice in (None, lambda _: direction))
            if base['context'] != probe['context'] or base['pulse'] != probe['pulse']:
                raise ValueError('paired local exploration had mismatched first reward cue')
            if base['pre_pulse_rgb_sha256'] != probe['pre_pulse_rgb_sha256']:
                raise ValueError('paired local exploration differed before the intervention')
            report['practice'].append(dict(seed=seed, base=base, probe=probe))
            _write(RECEIPT, report)
            print('paired wrist', seed, base['score'], probe['score'], base['achieved'] is not None,
                  probe['achieved'] is not None, flush=True)
        learned = memories(report['practice'])
        report['practice_audit'] = practice_audit(report['practice'])
        if learned is None:
            report['assessment'] = 'insufficient paired achieved wrist goals'
            _write(RECEIPT, report)
            print(report['assessment'], flush=True)
            return
        report['memory'] = serialize(learned)
        for i, seed in enumerate(schedule[TRAIN:]):
            for mode in MODES if i % 2 == 0 else MODES[::-1]:
                if len(report['held'][mode]) > i:
                    continue
                episode = drop(arena, seed, 'held', body, choose=lambda cue: motor(cue, learned, mode))
                report['held'][mode].append(episode)
                _write(RECEIPT, report)
                print('held', seed, mode, episode['rebounds'], round(np.linalg.norm(episode['residual']), 3), flush=True)
            if len({report['held'][mode][i]['pre_pulse_rgb_sha256'] for mode in MODES}) != 1:
                raise ValueError('held courses differ before the first public reward')
        for runs in (report['held'], {row['seed']: [row['base'], row['probe']] for row in report['practice']}):
            for rows in runs.values():
                for episode in rows:
                    verify_replay(arena, episode)
        report['summary'] = summarize(report['held'])
        report['replayed'] = 2 * TRAIN + len(MODES) * HELD
        report['seconds'] = perf_counter() - began
        _write(RECEIPT, report)
        print(report['summary'], flush=True)
    finally:
        arena.close()


if __name__ == '__main__':
    run()
