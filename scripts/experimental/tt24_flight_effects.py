"""Learn optical effects of interventions during one donor flight and spend eight bounce trials."""

import argparse
from collections import deque
import gzip
import json
from pathlib import Path

import numpy as np

from .tt4_arena import mujoco
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_bounce_trial import seeds, verify_replay
from .tt24_donor_trial import DESIGNS, DonorBounce, ROOT
from .tt24_motor import optical_sightings
from .tt24_volley_source import _feed


DATA = ROOT / 'flight-effects-source.json.gz'
RESULT = ROOT / 'flight-effects-transfer.json.gz'
AUDIT = ROOT / 'flight-effects-command-audit.json.gz'
SOURCE_SEED = 266120011
PRACTICE_SEED = 266120021
HELD_SEED = 266120022
FIRST_WINDOW = 30
STEP = 5.2 / 60.
SHAPE = (5, 4)


def write(path, result):
    with gzip.open(path, 'wt') as stream:
        json.dump(result, stream, separators=(',', ':'))


def read(path):
    with gzip.open(path, 'rt') as stream:
        return json.load(stream)


class WristFeedback:
    """Use current optical observations to choose a new five-joint servo target."""

    def __init__(self, weights):
        self.weights = weights
        self.issued = np.zeros(5)
        self.recent = deque(maxlen=4)
        self.origin = None

    def sense(self, obs):
        ball, blade = optical_sightings(obs.get('rgb'))
        if ball is None or blade is None:
            self.recent.clear()
            return np.zeros(4), np.r_[[np.nan] * 3, 0.]
        span = max(.05, blade.half_span)
        center = (ball.center - blade.center) / span
        scale = np.log(max(.02, ball.radius / span))
        observed = np.r_[center, scale]
        if self.origin is None:
            self.origin = observed.copy()
        retinal = observed - self.origin
        self.recent.append(retinal)
        motion = (retinal - self.recent[0]) / max(1, len(self.recent) - 1)
        feature = np.r_[.60, np.tanh(retinal[:2]), np.tanh(9. * motion[2])]
        return feature, np.r_[retinal, 1.]

    def command(self, feature):
        target = .19 * np.tanh(self.weights @ feature)
        return np.clip(.72 * (target - self.issued) / STEP, -.48, .48)

    def applied(self, command):
        self.issued += STEP * command


def reset(arena, seed, source, render):
    obs = arena.reset(seed=seed, split='held' if source else 'practice',
                      regime='changed' if source else 'standard', render=False)
    if source:
        offset, velocity = _feed(seed, catchable=True)
        arena._ball_position[:] = arena._data.geom_xpos[arena._blade] + offset
        arena._ball_velocity[:] = velocity
        mujoco.mj_forward(arena._model, arena._data)
    return arena._observation(render) if render else obs


def signature(optical, q):
    optics = np.asarray(optical)
    poses = np.asarray(q)
    recent = np.zeros(3)
    filled = []
    for row in optics:
        if row[3]:
            recent = row[:3]
        filled.append(np.r_[recent, row[3]])
    while len(filled) < FIRST_WINDOW:
        filled.append(np.r_[recent, 0.])
    filled = np.asarray(filled)
    bins = ((5, 9), (12, 18), (21, 27))
    visual = np.concatenate([filled[left:right].mean(axis=0) for left, right in bins])
    motor = np.concatenate([poses[min(tick, len(poses) - 1)] - poses[0] for tick in (7, 14, 24)])
    return visual, motor


def episode(arena, seed, weights, *, source=False, split='practice'):
    obs = reset(arena, seed, source, render=True) if source else arena.reset(seed=seed, split=split)
    feedback = WristFeedback(weights)
    actions, visual, joints = [], [], []
    first_contact = None
    done = False
    while not done:
        tick = len(actions)
        x, optics = feedback.sense(obs)
        if tick < FIRST_WINDOW:
            joints.append(obs['joints'].copy())
            visual.append(optics)
        active = first_contact is None or not source
        action = feedback.command(x) if active else np.zeros(5)
        obs, reward, done = arena.step(action)
        feedback.applied(action)
        actions.append(action.tolist())
        # The source fixture stops the agent at the first physical encounter.
        # Its learning row contains only observations, actions and parameters.
        if source and first_contact is None and arena._paddle_contacts:
            first_contact = len(actions)
        if source and (reward > 0. or arena._paddle_contacts >= 2 or len(actions) >= FIRST_WINDOW):
            break
    if not source:
        metrics = arena.judge_metrics()
        row = metrics | dict(actions=actions, score=metrics['reward'])
        verify_replay(arena, row)
    else:
        metrics = arena.judge_metrics()
        reset(arena, seed, source=True, render=False)
        for action in actions:
            arena.step(action, render=False)
        if digest_json(metrics) != digest_json(arena.judge_metrics()):
            raise ValueError(f'source first flight changed physical tape on {seed}')
        row = dict(seed=seed, actions=actions, frames=metrics['steps'], first_contact=first_contact,
                   sensory_decisions=first_contact or len(actions), referee_rebound=metrics['rebounds'])
    visual_out, motor_out = signature(visual, joints)
    return row | dict(optical_effect=visual_out.tolist(), joint_effect=motor_out.tolist())


def collect(output):
    random = np.random.default_rng(SOURCE_SEED)
    opts = dict(view='wrist', width=96, height=72, max_steps=FIRST_WINDOW)
    source = []
    for body, model in enumerate(DESIGNS):
        arena = DonorBounce(model, **opts)
        try:
            for seed in random.integers(1, 2**31 - 1, 8):
                perturbation = random.normal(size=SHAPE)
                perturbation *= .52 / np.linalg.norm(perturbation)
                first, second = (episode(arena, int(seed), sign * perturbation, source=True) for sign in (1., -1.))
                source.append(dict(body=body, seed=int(seed), delta=(2. * perturbation).tolist(), plus=first, minus=second))
                print('one-flight source', body, int(seed), first['frames'], second['frames'],
                      first['first_contact'], second['first_contact'], flush=True)
        finally:
            arena.close()
    write(output, dict(schema='tt24-first-flight-effects-1', source=source, physical_replays=len(source) * 2,
                       conditioning='matched release and body, randomized signed visual feedback coefficients',
                       source_control_frames=sum(item[condition]['sensory_decisions'] for item in source
                                                 for condition in ('plus', 'minus')),
                       source_physics_frames=sum(item[condition]['frames'] for item in source
                                                 for condition in ('plus', 'minus'))))


def arrays(pairs):
    change = np.asarray([item['delta'] for item in pairs]).reshape(-1, int(np.prod(SHAPE)))
    outcome = np.asarray([np.r_[np.asarray(item['plus']['optical_effect']) - item['minus']['optical_effect'],
                                np.asarray(item['plus']['joint_effect']) - item['minus']['joint_effect']]
                          for item in pairs])
    return change, outcome


class Reaction:
    """Fit signed action-policy interventions to their observable effects."""

    def __init__(self, x, observed):
        self.scale = np.maximum(np.std(observed, axis=0), np.r_[np.full(12, .08), np.full(15, .016)])
        y = observed / self.scale
        regularizer = np.eye(x.shape[1]) * .36
        self.readout = np.linalg.solve(x.T @ x + regularizer, x.T @ y)

    def predict(self, x):
        return x @ self.readout * self.scale

    def modes(self):
        u, _, _ = np.linalg.svd(self.readout[:, :12], full_matrices=False)
        basis = u[:, :2].T.copy()
        for mode in basis:
            if mode[np.argmax(abs(mode))] < 0.:
                mode *= -1.
        return basis


def rmse(x, y, true, shuffled, subsets=(slice(None, 12), slice(12, None))):
    expected = (true.predict(x), shuffled.predict(x), np.zeros_like(y))
    return {mode: [float(np.sqrt(np.mean(((pred[:, subset] - y[:, subset]) /
                                         true.scale[subset])**2))) for subset in subsets]
            for mode, pred in zip(('causal', 'shuffled', 'null'), expected, strict=True)}


def baseline(x, y, random):
    actual = Reaction(x, y)
    altered = Reaction(x, y[random.permutation(len(y))])
    return actual, altered


def target(path, output):
    source = read(path)
    x, y = arrays(source['source'])
    random = np.random.default_rng(SOURCE_SEED ^ 227)
    pairs = np.asarray([item['body'] for item in source['source']])
    validation = {}
    for body in range(3):
        use = pairs != body
        validation[str(body)] = rmse(x[~use], y[~use], *baseline(x[use], y[use], random))
    true, shuffle = baseline(x, y, random)
    cold, _ = np.linalg.qr(random.normal(size=(x.shape[1], 2)))
    modes = dict(causal=true.modes(), shuffled=shuffle.modes(), cold=cold.T)
    practice = seeds(PRACTICE_SEED, 2)
    held = seeds(HELD_SEED, 12)
    result = dict(schema='tt24-first-flight-effect-transfer-1', source=str(path),
                  source_control_frames=source['source_control_frames'], source_episodes=len(pairs) * 2,
                  source_cross_validation=validation, practice_starts=practice, attempts_per_learner=8,
                  held_starts=held, target_demonstrations=0, practice={}, held={}, summary={})
    print('unseen donor-body effect prediction', validation, flush=True)
    new_x, new_y = [], []
    arena = BounceArena(view='wrist', width=96, height=72, max_steps=300, apex_reward_weight=.24)
    try:
        for name, basis in modes.items():
            first, second = (.43 * direct.reshape(SHAPE) for direct in basis)
            candidates = (np.zeros(SHAPE), first, -first, second)
            trials = [[episode(arena, seed, weights) for seed in practice] for weights in candidates]
            merits = [np.mean([row['score'] + .2 * min(row['steps'] / 150., 1.) for row in runs]) for runs in trials]
            best = int(np.argmax(merits))
            predictions = [dict(delta=(2. * first).tolist(), plus=trials[1][i], minus=trials[2][i]) for i in range(2)]
            predictions += [dict(delta=second.tolist(), plus=trials[3][i], minus=trials[0][i]) for i in range(2)]
            target_x, target_y = arrays(predictions)
            new_x.append(target_x)
            new_y.append(target_y)
            result['practice'][name] = dict(trials=trials, weights=[item.tolist() for item in candidates], merits=merits,
                                            selected=best, body_counterfactual=rmse(target_x, target_y, true, shuffle))
            rows = [episode(arena, seed, candidates[best], split='held') for seed in held]
            result['held'][name] = rows
            counts = [row['longest_consecutive_rebounds'] for row in rows]
            result['summary'][name] = dict(counts=counts, mean=float(np.mean(counts)), max=max(counts))
            print('first-flight effects held', name, 'selected', best, result['summary'][name], flush=True)
            write(output, result)
        rows = [episode(arena, seed, np.zeros(SHAPE), split='held') for seed in held]
        result['held']['stationary'] = rows
        values = [row['longest_consecutive_rebounds'] for row in rows]
        result['summary']['stationary'] = dict(counts=values, mean=float(np.mean(values)), max=max(values))
        result['target_body_counterfactual'] = rmse(np.vstack(new_x), np.vstack(new_y), true, shuffle)
        print('first-flight negative and positive controls', result['summary'], result['target_body_counterfactual'], flush=True)
        write(output, result)
    finally:
        arena.close()


def command_blocks(tape):
    actions = np.asarray(tape['actions'])
    padded = np.zeros((10, 5))
    use = min(len(actions), 10)
    padded[:use] = actions[:use]
    return padded.reshape(2, 5, 5).sum(axis=1).ravel()


def direct_input(pairs):
    _, effects = arrays(pairs)
    signals = np.asarray([command_blocks(item['plus']) - command_blocks(item['minus']) for item in pairs])
    return signals, effects


def audit(source_path, target_path, out):
    records = read(source_path)['source']
    tested = read(target_path)['practice']
    rng = np.random.default_rng(SOURCE_SEED + 867)
    source_x, source_y = direct_input(records)
    scopes = (slice(4, 12), slice(17, None))
    bodies = np.asarray([item['body'] for item in records])
    source = {}
    for body in range(3):
        train = bodies != body
        estimate = baseline(source_x[train], source_y[train], rng)
        source[str(body)] = rmse(source_x[~train], source_y[~train], *estimate, subsets=scopes)
    estimate = baseline(source_x, source_y, rng)
    compared = {}
    all_x, all_y = [], []
    for mode, group in tested.items():
        trial = group['trials']
        weight = np.asarray(group['weights'])
        pairs = [dict(plus=trial[a][i], minus=trial[b][i], delta=weight[a] - weight[b])
                 for a, b in ((1, 2), (3, 0)) for i in range(2)]
        x, y = direct_input(pairs)
        compared[mode] = rmse(x, y, *estimate, subsets=scopes)
        all_x.append(x)
        all_y.append(y)
    report = dict(schema='tt24-first-flight-command-effect-audit-1', input='signed change in issued rates summed over two five-frame blocks',
                  outcome='wrist appearance at frames 12–18 and 21–27, encoders at 14 and 24',
                  source_first_flight_pairs=len(records), target_pairs=sum(len(x) for x in all_x),
                  unseen_body_effects=source, target_sets=compared,
                  combined_target=rmse(np.vstack(all_x), np.vstack(all_y), *estimate, subsets=scopes),
                  extra_physical_probes=0)
    write(out, report)
    print('actual command effect audit', report, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('collect', 'target', 'audit'))
    parser.add_argument('--data', type=Path, default=DATA)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--target', type=Path, default=RESULT)
    args = parser.parse_args()
    if args.stage == 'collect':
        collect(args.data)
    elif args.stage == 'audit':
        audit(args.data, args.target, args.output or AUDIT)
    else:
        target(args.data, args.output or RESULT)


if __name__ == '__main__':
    main()
