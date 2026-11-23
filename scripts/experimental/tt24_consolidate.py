"""Consolidate two earlier wrist motor policies from their self-executed tapes."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

from .tt19_so101 import SO101_HOME
from .tt19_view_full_game import digest_json
from .tt23_sequence import course, verify
from .tt24_bounce import BounceArena
from .tt24_bounce_ident import DirectPupil, WristPupil, select_pair, summarize
from .tt24_bounce_trial import verify_replay
from .tt24_motor import optical_sightings
from .tt24_rally_improve_optical import OpticalPolicy
from .tt24_rally_improve_phase import PhasePolicy
from .tt24_rally_trial import COACHED, STEP_SIZE, PublicBody, _read, returns
from .tt24_wrist_match import (AUGMENTED, DATA as WRIST, SOURCE as RALLY_OPTS, EncoderRig, WristHistory,
                               WristStation, fit)


DIR = Path('artifacts/tt24')
DATA = DIR / 'consolidate-wrist-actions.npz'
OUT = DIR / 'consolidate-wrist.json.gz'
TABLE_TAPE = DIR / 'wrist-match-held.json.gz'
BOUNCE_TAPE = DIR / 'bounce-ident.json.gz'
BOUNCE_POLICY = DIR / 'bounce-ident-pupil.npz'
ASSESS_TABLE = {'w': 2651000131, 'x': 2651000149}
ASSESS_BOUNCE = tuple(range(265201, 265209))
CONTROL_UNIT = 5.2 / 60.


def vision_context(rgb):
    image = np.asarray(rgb)
    halves = np.array_split(image, 2, axis=0)
    color = np.asarray([view.mean(axis=(0, 1)) / 255. for view in halves]).ravel()
    ball, blade = optical_sightings(image)
    cue = np.zeros(2) if ball is None else ball.center
    size = 0. if blade is None else blade.radius
    return np.r_[color, cue, size]


def context(rgb, joints):
    return np.r_[joints[:5], vision_context(rgb)]


def table_labels():
    with np.load(WRIST) as calibration, np.load(RALLY_OPTS) as ordinary, np.load(AUGMENTED) as augmented:
        lens, _ = fit(calibration)
        sphere = ordinary['sphere']
        actor = PhasePolicy(OpticalPolicy(ordinary, 'interaction'), OpticalPolicy(augmented, 'interaction'), .88)
    return lens, sphere, actor


class TableFeatures:
    def __init__(self, rig, lens, sphere):
        self.sensor = WristHistory(rig, lens, sphere)
        self.home = None
        self.blade = None
        self.native = None

    def observe(self, rgb, joints, issued):
        full = np.r_[joints, -.10] if len(joints) == 5 else joints
        target = np.asarray(SO101_HOME) + np.r_[issued, 0.]
        self.native = self.sensor.observe({'left': rgb}, full, target)
        position, speed = self.native[38:41], self.native[41:44] * 30.
        blade = position - self.native[50:53]
        if self.home is None:
            self.home, self.blade = full[:5].copy(), blade.copy()
        shape = (self.native[49] / 25., .2 * self.native[54])
        return np.r_[position - self.blade, speed, blade - self.blade, shape, self.home, full[:5] - self.home]

    def applied(self, command):
        self.sensor.issued(np.r_[command, 0.])


class BounceFeatures:
    def __init__(self, trained):
        self.trained = trained
        self.sensor = trained.sensor
        self.home = None

    def observe(self, rgb, joints, issued):
        q = np.asarray(joints[:5])
        if self.home is None:
            self.home = q.copy()
        scene = self.sensor.observe({'rgb': rgb, 'joints': q})
        return np.r_[scene, self.home, q - self.home]

    def applied(self, command):
        pass


def consume_table(station, rig, lens, sphere):
    with gzip.open(TABLE_TAPE, 'rt') as stream:
        data = json.load(stream)
    arena = station.arena
    records = []
    for case in data['cases']:
        row = case['results']['wrist']
        arena.lane = row['lane']
        arena.reset(seed=row['seed'], server=-1, render=False)
        body = PublicBody(arena)
        adapter = TableFeatures(rig, lens, sphere)
        x, y, samples = [], [], []
        for tick, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True)):
            rgb = arena.render_cameras()['left']
            q = arena.joints.copy()
            if tick < 3:
                samples.append(context(rgb, q))
            issued = body.target[:5] - np.asarray(SO101_HOME[:5])
            x.append(adapter.observe(rgb, q, issued))
            y.append(issued + STEP_SIZE[:5] * np.asarray(near[:5]))
            opponent = np.asarray(far)
            arena._aim = arena._command_to = lambda *args, opponent=opponent: opponent
            _, reward, done, cap, _ = arena.step(near, render=False)
            body.issued(near)
            adapter.applied(near[:5])
            if done or cap:
                break
        if (len(x) != len(row['actions']) or reward != row['reward'] or arena.point_result() != row['outcome']
                or digest_json(arena.events) != digest_json(row['events'])):
            raise ValueError('table self-practice source replay deviated')
        records.append((row['seed'], x, y, samples))
        print('consolidation source table', row['seed'], len(x), len(returns(row, 1)), flush=True)
    return records


def consume_bounce(arena):
    with gzip.open(BOUNCE_TAPE, 'rt') as stream:
        data = json.load(stream)
    rows = data['cohorts']['wrist_fitted_motion']['episodes']
    records = []
    for row in rows:
        obs = arena.reset(seed=row['seed'], split='held')
        adapter = BounceFeatures(WristPupil.load(BOUNCE_POLICY))
        issued = np.zeros(5)
        x, y, samples = [], [], []
        for tick, action in enumerate(row['commands']):
            if tick < 3:
                samples.append(context(obs['rgb'], obs['joints']))
            x.append(adapter.observe(obs['rgb'], obs['joints'], issued))
            issued += CONTROL_UNIT * np.asarray(action)
            y.append(issued.copy())
            obs, _, done = arena.step(action)
            if done:
                break
        check = arena.judge_metrics()
        if (len(x) != len(row['commands']) or check['longest_consecutive_rebounds'] != row['longest_consecutive_rebounds']
                or check['rebounds'] != row['rebounds']
                or digest_json(check['rebound_events']) != digest_json(row['rebound_events'])):
            raise ValueError('bounce self-practice source replay deviated')
        records.append((row['seed'], x, y, samples))
        print('consolidation source bounce', row['seed'], len(x), check['longest_consecutive_rebounds'], flush=True)
    return records


def collect(path):
    profile = _read(COACHED)['metadata']['opening']['far_profile']
    station = WristStation(profile)
    bounce = BounceArena(view='wrist', width=192, height=144, max_steps=300)
    lens, sphere, _ = table_labels()
    rig = EncoderRig()
    archive = {}
    try:
        for key, groups in (('a', consume_table(station, rig, lens, sphere)), ('b', consume_bounce(bounce))):
            for seed, features, targets, sightings in groups:
                prefix = f'{key}_{seed}'
                archive[f'{prefix}_x'] = np.asarray(features)
                archive[f'{prefix}_y'] = np.asarray(targets)
                archive[f'{prefix}_c'] = np.asarray(sightings)
        np.savez_compressed(path, **archive)
    finally:
        bounce.close()
        station.arena.close()


def training_data(path):
    with np.load(path, allow_pickle=False) as a:
        names = [key[:-2] for key in a.files if key.endswith('_x')]
        return [(name[0], int(name[2:]), *(a[f'{name}_{key}'] for key in ('x', 'y', 'c'))) for name in names]


class Context:
    """Cluster first glimpses without recorded game identities."""

    def __init__(self, episodes):
        sensed = np.vstack([item[4] for item in episodes])
        actual = np.concatenate([np.full(len(item[4]), item[0]) for item in episodes])
        self.center = np.mean(sensed, axis=0)
        self.scale = np.maximum(np.std(sensed, axis=0), .09)
        points = (sensed - self.center) / self.scale
        distance = np.sum((points[:, None, :] - points)**2, axis=2)
        i, j = np.unravel_index(np.argmax(distance), distance.shape)
        anchors = points[[i, j]].copy()
        for _ in range(12):
            membership = np.argmin(np.sum((points[:, None, :] - anchors)**2, axis=2), axis=1)
            anchors = np.asarray([np.mean(points[membership == n], axis=0) for n in range(2)])
        labels = [max(('a', 'b'), key=lambda name: int(np.sum((membership == n) & (actual == name)))) for n in range(2)]
        if len(set(labels)) != 2:
            raise ValueError('unlabelled context did not separate the previously calibrated observation streams')
        self.correct = int(np.sum(np.asarray(labels)[membership] == actual))
        self.samples = len(actual)
        self.prototypes = anchors[np.argsort(labels)]

    def recognize(self, first):
        state = (first - self.center) / self.scale
        cost = np.mean((self.prototypes - state)**2, axis=1)
        score = np.exp(-7.5 * (cost - min(cost)))
        return score / score.sum()


class Actor:
    """One regularized joint output head with shared and context-conditioned features."""

    def __init__(self, episodes, contexts, bases):
        self.contexts = contexts
        pooled = np.concatenate([item[2] for item in episodes])
        self.center = np.median(pooled, axis=0)
        self.scale = np.maximum(np.std(pooled, axis=0), np.r_[[.04] * 3, [.25] * 3, [.025] * 5, [.08] * 10])
        rng = np.random.default_rng(265214177)
        self.frequencies = rng.normal(0., 1.15 / np.sqrt(pooled.shape[1]), (pooled.shape[1], 144))
        self.bases = bases
        rows = [self.design(item[2], self.contexts.recognize(item[4][0])) for item in episodes]
        features = np.concatenate(rows)
        y = np.concatenate([item[3] for item in episodes])
        norm = np.eye(features.shape[1]) * .60
        norm[0, 0] = .01
        self.coefficients = np.linalg.solve(features.T @ features + norm, features.T @ y)

    def design(self, state, gate):
        x = np.clip((np.atleast_2d(state) - self.center) / self.scale, -4., 4.)
        w = x @ self.frequencies
        common = np.c_[np.ones(len(x)), x, .42 * np.cos(w), .42 * np.sin(w)]
        residuals = [basis.design(state) * gate[i] for i, basis in enumerate(self.bases)]
        return np.column_stack((common, *residuals))

    def estimate(self, state, gate):
        return (self.design(state, gate) @ self.coefficients)[0]


def fit_basis(episodes):
    groups = [(item[2], item[3]) for item in episodes]
    pair, _ = select_pair(groups[:4])
    joint_state = np.concatenate([row[0] for row in groups])
    goal = np.concatenate([row[1] for row in groups])
    return DirectPupil(joint_state, goal, pair=pair)


class Controller:
    """Choose decoding from observed context, then issue new joint rates every step."""

    def __init__(self, bank, contexts, rgb, joints, *, learned=None, force=None, ablate=False):
        self.sensors, self.specialist = bank
        self.gate = contexts.recognize(context(rgb, joints))
        observed = int(np.argmax(self.gate))
        self.guess = observed
        self.selected = observed if force is None else force
        self.reader = self.sensors[self.selected]()
        self.actor, self.ablate = learned, ablate

    def command(self, rgb, joints, target, increment):
        state = self.reader.observe(rgb, joints, target)
        if self.actor is not None:
            gate = np.full(2, .5) if self.ablate else self.gate
            desired = self.actor.estimate(state, gate)
            action = np.clip((desired - target) / increment, -1., 1.)
        elif self.selected == 0:
            absolute = np.asarray(SO101_HOME) + np.r_[target, 0.]
            action = self.specialist.command(self.reader.native, absolute)[:5]
        else:
            action = self.reader.trained.actor.command(state, target)
        self.reader.applied(action)
        return action


def create_bounce_features():
    return BounceFeatures(WristPupil.load(BOUNCE_POLICY))


def play_bounce(arena, start, bank, contexts, actor=None, *, force=None, ablate=False):
    obs = arena.reset(seed=start, split='held')
    controller = Controller(bank, contexts, obs['rgb'], obs['joints'], learned=actor, force=force, ablate=ablate)
    issued = np.zeros(5)
    actions = []
    done = False
    while not done:
        action = controller.command(obs['rgb'], obs['joints'], issued, CONTROL_UNIT)
        issued += CONTROL_UNIT * action
        obs, _, done = arena.step(action)
        actions.append(action.tolist())
    metrics = arena.judge_metrics()
    row = metrics | dict(actions=actions, score=metrics['reward'], route=controller.guess,
                         routed_probability=controller.gate.tolist())
    verify_replay(arena, row)
    return row


def play_table(station, start, bank, contexts, actor=None, *, force=None, ablate=False):
    arena = station.arena
    arena.lane = start['lane']
    arena.reset(seed=start['seed'], server=-1, render=False)
    station.tape.steps.clear()
    rgb = arena.render_cameras()['left']
    controller = Controller(bank, contexts, rgb, arena.joints, learned=actor, force=force, ablate=ablate)
    body = PublicBody(arena)
    actions = []
    for _ in range(arena.max_steps):
        issued = body.target[:5] - np.asarray(SO101_HOME[:5])
        action = controller.command(rgb, arena.joints, issued, STEP_SIZE[:5])
        full = np.r_[action, 0.]
        _, reward, done, cap, _ = arena.step(full, render=False)
        actions.append(full.tolist())
        body.issued(full)
        if done or cap:
            break
        rgb = arena.render_cameras()['left']
    from .tt22_rally_trial import alternating

    return dict(seed=start['seed'], lane=start['lane'], actions=actions, reward=reward, outcome=arena.point_result(),
                events=arena.events.copy(), longest=alternating(arena.events), rival_actions=station.tape.tape(len(actions)),
                route=controller.guess, routed_probability=controller.gate.tolist())


def table_summary(trials):
    own = [len(returns(row, 1)) for row in trials]
    return dict(returns=own, total_returns=sum(own), maximum=max(own), second=sum(value >= 2 for value in own),
                longest=max(row['longest'] for row in trials))


def qualify(path, output):
    episodes = training_data(path)
    contexts = Context(episodes)
    table_only = [item for item in episodes if item[0] == 'a']
    bounce_only = [item for item in episodes if item[0] == 'b']
    profile = _read(COACHED)['metadata']['opening']['far_profile']
    station = WristStation(profile)
    bounce = BounceArena(view='wrist', width=192, height=144, max_steps=300)
    lens, sphere, specialist = table_labels()
    rig = EncoderRig()
    banks = ((lambda: TableFeatures(rig, lens, sphere), create_bounce_features), specialist)
    table_starts = course(ASSESS_TABLE)[::3]
    if {item[1] for item in episodes} & ({start['seed'] for start in table_starts} | set(ASSESS_BOUNCE)):
        raise ValueError('qualification overlaps acquired motor tapes')
    result = dict(schema='tt24-two-game-consolidation-1', source_frames={name: sum(len(item[2])
                  for item in episodes if item[0] == name) for name in ('a', 'b')},
                  source_episodes={name: sum(item[0] == name for item in episodes) for name in ('a', 'b')},
                  context_accuracy=[contexts.correct, contexts.samples], table={}, bounce={}, summary={})

    def save():
        with gzip.open(output, 'wt') as stream:
            json.dump(result, stream, separators=(',', ':'))

    def table_batch(name, actor=None, **params):
        rows = [play_table(station, start, banks, contexts, actor, **params) for start in table_starts]
        result['table'][name] = rows
        result['summary'][name] = table_summary(rows)
        print('consolidated table', name, result['summary'][name], flush=True)
        save()

    def bounce_batch(name, actor=None, **params):
        rows = [play_bounce(bounce, seed, banks, contexts, actor, **params) for seed in ASSESS_BOUNCE]
        result['bounce'][name] = rows
        result['summary'][name] = summarize(rows)
        print('consolidated bounce', name, result['summary'][name], flush=True)
        save()

    try:
        table_basis = fit_basis(table_only)
        first = Actor(table_only, contexts, (table_basis,))
        table_batch('table_before', first)
        bases = (table_basis, fit_basis(bounce_only))
        both = Actor(episodes, contexts, bases)
        overwrite = Actor(bounce_only, contexts, bases)
        bounce_batch('bounce_learned', both)
        bounce_batch('bounce_without_context_weights', both, ablate=True)
        bounce_batch('bounce_specialist')
        table_batch('table_after', both)
        table_batch('table_overwritten', overwrite)
        table_batch('table_specialist')
        result['table_replays'] = verify([dict(results={str(i): row for i, row in enumerate(rows)})
                                          for rows in result['table'].values()])
        result['bounce_replays'] = sum(len(rows) for rows in result['bounce'].values())
        save()
    finally:
        bounce.close()
        station.arena.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('collect', 'qualify'))
    p.add_argument('--data', type=Path, default=DATA)
    p.add_argument('--output', type=Path, default=OUT)
    a = p.parse_args()
    if a.mode == 'collect':
        collect(a.data)
    else:
        qualify(a.data, a.output)


if __name__ == '__main__':
    main()
