"""Apply an anonymous score-urgency decision to the continuous TT22 rally controller."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_direct_receive import previous_seeds
from .tt19_flight import FlightLaw
from .tt19_model_rollout import BallObservation
from .tt19_shoot import COURT
from .tt20_natural import NaturalViewTransport
from .tt22_opponent_trial import FarTape
from .tt22_rally import RallyPlayer
from .tt22_rally_trial import CONFIG, RallyArena, alternating, opponent, rates
from .tt22_relation import Observation, learn, predict
from .tt22_relation_trial import OUTPUT, library


SEED = 1984053397
DECISION = 9
TARGET_HORIZON = 132
BUDGETS = (6, 12)
MODES = ('pong', 'score_shuffled', 'relation_shuffled', 'scratch')
ARCHIVES = ('rally-anchor-612423769.json.gz', 'rally-anchor-220381117.json.gz')
SPREAD, PACE, RIVAL = .30, .82, 'anchor'


class UrgentPlayer(RallyPlayer):
    def search(self, estimate, observation, target):
        if self.lead:
            p, v = self.law.trajectory(*estimate, 1, bounds=COURT)
            estimate = p[-1], v[-1]
            bounced = observation.bounced or (p[-1, 1] < 0 and v[0, 2] < 0. < v[-1, 2])
            observation = BallObservation(p[-1], 30. * v[-1], observation.spin, bool(bounced))
        return super().search(estimate, observation, target)


class Station:
    def __init__(self, shared):
        self.shared = shared
        self.arena = RallyArena(SPREAD, PACE)
        self.rival = opponent(self.arena, RIVAL)
        self.recorder = FarTape(self.arena)
        self.law = FlightLaw()
        self.player = UrgentPlayer(self.law)

    def close(self):
        self.player.close()
        self.arena.close()

    def context(self, seed, reference=None):
        self.arena.reset(seed=seed, server=-1, render=False)
        self.player.reset()
        self.player.lead = 0
        observer = Observation(self.shared)
        previous, commands = None, []
        for tick in range(DECISION + 1):
            frames = self.arena.render_cameras()
            cue = observer.update(frames['left'], previous)
            if tick < DECISION:
                action = self.player.command(tick, None)
                if reference is not None and action.tolist() != reference[tick]:
                    raise ValueError('the replayed neutral prefix changed')
                commands.append(action.tolist())
                self.arena.step(action, render=False)
                previous = action
        return cue, commands

    def point(self, seed, lead, prefix, cue):
        self.arena.reset(seed=seed, server=-1, render=False)
        self.player.reset()
        self.player.lead = 0
        self.recorder.steps.clear()
        scene = NaturalViewTransport('fixed', lens_fov=self.arena.lens_intrinsics(), law=self.law,
                                     flight=False, associate=True).tracker()
        observer = Observation(self.shared)
        commands, previous = [], None
        for tick in range(self.arena.max_steps):
            frames = self.arena.render_cameras()
            if tuple(frames) != ('left', 'right'):
                raise ValueError('unexpected scaffold cameras')
            if tick <= DECISION:
                observed = observer.update(frames['left'], previous)
                if tick == DECISION:
                    if not np.array_equal(cue, observed):
                        raise ValueError('early anonymous context changed')
                    self.player.lead = lead
            scene.update(frames, tick=tick)
            ball = scene.ball
            point = ball.position if (ball is not None and ball.observed
                                      and scene.sources.get('ball_range') == 'stereo') else None
            action = self.player.command(tick, point)
            if tick < DECISION and action.tolist() != prefix[tick]:
                raise ValueError('an incoming stroke interrupted the common decision prefix')
            commands.append(action.tolist())
            _, reward, done, capped, _ = self.arena.step(action, render=False)
            previous = action
            if done or capped:
                break
        return dict(seed=seed, role=-1, lead=lead, actions=commands, rival_actions=self.recorder.tape(len(commands)),
                    outcome=self.arena.point_result(), reward=reward, longest=alternating(self.arena.events),
                    events=self.arena.events.copy(), rgb_sha256=scene.digest.hexdigest(), plans=self.player.shots)


def practice(folder, station):
    observations = []
    for filename in ARCHIVES:
        path = folder / filename
        archive = _read(path)
        if (archive['spread'], archive['pace'], archive['rival'], archive['privileged']) != (SPREAD, PACE, RIVAL, False):
            raise ValueError('practice is from a different arena or control condition')
        for row in archive['rows'][:6]:
            cue, _ = station.context(row['seed'], row['actions'])
            public_score = row['reward']
            remaining = len(row['actions']) - DECISION
            observations.append(dict(seed=row['seed'], visual=cue.tolist(), reward=public_score,
                                     control_frames=len(row['actions']), near_score=remaining <= TARGET_HORIZON
                                     and public_score != 0, archive=filename, archive_sha256=_hash(path)))
    if len(observations) != max(BUDGETS) or len({row['seed'] for row in observations}) != len(observations):
        raise ValueError('wrong number of disjoint target practice points')
    return observations


def models(observations, priors):
    result = {}
    for budget in BUDGETS:
        rows = observations[:budget]
        x = np.asarray([row['visual'] for row in rows])
        y = np.asarray([row['near_score'] for row in rows], dtype=float)
        result[str(budget)] = {}
        for mode in MODES:
            weights = learn(x, y, prior=priors.get(mode))
            cutoff = float(np.median(predict(weights, x)))
            result[str(budget)][mode] = (weights, cutoff)
    return result


def replay(points):
    arena = RallyArena(SPREAD, PACE)
    try:
        for point in points:
            arena.reset(seed=point['seed'], server=-1, render=False)
            for tick, (near, far) in enumerate(zip(point['actions'], point['rival_actions'], strict=True)):
                arena._rival_attempt = True
                arena._aim = lambda *args: np.asarray(far)
                _, reward, done, capped, _ = arena.step(near, render=False)
                if bool(done or capped) != (tick == len(point['actions']) - 1):
                    raise ValueError('two-arm tapes changed physical point duration')
            if (arena.point_result() != point['outcome'] or reward != point['reward'] or
                    json.dumps(arena.events) != json.dumps(point['events'])):
                raise ValueError(f'two-arm replay changed at {point["seed"]}')
    finally:
        arena.close()
    return len(points)


def summary(rows):
    return {str(budget): {mode: rates([row['options'][str(row['choices'][str(budget)][mode])] for row in rows])
                          for mode in MODES} for budget in BUDGETS} if rows else {}


def run(folder, pairs, screen):
    forecast = folder / 'relation-forecast.json.gz'
    output = folder / 'relation-urgency.json.gz'
    prior = {mode: np.asarray(w) for mode, w in _read(forecast)['prior']['terminal'].items()}
    seeds = np.random.default_rng(SEED).integers(2**31, size=screen).tolist()
    reserved = previous_seeds(folder)
    if output.exists():
        previous = _read(output)
        if previous.get('seed') == SEED and previous.get('schema') == 'tt22-relation-urgency-2':
            reserved.difference_update(row['seed'] for row in previous['screened'])
    if len(set(seeds)) != screen or set(seeds) & reserved:
        raise ValueError('a screened seed overlaps existing TT22 practice or assessment')
    result = dict(schema='tt22-relation-urgency-2', seed=SEED, screened=[], rows=[], arena=CONFIG,
                  target_score_horizon=TARGET_HORIZON, decision_tick=DECISION, budgets=BUDGETS,
                  policy_sensors=['left', 'right'], relation_sensors=['left'], spread=SPREAD, pace=PACE,
                  rival=RIVAL, forecast_sha256=_hash(forecast), matched_required=pairs, complete=False)
    with TemporaryDirectory() as temp:
        station = Station(library(temp))
        try:
            result['practice'] = practice(folder, station)
            fitted = models(result['practice'], prior)
            for seed in seeds:
                cue, prefix = station.context(seed)
                choices = {n: {m: int(float(predict(w, cue)) >= cut) for m, (w, cut) in group.items()}
                           for n, group in fitted.items()}
                disagreement = any(group['pong'] != group['score_shuffled'] for group in choices.values())
                result['screened'].append(dict(seed=seed, selected=disagreement, choices=choices))
                _write(output, result)
                if not disagreement:
                    continue
                played = {str(lead): station.point(seed, lead, prefix, cue) for lead in (0, 1)}
                result['rows'].append(dict(seed=seed, choices=choices, options=played))
                print(seed, choices, {k: (v['outcome']['legal_returns'], v['longest']) for k,v in played.items()}, flush=True)
                _write(output, result)
                if len(result['rows']) >= pairs:
                    break
        finally:
            station.close()
    result['verified_tapes'] = replay([row for experiment in result['rows'] for row in experiment['options'].values()])
    result['metrics'] = summary(result['rows'])
    result['complete'] = True
    _write(output, result)
    print(result['metrics'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--pairs', type=int, default=4)
    parser.add_argument('--screen', type=int, default=32)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify is not None:
        receipt = _read(args.verify)
        count = replay([row for case in receipt['rows'] for row in case['options'].values()])
        if count != receipt['verified_tapes']:
            raise ValueError('replay count changed')
        print('replayed', count)
    elif not 1 <= args.pairs <= args.screen:
        parser.error('choose positive pair count and at least as many screened starts')
    else:
        run(args.folder, args.pairs, args.screen)
