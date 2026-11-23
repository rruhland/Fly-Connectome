"""Compare native Pong credit through a fixed eye and an eye on the moving wrist."""

from argparse import ArgumentParser
from copy import deepcopy
import gzip
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_bounce_trial import play, search
from .tt24_motor import MotorEffects, MotorPolicy, Sighting, optical_sightings
from .tt24_predict import seeds
from .tt24_source import VisualConsequence


OUT = Path('artifacts/tt24/predict-external.json.gz')
PRACTICE_SEED = 85441
HELD_SEED = 85442
STRIDE = 2
CONTROL = dict(stride=STRIDE, retinal='relative', readout='flow')


def track(sighting):
    return np.r_[sighting.center, np.log(sighting.radius), np.log(sighting.half_span)]


def sighting(value):
    return Sighting(value[:2], *np.exp(value[2:]))


class PairedArena(BounceArena):
    """Record synchronized images without adding an observation to the teacher."""

    def __init__(self, view):
        self.pair = False
        self.samples = []
        super().__init__(view=view, width=192, height=144, max_steps=360)

    def _observation(self, render):
        obs = super()._observation(render)
        if self.pair and render:
            outside = optical_sightings(obs['rgb'])
            wrist = optical_sightings(self.render_wrist())
            if all(item is not None for item in outside + wrist):
                self.samples.append((self._seed, np.r_[track(wrist[0]), track(wrist[1]), obs['joints']],
                                     np.r_[track(outside[0]), track(outside[1])]))
        return obs


class FixedFromWrist:
    """Learn the optical correspondence without motor labels or true ball state."""

    def __init__(self, rows):
        _, wrist, outside = zip(*rows, strict=True)
        state = np.asarray(wrist)
        targets = np.asarray(outside)
        self.center = state.mean(axis=0)
        self.scale = np.maximum(state.std(axis=0), .012)
        self.low = targets.min(axis=0) - .10
        self.high = targets.max(axis=0) + .10
        x = self.design(state)
        regularization = np.eye(x.shape[1]) * 1.1
        regularization[0, 0] = 0.
        self.readout = np.linalg.solve(x.T @ x + regularization, x.T @ targets)
        self.joints = None

    def design(self, state):
        x = np.clip((state - self.center) / self.scale, -4., 4.)
        left, right = np.triu_indices(x.shape[1])
        return np.c_[np.ones(len(x)), x, .40 * x[:, left] * x[:, right]]

    def predict(self, state):
        return np.clip(self.design(state) @ self.readout, self.low, self.high)

    def observe(self, rgb, previous_action=None):
        ball, paddle = optical_sightings(rgb)
        if ball is None or paddle is None:
            return None, None
        state = np.r_[track(ball), track(paddle), self.joints]
        values = self.predict(state[None])[0]
        return sighting(values[:4]), sighting(values[4:])


def paired_validation(samples):
    groups = list(dict.fromkeys(row[0] for row in samples))
    results = []
    for selection in range(3):
        test = set(groups[selection::3])
        training = [row for row in samples if row[0] not in test]
        _, state, target = zip(*(row for row in samples if row[0] in test), strict=True)
        prediction = FixedFromWrist(training).predict(np.asarray(state))
        expected = np.asarray(target)
        results.append(dict(n=len(expected), ball_center_rmse=float(np.mean((prediction[:, :2] - expected[:, :2])**2)**.5),
                            paddle_center_rmse=float(np.mean((prediction[:, 4:6] - expected[:, 4:6])**2)**.5)))
    return results


def deploy(arena, launch, weights, effects, decoder=None):
    obs = arena.reset(seed=launch, split='held', render=decoder is not None)
    policy = MotorPolicy(weights, deepcopy(effects), readout='flow', retinal='relative', detector=decoder)
    commands = []
    done = False
    while not done:
        if decoder is not None:
            decoder.joints = obs['joints']
        sampled = len(commands) % STRIDE == 0
        command = policy.command(obs, sampled=sampled, blind=decoder is None)
        commands.append(command.tolist())
        obs, _, done = arena.step(command, render=decoder is not None and len(commands) % STRIDE == 0)
    return arena.judge_metrics() | dict(commands=commands, visible=policy.optics.visible)


def summarize(rows):
    return dict(mean=float(np.mean([r['rebounds'] for r in rows])),
                best=max(r['rebounds'] for r in rows),
                longest=max(r.get('longest_consecutive_rebounds', r['rebounds']) for r in rows))


def run(output):
    started = perf_counter()
    learning, evaluation = {}, {}
    practice, held = seeds(PRACTICE_SEED, 12), seeds(HELD_SEED, 12)
    teacher = None
    for lens in ('external', 'wrist'):
        arena = PairedArena(view=lens)
        try:
            for mode in ('pong', 'shuffled', 'flat'):
                name = f'{lens}_{mode}'
                effects = MotorEffects(arena.action_size)
                prior = None if mode == 'flat' else VisualConsequence.load(mode)
                arena.pair = name == 'external_pong'
                weights, training = search(arena, practice, prior, effects, seed=PRACTICE_SEED - 1,
                                           radius=.29, credit_weight=4., log=True, **CONTROL)
                arena.pair = False
                learning[name] = dict(weights=weights.tolist(), rows=training,
                                      source_support=float(np.mean([row['source_support'] for row in training])),
                                      source_credit=float(np.mean([row['credit'] for row in training])))
                if name == 'external_pong' and len(arena.samples) >= 35:
                    teacher = (weights.copy(), deepcopy(effects), FixedFromWrist(arena.samples))
                    learning[name]['paired_frames'] = len(arena.samples)
                    learning[name]['decoder_calibration'] = paired_validation(arena.samples)
                rows = [play(arena, launch, weights, deepcopy(effects), split='held', keep_actions=True,
                             **CONTROL) for launch in held]
                evaluation[name] = rows
                print('condition', name, summarize(rows), learning[name]['source_support'], flush=True)
            if lens == 'wrist' and teacher is not None:
                for mode in ('cross_eye', 'no_eye'):
                    weights, effects, decoder = teacher
                    rows = [deploy(arena, launch, weights, effects, decoder if mode == 'cross_eye' else None)
                            for launch in held]
                    evaluation[mode] = rows
                    print('condition', mode, summarize(rows), flush=True)
            if lens == 'wrist':
                evaluation['hold'] = [play(arena, launch, weights, effects, split='held', blind=True,
                                           hold=True, **CONTROL) for launch in held]
        finally:
            arena.close()
        with gzip.open(output, 'wt') as stream:
            json.dump(dict(schema='tt24-predict-external-1', learning=learning, held=evaluation,
                           summary={key: summarize(rows) for key, rows in evaluation.items()},
                           wall_seconds=perf_counter() - started), stream, separators=(',', ':'))


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    run(args.output)


if __name__ == '__main__':
    main()
