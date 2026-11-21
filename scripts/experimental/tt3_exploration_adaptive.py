"""Learn when to stop probing from independently repeated visual motor evidence."""

from collections import deque
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from scripts.experimental import tt3_exploration as experiment
from scripts.experimental.tt3_exploration import tt3_arena, tt3_pong, tt3_visual


FUTURE_FRAMES = 28
SOURCE_FRAMES = experiment.PROBE_FRAMES + FUTURE_FRAMES
TARGET_SEED = 348077129
CUTOFFS = (.01, .02, .035, .05, .08, .12, .18, .28, .42, np.inf)
ARMS = ('prior_stop', 'fixed_16', 'simple_stop', 'sham_stop')


def _contrast(evidence):
    return float(evidence.features[0] * evidence.features[1])


class _IndependentWindows:
    def __init__(self, seed):
        self.dwell = experiment.DWELLS[seed % len(experiment.DWELLS)]
        self.actions = iter(experiment._balanced_actions(seed, self.dwell, SOURCE_FRAMES))
        self.scene = tt3_visual._SceneHistory()
        self.candidates = []

    def choose(self):
        return next(self.actions)

    def observe(self, image, preceding_action, reward):
        tracks = self.scene.update(image, preceding_action)
        if not 8 <= self.scene.tick < experiment.PROBE_FRAMES:
            return
        candidates, effects = tt3_visual._effects(self.scene, tracks)
        if effects:
            best = max(range(len(effects)), key=lambda i: _contrast(effects[i]))
            self.candidates.append((self.scene.tick, candidates[best], effects[best]))

    def result(self):
        predictions = []
        for tick, track, prefix in self.candidates:
            remaining = tt3_visual._SceneHistory()
            remaining.actions = deque((t, action) for t, action in self.scene.actions
                                      if tick < t <= tick + FUTURE_FRAMES)
            remaining.paths = {track.id: [(t, position) for t, position in self.scene.paths[track.id]
                              if tick < t <= tick + FUTURE_FRAMES]}
            suffix = remaining.effect(track)
            cosine = 0.
            repeat = False
            if suffix is not None:
                norm = prefix.features[2] * suffix.features[2]
                cosine = float(np.sum(prefix.effect * suffix.effect) / max(norm, 1e-9))
                repeat = cosine >= .50 and _contrast(suffix) >= .015
            predictions.append(dict(frame=tick, score=_contrast(prefix),
                                    repeated=repeat, future_cosine=cosine))
        return dict(dwell=self.dwell, predictions=predictions)


def _gather(seed, count):
    cases = experiment._source_cases(seed, count)
    agents = [_IndependentWindows(case.exploration) for case in cases]
    outcomes = tt3_pong._native_rallies(cases, agents,
        max_ticks=SOURCE_FRAMES * experiment.NATIVE_CAMERA_TICKS, opponent_speed=.35,
        camera_ticks=experiment.NATIVE_CAMERA_TICKS)
    return [dict(**outcome, **agent.result()) for outcome, agent in zip(outcomes, agents, strict=True)]


def _score(rows, threshold):
    probes = experiment.PROBE_FRAMES
    chosen = [next((point for point in row['predictions'] if point['score'] >= threshold), None)
              for row in rows]
    stops = [point for point in chosen if point is not None]
    correct = sum(point['repeated'] for point in stops)
    # Charge a false release more than a full camera window. A safe release costs its frame.
    safe_and_full_cost = sum(point['frame'] for point in stops if point['repeated'])
    safe_and_full_cost += (len(rows) - len(stops)) * probes
    cost = safe_and_full_cost + (len(stops) - correct) * (probes + 5)
    mean_stop = float(np.mean([point['frame'] if point else probes for point in chosen]))
    return dict(cutoff=None if not np.isfinite(threshold) else threshold, early_stops=len(stops),
                repeatable_stops=correct, mean_cost=cost / len(rows),
                mean_cameras=mean_stop,
                mean_native_ticks_saved=(probes - mean_stop) * experiment.NATIVE_CAMERA_TICKS)


def _learn(rows):
    scores = [_score(rows, cutoff) for cutoff in CUTOFFS]
    index = min(range(len(scores)), key=lambda i: (scores[i]['mean_cost'], -CUTOFFS[i]))
    return CUTOFFS[index], scores


def _sham(rows):
    rng = np.random.default_rng(973178563)
    labels = {frame: rng.permutation([point['repeated'] for row in rows for point in row['predictions']
                                     if point['frame'] == frame]).tolist()
              for frame in range(8, experiment.PROBE_FRAMES)}
    return [dict(row, predictions=[dict(point, repeated=labels[point['frame']].pop())
                for point in row['predictions']]) for row in rows]


def _stop_after_repeatable_evidence(threshold):
    def check(learner, binding):
        track = binding.controlled
        if binding.relevant is None or not track.observed:
            return False
        commands = [row.action for row in learner.history
                    if row.action is not None and track.id in row.positions]
        if len(commands) < learner.motor.actions:
            return False
        excitation = np.linalg.svd(commands, compute_uv=False)
        if excitation[-1] < .9:
            return False
        evidence = learner.binder._scene.effect(track)
        return evidence is not None and _contrast(evidence) >= threshold

    return check


def run(output, role_model, target_count=8):
    torch.set_num_threads(1)
    source_paths = [Path(__file__), Path(experiment.__file__)] + [Path(module.__file__) for module in
                 (tt3_arena, experiment.tt3_learning, tt3_pong, tt3_visual)]
    hashes = {path.name: sha256(path.read_bytes()).hexdigest() for path in source_paths}
    train = _gather(726348837, 48)
    check = _gather(143808025, 24)
    threshold, train_scores = _learn(train)
    sham, sham_scores = _learn(_sham(train))
    thresholds = dict(prior_stop=threshold, fixed_16=np.inf, simple_stop=.035, sham_stop=sham)
    stats = {arm: _score(check, value) for arm, value in thresholds.items()}
    rows = []
    for variant in tt3_arena.VARIANTS:
        for case in tt3_arena.heldout_cases(TARGET_SEED, target_count, variant=variant):
            evaluated = {}
            for arm, value in thresholds.items():
                if value not in evaluated:
                    stop = _stop_after_repeatable_evidence(value) if np.isfinite(value) else None
                    evaluated[value] = experiment._target_point(case, role_model, arm, None,
                                            stop_probing=stop, probe_arm='fixed_default')
                rows.append(dict(evaluated[value], arm=arm))
    report = dict(source_sha256=hashes, thresholds=stats, train_scores=train_scores,
                  sham_train_scores=sham_scores, source_train=train, source_check=check,
                  target_role_model=asdict(role_model), target_seed=TARGET_SEED,
                  target=experiment._target_summary(rows, ARMS),
                  target_compared=experiment._paired(rows, ARMS), target_runs=rows)
    output.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('role_model', type=Path)
    parser.add_argument('--target-count', type=int, default=8)
    args = parser.parse_args()
    state = json.loads(args.role_model.read_text())
    model = tt3_visual.RoleModel(**state.get('role_model', state))
    result = run(args.output, model, args.target_count)
    print(json.dumps(dict(thresholds=result['thresholds'], target=result['target'])))
