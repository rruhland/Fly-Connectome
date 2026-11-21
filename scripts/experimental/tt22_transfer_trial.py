"""Paired native-game test of transferred, role-free optical motion weights."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from fly_connectome.pong import Pong

from .tt3_pong import state_sha256
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_view import _in_court, _triangulate
from .tt20_camera_carry import SETTINGS, summarize
from .tt20_curriculum import OUTPUT as TT20, receiver
from .tt20_fixture import TT20Arena
from .tt20_natural import NaturalViewTransport
from .tt22_transfer import MotionWeights, Tracklets, VELOCITY, ViewMotion, patches


OUTPUT = ROOT/'artifacts/tt22'
ARENA = dict(SETTINGS, width=320, height=240, view_mode='fixed')
SOURCE = 1358403459
SOURCE_HOLD = 1009616734
PRACTICE = 889147313
VISION_HOLD = 1037194021
GAME_HOLD = 2177826788
CONDITIONS = ('pong', 'scratch', 'time_shuffled')
SOURCE_TICKS = 432
SOURCE_STRIDE = 3
OBSERVE_TICKS = 32
TARGET_PRIOR_STRENGTH = 180.
HEIGHT, WIDTH = 120, 160


def seedlist(seed, count):
    return np.random.default_rng(seed).integers(2**31, size=count).tolist()


def source_windows(seed, batch):
    """Use the native environment's rendered bitmap and ordinary rate commands."""
    rng = np.random.default_rng(seed)
    native = Pong(seedlist(seed, batch))
    streams = [Tracklets() for _ in range(batch)]
    windows, scores = [], np.zeros((batch, 2), dtype=int)
    for tick in range(SOURCE_TICKS + 1):
        if tick % SOURCE_STRIDE == 0:
            for tracker, image in zip(streams, native.render(HEIGHT, WIDTH).numpy(), strict=True):
                windows.extend(tracker.push(image.astype(np.float32)))
        if tick == SOURCE_TICKS:
            break
        if tick % 9 == 0:
            drive = torch.as_tensor(rng.choice((-1., 0., 1.), size=batch), dtype=torch.float32)
        outcome = native.step(drive)
        scored = outcome.scores.numpy()
        scores[:, 0] += scored > 0
        scores[:, 1] += scored < 0
        for index in np.flatnonzero(scored):
            streams[index] = Tracklets()
    return windows, dict(instances=batch, physics_actions=batch*SOURCE_TICKS, captured_frames=batch*(SOURCE_TICKS//SOURCE_STRIDE+1),
                         own_points=int(scores[:, 0].sum()), rival_points=int(scores[:, 1].sum()), windows=len(windows))


def grayscale(rgb):
    return rgb.reshape(HEIGHT, 2, WIDTH, 2, 3).mean(axis=(1, 3, 4), dtype=np.float32) / 255.


def vid_windows(arena, seed):
    """Read a new ordinary target opening without task or entity annotations."""
    arena.reset(seed=seed, server=-1, render=False)
    streams = {eye: Tracklets() for eye in arena.streams}
    windows, frames = [], 0
    for _ in range(OBSERVE_TICKS):
        for eye, rgb in arena.render_cameras().items():
            windows.extend(streams[eye].push(grayscale(rgb)))
            frames += 1
        _, _, stop, exhausted, _ = arena.step(np.zeros(arena.action_size), render=False)
        if stop or exhausted:
            break
    return windows, frames


def train(path, count):
    source, experience = source_windows(SOURCE, 24)
    old_validation, old_receipt = source_windows(SOURCE_HOLD, 8)
    x, y = patches(source)
    old_x, old_y = patches(old_validation)
    aligned, shuffled = MotionWeights(), MotionWeights()
    aligned.update(x, y)
    # Preserve source marginals, command experience, and sample count while
    # separating each continuation from its own preceding motion.
    wrong = np.random.default_rng(SOURCE+1).permutation(y)
    shuffled.update(x, wrong)
    initial = dict(pong=aligned.weights, scratch=VELOCITY, time_shuffled=shuffled.weights)
    current = {name: MotionWeights(weights, TARGET_PRIOR_STRENGTH) for name, weights in initial.items()}
    arena = TT20Arena(**ARENA)
    try:
        windows, target_valid = [], []
        for seed in seedlist(VISION_HOLD, 6):
            observed, frames = vid_windows(arena, seed)
            windows.extend(observed)
            target_valid.append(dict(seed=seed, frames=frames, tracklets=len(observed)))
        test_x, test_y = patches(windows)

        def qualify():
            return {name: dict(state=model.state(), pong_rmse=model.error(old_x, old_y),
                               so101_rmse=model.error(test_x, test_y)) for name, model in current.items()}

        curve = [dict(after_points=0, after_control_frames=0, conditions=qualify())]
        updated, total = [], 0
        for seed in seedlist(PRACTICE, count):
            trajectories, frames = vid_windows(arena, seed)
            total += frames//len(arena.streams)
            if trajectories:
                observations, future = patches(trajectories)
                for model in current.values():
                    model.update(observations, future)
            updated.append(dict(seed=seed, frames=frames, tracklets=len(trajectories)))
            curve.append(dict(after_points=len(updated), after_control_frames=total, conditions=qualify()))
            print('practice', len(updated), total, len(trajectories),
                  {key: round(val['so101_rmse'], 4) for key, val in curve[-1]['conditions'].items()}, flush=True)
    finally:
        arena.close()
    data = dict(schema='tt22-role-free-motion-practice-v1', arena=ARENA, source=experience,
                source_validation=old_receipt, source_models={'aligned': aligned.state(), 'time_shuffled': shuffled.state()},
                generators=dict(pong=SOURCE, pong_validation=SOURCE_HOLD, so101=PRACTICE, so101_validation=VISION_HOLD),
                train=updated, validation=target_valid, validation_tracklets=len(windows), curve=curve,
                target_strength=TARGET_PRIOR_STRENGTH, image_training_dimensions=[HEIGHT, WIDTH])
    _write(path, data)


class Station:
    def __init__(self):
        self.arena = TT20Arena(**ARENA)
        self.player = receiver(False, TT20)
        self.lenses = self.arena.lens_intrinsics()

    def close(self):
        self.player.player.close()
        self.arena.close()

    def point(self, seed, weights):
        role = -1
        self.arena.reset(seed=seed, server=role, render=False)
        self.player.reset(role)
        raw = NaturalViewTransport('fixed', lens_fov=self.lenses, flight=False).tracker()
        filters = {eye: ViewMotion(weights) for eye in self.arena.streams}
        actions, optical = [], []
        changed = filled = 0
        for tick in range(self.arena.max_steps):
            raw.update(self.arena.render_cameras(), tick=tick)
            old = None if raw.ball is None or not raw.ball.observed else np.asarray(raw.ball.position)
            detections = raw.detections['ball']
            projected, missing = {}, False
            for eye, predictor in filters.items():
                observation = detections.get(eye)
                center, inferred = predictor.step(tick, None if observation is None else observation.center)
                if center is not None:
                    projected[eye] = SimpleNamespace(center=center)
                    missing |= inferred
            position = old
            if detections and len(projected) >= 2:
                forecast, support = _triangulate(raw.fits, projected)
                if len(support) == 2 and _in_court('ball', forecast):
                    position = forecast
                    changed += int(old is not None and np.linalg.norm(forecast-old) > .001)
                    filled += int(missing)
            action = self.player.command(tick, position)
            optical.append(None if position is None else position.tolist())
            actions.append(action.tolist())
            _, reward, stop, capped, _ = self.arena.step(action, render=False)
            if stop or capped:
                break
        return dict(seed=seed, role=role, actions=actions, rgb=raw.digest.hexdigest(), optical=optical, changed_positions=changed,
                    missing_view_fills=filled, outcome=self.arena.point_result(), events=self.arena.events.copy(), reward=reward,
                    plans=self.player.player.plans)


def replay(rows):
    arena = TT20Arena(**ARENA)
    try:
        for row in rows:
            arena.reset(seed=row['seed'], server=row['role'], render=False)
            for action in row['actions']:
                _, reward, stop, capped, _ = arena.step(action, render=False)
            if (not (stop or capped) or reward != row['reward'] or arena.point_result() != row['outcome'] or
                    state_sha256(arena.events) != state_sha256(row['events'])):
                raise ValueError(f"the physical tape changed at {row['seed']}")
        return len(rows)
    finally:
        arena.close()


def held(path, training, count):
    learned = _read(training)
    weights = {name: MotionWeights.restore(row['state']).weights for name, row in learned['curve'][-1]['conditions'].items()}
    seeds = seedlist(GAME_HOLD, count)
    previous = {row['seed'] for key in ('train', 'validation') for row in learned[key]}
    if len(set(seeds)) != count or previous.intersection(seeds):
        raise ValueError('held receiving starts overlap optical training or validation')
    data = dict(schema='tt22-role-free-motion-held-v1', arena=ARENA, training_sha256=_hash(training),
                seeds=seeds, generator=GAME_HOLD, applied_weights={k: v.tolist() for k, v in weights.items()},
                rows={name: [] for name in CONDITIONS})
    if path.exists():
        prior = _read(path)
        if any(prior[key] != val for key, val in data.items() if key != 'rows'):
            raise ValueError('held restart uses a different frozen plan')
        data['rows'] = prior['rows']
    named = list(CONDITIONS)
    station = Station()
    try:
        for index, seed in enumerate(seeds):
            for name in named[index % len(named):] + named[:index % len(named)]:
                rows = data['rows'][name]
                if len(rows) <= index:
                    rows.append(station.point(seed, weights[name]))
                    row = rows[-1]
                    print('held', index+1, name, seed, row['reward'], row['outcome']['legal_returns'],
                          row['changed_positions'], row['missing_view_fills'], flush=True)
                    _write(path, data)
    finally:
        station.close()
    data['scores'] = {name: summarize(rows) for name, rows in data['rows'].items()}
    data['replayed'] = replay([row for rows in data['rows'].values() for row in rows])
    _write(path, data)
    print('held final', data['scores'], 'replayed', data['replayed'], flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('train', 'held', 'all', 'verify'))
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--practice', type=int, default=10)
    parser.add_argument('--starts', type=int, default=16)
    args = parser.parse_args()
    args.folder.mkdir(exist_ok=True, parents=True)
    training = args.folder/'transfer-practice.json.gz'
    trial = args.folder/'transfer-held.json.gz'
    if args.stage in ('train', 'all'):
        train(training, args.practice)
    if args.stage in ('held', 'all'):
        held(trial, training, args.starts)
    if args.stage == 'verify':
        print('replayed', replay([row for rows in _read(trial)['rows'].values() for row in rows]))
