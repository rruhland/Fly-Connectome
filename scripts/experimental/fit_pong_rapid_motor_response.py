"""Fit the rapid motor observer on fresh ordinary Pong public camera histories."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import time

import numpy as np
from scipy.optimize import least_squares
import torch

from fly_connectome.control.environment import PongCamera
from fly_connectome.control.session import model_bytes, source_signature
from fly_connectome.vision import VisualStateEncoder, load_default
from pong_relational_feedback import RelationalPolicy
from pong_relational_persistent_binding import PersistentRelationalPolicy
from pong_visual_button_effects import VisualButtonEffects, VisualIntentController
from public_entity_pong_native import snapshot
from run_pong_inverse_cycle import save, sha
from run_pong_surprise_recovery import MODULUS, POLICY


START_SEED = 52739194
FIT_GAMES = 12
CHECK_GAMES = 4
LIMIT_TICKS = 720
INITIAL_GUESS = [.065, .24, .72, .68, .40, 0.]
FIT_BOUNDS = ([.01, .14, .3, .5, .36, -.15], [.13, .5, 1.4, .85, .45, .15])


def schedule():
    rng = np.random.default_rng(START_SEED)
    seeds = []
    while len(seeds) < FIT_GAMES + CHECK_GAMES:
        seed = int(rng.integers(1, MODULUS))
        if seed not in seeds and ((seed * 48271 + 1) % MODULUS) % 2 == len(seeds) % 2:
            seeds.append(seed)
    return seeds


def collect(job):
    seed, checkpoint = job
    torch.set_num_threads(1)
    learner = VisualButtonEffects.from_state(checkpoint).frozen_branch()
    policy = PersistentRelationalPolicy(RelationalPolicy(*POLICY['choices'],
        POLICY['coefficients'], POLICY['scales']), POLICY['require_first_hit'])
    controller = VisualIntentController(policy, learner)
    camera = PongCamera(seed, max_episode_ticks=LIMIT_TICKS, opponent_speed=.65)
    vision, encoder = load_default(), VisualStateEncoder(64, 64, .02)
    frame = camera.game.render(64, 64)[0].clone()
    previous = torch.zeros_like(frame)

    def observe(current, prior):
        changes = torch.stack((prior & ~current, ~prior & current)).float()
        return snapshot(encoder.encode_entities(*vision.track_entities(changes, current.float())))

    public = observe(frame, previous)
    rows = []
    while True:
        request, _ = controller.choose(public, 1)
        position = controller._motor_position(public)
        result = camera.step(request)
        rows.append(dict(position=position, request=request, elapsed_seconds=result.elapsed_seconds))
        if result.terminated or result.truncated:
            break
        public = observe(result.frame, frame)
        frame = result.frame
        controller.observe(public_entities=public, own_hits=result.hits, reward=result.reward,
            terminated=result.terminated, truncated=result.truncated,
            elapsed_seconds=result.elapsed_seconds)
    return dict(seed=seed, rows=rows)


def targets(traces):
    length = max(len(trace['rows']) for trace in traces)
    shape = len(traces), length
    positions = np.full(shape, np.nan)
    elapsed = np.zeros(shape)
    requests = np.ones(shape, int)
    for i, trace in enumerate(traces):
        for j, row in enumerate(trace['rows']):
            if row['position'] is not None:
                positions[i, j] = row['position']
            elapsed[i, j] = row['elapsed_seconds']
            requests[i, j] = row['request']
    return positions, elapsed, requests


def predict(parameters, elapsed, requests):
    lag, coast, gain, speed_limit, position_limit, neutral = parameters
    position = np.zeros(len(elapsed))
    velocity = np.zeros_like(position)
    response = np.zeros_like(position)
    predictions = [position.copy()]
    for dt, request in zip(elapsed.T[:-1], requests.T[:-1], strict=True):
        lag_retained = np.exp(-dt / lag)
        velocity_retained = np.exp(-dt / coast)
        response_velocity = lag / (lag - coast) * (lag_retained - velocity_retained)
        velocity_position = coast * (1 - velocity_retained)
        response_position = lag / (lag - coast) * (lag * (1 - lag_retained) - velocity_position)
        drive = np.where(request == 1, neutral, (1 - request) * gain)
        position += velocity_position * velocity + response_position * response
        position += (dt - velocity_position - response_position) * drive
        velocity = velocity_retained * velocity + response_velocity * response
        velocity += (1 - velocity_retained - response_velocity) * drive
        np.clip(velocity, -speed_limit, speed_limit, out=velocity)
        response = lag_retained * response + (1 - lag_retained) * drive
        velocity[abs(position) > position_limit] = 0.
        np.clip(position, -position_limit, position_limit, out=position)
        predictions.append(position.copy())
    return np.asarray(predictions).T


def fit(traces):
    positions, elapsed, requests = targets(traces)
    valid = np.isfinite(positions)
    train = valid & (np.arange(len(traces))[:, None] < FIT_GAMES)
    check = valid & ~train

    def residual(parameters):
        return predict(parameters, elapsed, requests) - positions

    result = least_squares(lambda values: residual(values)[train], INITIAL_GUESS,
                           bounds=FIT_BOUNDS, loss='soft_l1', f_scale=.007, max_nfev=50)
    if not result.success:
        raise RuntimeError(result.message)
    errors = residual(result.x)
    return dict(
        parameters=dict(zip(('response_seconds', 'coast_seconds', 'gain', 'speed_limit',
                             'position_limit', 'neutral'), result.x.tolist(), strict=True)),
        errors={name: dict(samples=int(mask.sum()), rms=float(np.sqrt(np.mean(errors[mask] ** 2))),
                           absolute_percentiles={str(q): float(np.percentile(abs(errors[mask]), q))
                                                 for q in (50, 75, 90, 95, 98, 99, 100)})
                for name, mask in [('fit', train), ('check', check)]})


def run(directory, checkpoint_path, workers):
    model_bytes()
    directory.mkdir(parents=True, exist_ok=False)
    acquired = json.loads(checkpoint_path.read_text())
    seeds = schedule()
    save(directory / 'schedules.json', dict(fit=seeds[:FIT_GAMES], check=seeds[FIT_GAMES:]))
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        traces = list(pool.map(collect, [(seed, acquired) for seed in seeds]))
    trace_path = directory / 'public-observations.json'
    save(trace_path, traces)
    report = dict(**fit(traces), source=source_signature(), checkpoint_sha256=sha(checkpoint_path),
                  script_sha256=sha(Path(__file__)), public_observations_sha256=sha(trace_path),
                  transition_cameras=sum(len(trace['rows']) for trace in traces),
                  initial_cameras=len(traces), wall_seconds=time.monotonic() - started)
    save(directory / 'summary.json', report)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    print(json.dumps(run(args.directory, args.checkpoint, args.workers), sort_keys=True))
