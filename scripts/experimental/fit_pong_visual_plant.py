"""Fit the visual plant from chronological ordinary-control camera recordings."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from pong_continuous_motor_observer import VisualPlant


def fit_plant(recordings, training_games):
    """Use requested IDs, observed positions and elapsed camera intervals."""
    if not 0 < training_games < len(recordings):
        raise ValueError('training and holdout both require games')
    horizon = max(len(recording['rows']) for recording in recordings)
    shape = (len(recordings), horizon)
    positions = np.full(shape, np.nan)
    actions = np.ones(shape, dtype=np.int64)
    elapsed = np.full(shape, .02)
    for index, recording in enumerate(recordings):
        for step, row in enumerate(recording['rows']):
            if row['position'] is not None:
                positions[index, step] = row['position']
            actions[index, step] = row['action']
            elapsed[index, step] = row['elapsed']
    observed = np.isfinite(positions)
    training = observed & (np.arange(len(recordings))[:, None] < training_games)
    holdout = observed & ~training
    if not training.any() or not holdout.any():
        raise ValueError('observed training and holdout camera positions required')

    def predict(parameters):
        plant = VisualPlant(*parameters)
        state = np.zeros((len(recordings), 3))
        prediction = np.zeros_like(positions)
        gains = plant.normal_gains
        for step in range(horizon - 1):
            interval = elapsed[:, step]
            for duration in np.unique(interval):
                selected = interval == duration
                transition, drive = plant.interval(float(duration))
                state[selected] = state[selected] @ transition.T + gains[actions[selected, step], None] * drive
            position, velocity = state[:, :2].T
            velocity[abs(position) > plant.travel_limit] = 0.
            np.clip(position, -plant.travel_limit, plant.travel_limit, out=position)
            np.clip(velocity, -plant.speed_limit, plant.speed_limit, out=velocity)
            prediction[:, step + 1] = position
        return prediction

    result = least_squares(lambda parameters: (predict(parameters) - positions)[training],
        [.065, .24, .72, .68, .40, 0.],
        bounds=([.01, .14, .3, .5, .36, -.15], [.13, .5, 1.4, .85, .45, .15]),
        loss='soft_l1', f_scale=.007, max_nfev=50)
    plant = VisualPlant(*result.x)
    residuals = predict(result.x) - positions
    report = {}
    for label, mask in [('training', training), ('holdout', holdout)]:
        errors = residuals[mask]
        report[label] = dict(observations=int(mask.sum()), rms=float(np.sqrt(np.mean(errors ** 2))),
                             absolute_error_p95=float(np.quantile(abs(errors), .95)))
    return plant, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('recordings', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--training-games', type=int, required=True)
    args = parser.parse_args()
    recording_bytes = args.recordings.read_bytes()
    samples = json.loads(recording_bytes)
    plant, report = fit_plant(samples, args.training_games)
    with args.output.open('x') as stream:
        json.dump(dict(plant=asdict(plant), report=report, public_starts=[sample['seed'] for sample in samples],
                       training_games=args.training_games,
                       recordings_sha256=hashlib.sha256(recording_bytes).hexdigest()), stream, indent=2)
