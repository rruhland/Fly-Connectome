"""Offline diagnostic decoding of frozen T5 subtype rates."""

import json
from pathlib import Path

import numpy as np

from fly_connectome.data import checksum


BAR = Path('docs/experiments/2026-09-23-m1a-motion-front-end-extension-results.json')
DOT = Path('docs/experiments/2026-09-23-m1a-square-dot-results.json')
OUT = Path('docs/experiments/2026-09-23-m1a-motion-decode-results.json')
T5 = ('T5a', 'T5b', 'T5c', 'T5d')


def fit_centroids(samples):
    values = np.stack([features for _, features in samples])
    mean = values.mean(0)
    std = values.std(0).clip(min=.05)
    centroids = {label: np.stack([(features-mean)/std for kind, features in samples
                                  if kind == label]).mean(0)
                 for label in sorted({label for label, _ in samples})}
    return mean, std, centroids


def predict_direction(model, features):
    mean, std, centroids = model
    normalized = (features-mean)/std
    return min(centroids, key=lambda label: np.linalg.norm(normalized-centroids[label]))


def features(row, condition):
    return np.asarray([row['subtypes'][name][f'{condition}_spikes']
                       / row['subtypes'][name]['cells'] for name in T5])


def calibration_rows(result):
    rows = list(result['calibration'])
    if result.get('shape', 'bar') == 'bar':
        rows += [row for row in result['conditions']
                 if row['axis'] == 'horizontal' and row['center'] in (18, 46)
                 and row['polarity'] == 'off' and row['speed'] == 1]
    return rows


def test_rows(result):
    return [row for row in result['conditions'] if row['polarity'] == 'off'
            and ((row['axis'] == 'vertical' and row['center'] in (14, 18))
                 or (row['axis'] == 'horizontal' and row['center'] == 32))]


def fit_from_result(result):
    samples = []
    moving, static = [], []
    for row in calibration_rows(result):
        labels = ('down', 'up') if row['axis'] == 'vertical' else ('right', 'left')
        for condition, label in zip(('positive', 'negative'), labels):
            vector = features(row, condition)
            samples.append((label, vector))
            moving.append(float(vector.sum()))
        static.append(float(features(row, 'static').sum()))
    if len(samples) != 8:
        raise AssertionError('exactly two calibration trials per direction required')
    threshold = (max(static)+min(moving))/2
    return fit_centroids(samples), dict(
        moving_min=min(moving), static_max=max(static),
        threshold=threshold, separated=bool(max(static) < min(moving)))


def evaluate(result, model, intensity):
    trials = []
    for row in test_rows(result):
        labels = ('down', 'up') if row['axis'] == 'vertical' else ('right', 'left')
        for condition, truth in zip(('positive', 'negative'), labels):
            vector = features(row, condition)
            trials.append(dict(axis=row['axis'], center=row['center'],
                speed=row['speed'], truth=truth,
                prediction=predict_direction(model, vector),
                motion_energy=float(vector.sum()),
                moving_detected=bool(vector.sum() > intensity['threshold']),
                event_displacement=row['event_tracker'][f'{condition}_displacement']))
    static = [dict(axis=row['axis'], center=row['center'], speed=row['speed'],
                   motion_energy=float(features(row, 'static').sum()),
                   false_alarm=bool(features(row, 'static').sum()
                                    > intensity['threshold']))
              for row in test_rows(result)]
    return dict(trials=trials, static=static,
        correct=sum(row['prediction'] == row['truth'] for row in trials),
        total=len(trials),
        moving_detected=sum(row['moving_detected'] for row in trials),
        static_false_alarms=sum(row['false_alarm'] for row in static),
        event_tracker_correct=sum(row['event_displacement']
            == (row['speed'] if row['truth'] in ('down', 'right') else -row['speed'])
            for row in trials))


def main():
    bar, dot = (json.loads(path.read_text()) for path in (BAR, DOT))
    if not (bar['frozen_weights'] and dot['frozen_weights']):
        raise AssertionError('frozen-weight comparison required')
    bar_model, bar_intensity = fit_from_result(bar)
    dot_model, dot_intensity = fit_from_result(dot)
    result = dict(bar_sha256=checksum(BAR), dot_sha256=checksum(DOT),
        bar_intensity=bar_intensity, dot_intensity=dot_intensity,
        bar_with_bar_decoder=evaluate(bar, bar_model, bar_intensity),
        dot_with_dot_decoder=evaluate(dot, dot_model, dot_intensity),
        dot_with_bar_decoder=evaluate(dot, bar_model, bar_intensity))
    OUT.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: {metric: value for metric, value in row.items()
                             if metric not in ('trials', 'static')}
                      for key, row in result.items() if key.endswith('decoder')}),
          flush=True)


if __name__ == '__main__':
    main()
