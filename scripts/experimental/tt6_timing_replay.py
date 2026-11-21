"""Replay a scored timing-trial point from the exact applied actuator commands."""

import argparse
from hashlib import sha256
from pathlib import Path
import tempfile

import numpy as np

from .tt4_replay import replay
from .tt4_transfer_trial import _read, _write


def render(path, output, condition, course_index, point_index):
    receipt = _read(path)
    if receipt['schema'] != 'tt6-visual-actuator-timing-v1':
        raise ValueError('expected a timing trial receipt')
    course = receipt['courses'][course_index]
    point = course['runs'][condition]['points'][point_index]
    requested = np.asarray(point['actions'], dtype='<f8')
    if sha256(requested.tobytes()).hexdigest() != point['action_sha256']:
        raise ValueError('requested action tape hash mismatch')
    applied = np.asarray(point['actuator_actions'], dtype='<f8')
    if applied.shape != requested.shape or len(applied) != point['steps']:
        raise ValueError('applied command tape length differs from the played point')
    # Native replay renders the recorded physics even when the scored run used
    # the accelerated renderer. TT4 replay verifies the full evaluator outcome.
    item = dict(point, actions=point['actuator_actions'], action_sha256=sha256(applied.tobytes()).hexdigest())
    one_point = dict(geometry=receipt['geometry'], courses={'first_person': [
        dict(schedule=course['schedule'], runs={condition: dict(points=[item])})]})
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as directory:
        fixture = Path(directory) / 'replay.json'
        _write(fixture, one_point)
        replay(fixture, output, condition)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--condition', default='continual')
    parser.add_argument('--course', type=int, default=0)
    parser.add_argument('--point', type=int, default=0)
    args = parser.parse_args()
    render(args.receipt, args.output, args.condition, args.course, args.point)


if __name__ == '__main__':
    main()
