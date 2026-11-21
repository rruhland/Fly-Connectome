"""Render the matched held rally through the arm's first-person camera."""

import argparse
from contextlib import ExitStack
from hashlib import sha256
from itertools import zip_longest
from pathlib import Path

import numpy as np
from PIL import Image

from .tt4_replay import caption, recording
from .tt4_transfer_trial import _read
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import prepare_point
from .tt6_render import cache_hall_stereo
from .tt8_service_arena import TT8ServiceArena
from .tt8_service_trial import PointInterface
from .tt13_rotary_trial import path


def frames(report, row, title):
    arena = TT8ServiceArena(**report['arena'])
    try:
        cache_hall_stereo(arena, reflections=True)
        wiring = TT6Wiring(arena, report['installation_seed'])
        interface = PointInterface(wiring, prepare_point(arena, row['phase']), seed=report['seed'], server=-1)
        image = interface.reset(seed=row['seed'])
        label = 'Opponent serves'
        yield caption(image[0], title, label, 0), arena.control_dt
        tape = np.asarray(row['actions'], dtype='<f8')
        if sha256(tape.tobytes()).hexdigest() != row['action_sha256']:
            raise ValueError('recorded commands differ from their checksum')
        for command in tape:
            image, _, ended, capped, info = interface.step(command)
            for event in info['events']:
                if event['kind'] == 'legal_return':
                    label = 'Arm lands a return' if event['side'] == 1 else 'Opponent lands a return'
            if ended or capped:
                label = 'Arm wins the point' if info['points_for'] else 'Opponent wins the point'
            yield caption(image[0], title, label, arena.legal_returns), arena.control_dt
        if arena.point_result() != row['evaluator'] or row['applied'] != interface.applied:
            raise ValueError(f"replayed point differed from receipt {row['seed']}")
    finally:
        arena.close()


def movie(folder, output, seed):
    reports = [_read(path(folder, f'{mode}-held-replication')) for mode in ('baseline', 'recenter')]
    if any(reports[0][key] != reports[1][key] for key in ('arena', 'seed', 'installation_seed')):
        raise ValueError('the receipts must share the same arena and schedules')
    rows = [next(row for row in report['points'] if row['seed'] == seed) for report in reports]
    if rows[0]['phase'] != rows[1]['phase'] or any(row['server'] != -1 for row in rows):
        raise ValueError('the replay requires a matched opponent serve')
    streams = [frames(report, row, title) for report, row, title in
               zip(reports, rows, ('Carried baseline', 'Recentering between strokes'), strict=True)]
    with ExitStack() as stack:
        append = None
        previous = [None, None]
        for pair in zip_longest(*streams):
            previous = [old if new is None else new for old, new in zip(previous, pair, strict=True)]
            (left, dt), (right, right_dt) = previous
            if dt != right_dt:
                raise ValueError('the two cameras must have the same playback rate')
            view = Image.new('RGB', (left.width + right.width, left.height))
            view.paste(left, (0, 0))
            view.paste(right, (left.width, 0))
            if append is None:
                append = stack.enter_context(recording(output, view, dt))
            else:
                append(view)
    print(f"verified both action tapes with returns {[row['player_returns'] for row in rows]}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, default=1378943403)
    args = parser.parse_args()
    movie(args.folder, args.output, args.seed)
