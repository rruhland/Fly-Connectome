"""Replay the longest scored all-live pupil point from a verified assessment."""

import argparse
from pathlib import Path

from . import tt23_sequence_coach_movie as movie
from .tt4_transfer_trial import _read
from .tt24_rally_bank import DIMS
from .tt24_rally_improve_trial import HELD


def render(source, mode):
    receipt = _read(source)
    cases = [case for case in receipt['cases'] if case is not None]
    if not cases or receipt['verified_tapes'] != sum(len(case['results']) for case in cases):
        raise ValueError('use a complete independently verified assessment')
    row = max((case['results'][mode] for case in cases), key=lambda visit: visit['longest'])
    movie.CONFIG = dict(movie.CONFIG, width=DIMS[0], height=DIMS[1])
    movie.render_tape(row, source, source.with_suffix('').with_suffix('.mp4'), branch=mode,
                      subtitle='Near arm drives every frame from visual and joint feedback',
                      terminal_caption='Strict referee and camera replay from the two recorded command streams')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=HELD)
    parser.add_argument('--mode', default='retina_rate')
    args = parser.parse_args()
    render(args.source, args.mode)
