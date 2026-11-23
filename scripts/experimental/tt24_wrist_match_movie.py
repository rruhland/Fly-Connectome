"""Replay a wrist-only game with the measured wrist stream visible."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw

from . import tt23_sequence_coach_movie as movie
from .tt4_transfer_trial import _read
from .tt23_sequence import SequenceArena
from .tt24_wrist_match import DIMS, RECEIPT


class WristReplay(SequenceArena):
    def __init__(self, **config):
        super().__init__(**config)
        self.streams = ('wrist',)

    def render_cameras(self, mode=None):
        self.onboard = super().render_cameras(mode)['wrist']
        return dict(left=self.onboard, right=self.onboard)


def render(source, out):
    receipt = _read(source)
    if receipt['verified_tapes'] != sum(len(case['results']) for case in receipt['cases']):
        raise ValueError('wrist assessment replay is incomplete')
    row = max((case['results']['wrist'] for case in receipt['cases']), key=lambda run: run['longest'])
    movie.CONFIG = dict(movie.CONFIG, width=DIMS[0], height=DIMS[1])
    movie.SequenceArena = WristReplay
    frame = movie.frame

    def instrumented(arena, camera, tick, subtitle, footer='Both arms replaying their recorded control commands'):
        canvas = Image.frombytes('RGB', movie.SIZE, frame(arena, camera, tick, subtitle, footer))
        draw = ImageDraw.Draw(canvas)
        onboard = Image.fromarray(arena.onboard).resize((160,120), Image.Resampling.BILINEAR)
        draw.rectangle((506, 79, 674, 224), fill=movie.BG)
        draw.text((513, 84), 'Onboard wrist RGB', fill=movie.INK, font=movie.font(12))
        canvas.paste(onboard, (510, 101))
        return canvas.tobytes()

    movie.frame = instrumented
    movie.render_tape(row, source, out, branch='wrist',
                      subtitle='Near arm acts from live wrist RGB and its own joints',
                      terminal_caption='Verified referee and wrist-image replay')


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--source', type=Path, default=RECEIPT)
    cli.add_argument('--output', type=Path, default=RECEIPT.with_suffix('').with_suffix('.mp4'))
    args = cli.parse_args()
    render(args.source,args.output)
