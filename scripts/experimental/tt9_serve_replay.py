"""Compare two recorded strokes on the identical toss through the player's left camera."""
import argparse
from contextlib import ExitStack
from pathlib import Path

from PIL import Image

from .tt4_replay import caption, recording
from .tt4_transfer_trial import _read
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import prepare_point
from .tt8_service_arena import TT8ServiceArena
from .tt8_service_trial import PointInterface


def point_frames(report, row, title):
    arena = TT8ServiceArena(**report['arena'])
    try:
        wiring = TT6Wiring(arena, report['installation_seed'])
        interface = PointInterface(wiring, prepare_point(arena, row['phase']), server=row['server'], seed=report['seed'])
        frames = interface.reset(seed=row['seed'])
        label = 'Arm serves'
        yield caption(frames[0], title, label, 0)
        for command in row['actions']:
            frames, _, ended, capped, info = interface.step(command)
            for event in info['events']:
                if event['kind'] == 'serve_strike':
                    label = 'Arm strikes the toss'
                if event['kind'] == 'legal_serve':
                    label = 'Legal serve'
                if event['kind'] == 'legal_return':
                    label = 'Opponent returns' if event['side'] == -1 else 'Arm returns'
            if ended or capped:
                reason = info['reason'].replace('_', ' ')
                label = f"{'Arm wins' if info['points_for'] else 'Opponent wins' if info['points_against'] else 'No point'} · {reason}"
            yield caption(frames[0], title, label, arena.legal_returns)
        if arena.point_result() != row['evaluator'] or row['applied'] != interface.applied:
            raise ValueError('point replay diverged')
    finally:
        arena.close()


def movie(receipt, output, seeds):
    report = _read(receipt)
    sides = [('fixed_drive', 'Fixed stroke'), ('frozen_registration', 'Stroke after 12 practice serves')]
    sections = [(report, report['sections'][key]['rows'], title) for key, title in sides]
    pair_movie(sections, output, seeds)


def pair_movie(sections, output, seeds):
    report = sections[0][0]
    if any(other[k] != report[k] for other, _, _ in sections
           for k in ('arena', 'installation_seed', 'seed')):
        raise ValueError('the two players must share an installation and disturbance schedule')
    # Read the exact camera step from the arena, including XML physics options.
    arena = TT8ServiceArena(**report['arena'])
    try:
        dt = arena.control_dt
    finally:
        arena.close()
    with ExitStack() as stack:
        append = None
        for index, seed in enumerate(seeds, 1):
            shots = []
            for report, rows, title in sections:
                row = next(row for row in rows if row['seed'] == seed)
                shots.append(list(point_frames(report, row, f'{title}   |   Toss {index}   |   Half speed')))
            for tick in range(max(map(len, shots))):
                left, right = (shot[min(tick, len(shot) - 1)] for shot in shots)
                image = Image.new('RGB', (left.width + right.width, left.height))
                image.paste(left)
                image.paste(right, (left.width, 0))
                if append is None:
                    append = stack.enter_context(recording(output, image, dt * 2.))
                else:
                    append(image)
            for _ in range(round(.45 / (dt * 2.))):
                append(image)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('receipt', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('seeds', type=int, nargs='+')
    args = p.parse_args()
    movie(args.receipt, args.output, args.seeds)
