"""One fixed learned session with a visually unchanged Breakout actuator intervention."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch

from arcade_alternation import AlternatingSession
from breakout_arena import breakout_schedule
from run_arcade_alternation import MAX_TICKS, digest, identify_board, native_arena, play, save, source_digest, summarize


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / 'artifacts/m2/2026-11-20-arcade-alternation.zip'
REGISTRATION = ROOT / 'artifacts/m2/2026-11-20-arcade-deployment-stress-plan.json'
ENDPOINT = 'confirmation/conflicting/guarded-final.json'
BLOCKS = ('pong', 'breakout', 'pong', 'breakout', 'pong', 'breakout')
PONG_SEED = 1891830531
BRICK_SEED = 1295077863


def register(path):
    with ZipFile(ARCHIVE) as archive:
        prior = set()
        prior_pixels = set()
        pong_starts = set()
        for name in archive.namelist():
            if name.endswith('/board-provenance.json'):
                content = json.loads(archive.read(name))
                boards = content['practice'] + [
                    row for key in ('evaluations', 'continuation')
                    for entries in content.get(key, {}).values() for row in entries]
                prior.update(digest(row['layout']) for row in boards)
                prior_pixels.update(row['initial_frame_sha256'] for row in boards)
            if name.endswith('/plan.json'):
                plan = json.loads(archive.read(name))
                pong_starts.update(row['parameters']['seed'] for row in (*plan['cases'], *plan.get('continuation', ()))
                                   if row['game'] == 'pong')
        demo = json.loads(archive.read('diagnostics/full-length-continuation.json'))
        prior.add(digest(demo['board']['layout']))
        prior_pixels.add(demo['board']['initial_frame_sha256'])
        pong_starts.add(demo['lives'][1]['case']['parameters']['seed'])
        start = json.loads(archive.read(ENDPOINT))['shared']
    rng = np.random.default_rng(PONG_SEED)
    breakout = iter(breakout_schedule(BRICK_SEED, 12))
    cases = []
    for block, game in enumerate(BLOCKS):
        mapping, delay = ((2, 1, 0), 6) if block == 3 else ((0, 1, 2), 0)
        for index in range(4):
            if game == 'pong':
                while True:
                    seed = int(rng.integers(1, 2147483647))
                    direction = ((seed * 48271 + 1) % 2147483647) % 2
                    if seed not in pong_starts and direction == (index + 1) % 2:
                        pong_starts.add(seed)
                        break
                parameters = dict(seed=seed, opponent_speed=.35)
            else:
                parameters = asdict(next(breakout))
            case = dict(case=len(cases), block=block, index=index, game=game,
                        parameters=parameters, mapping=mapping, delay=delay)
            if game == 'breakout':
                board = identify_board(native_arena(case, MAX_TICKS))
                layout, image = digest(board['layout']), board['initial_frame_sha256']
                if len(board['layout']) != 9 or layout in prior or image in prior_pixels:
                    raise ValueError('a deployment board repeats a previously played layout or frame')
                prior.add(layout)
                prior_pixels.add(image)
                case['reserved_board'] = board
            cases.append(case)
    if path.exists():
        raise FileExistsError(path)
    save(path, dict(checkpoint=ENDPOINT, checkpoint_sha256=digest(start), archive_sha256=source_digest(ARCHIVE),
                    max_ticks=MAX_TICKS, cases=cases, fixed_options=dict(recall_roles=True, recall_timing=True,
                    causal_pursuit=True)))


def run(path, output):
    torch.set_num_threads(1)
    plan = json.loads(path.read_text())
    if source_digest(ARCHIVE) != plan['archive_sha256']:
        raise ValueError('registered checkpoint source differs')
    with ZipFile(ARCHIVE) as archive:
        start = json.loads(archive.read(plan['checkpoint']))['shared']
    session = AlternatingSession.from_state(start)
    if digest(session.state()) != plan['checkpoint_sha256'] or not (
            session.recall_roles and session.recall_timing and session.core.causal_pursuit):
        raise ValueError('registered fixed checkpoint differs')
    output.mkdir(parents=True, exist_ok=False)
    save(output / 'initial.json', session.state())
    save(output / 'registered-plan.json', dict(plan, plan_file_sha256=source_digest(path)))
    rows = []
    last = digest(start)
    for case in plan['cases']:
        row = play(session, case, plan['max_ticks'])
        if last != row['initial_state_sha256']:
            raise AssertionError('a life interrupted the carried state chain')
        if case['game'] == 'breakout' and row['initial_frame_sha256'] != case['reserved_board']['initial_frame_sha256']:
            raise AssertionError('live board differs from the registered scene')
        last = row['final_state_sha256']
        rows.append(row)
        save(output / 'lives.json', rows)
        if case['index'] == 3:
            print(f"block {case['block'] + 1}", summarize(rows[-4:]), flush=True)
    save(output / 'final.json', session.state())
    save(output / 'summary.json', [dict(game=game, **summarize(rows[i * 4:(i + 1) * 4]))
                                  for i, game in enumerate(BLOCKS)])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=('register', 'run'))
    parser.add_argument('--plan', type=Path, default=REGISTRATION)
    parser.add_argument('--output', type=Path)
    options = parser.parse_args()
    if options.operation == 'register':
        register(options.plan)
    else:
        if options.output is None:
            parser.error('--output is required with run')
        run(options.plan, options.output)
