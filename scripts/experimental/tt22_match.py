"""Play a complete native SO-101 match with the carried optical rally correction."""

import argparse
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt8_service_arena import ServiceScore
from .tt20_natural import NaturalViewTransport
from .tt21_online_strokes import FACTORY, STANDARD_PICK
from .tt21_service_drill import END, OPTICAL_TICKS, tape
from .tt22_rally import ReturnMemory
from .tt22_rally_trial import CONFIG, OUTPUT, RallyArena, Station, alternating, opponent


SEED = 403607823
CHECKPOINT = OUTPUT/'rally-learning-checkpoint.json.gz'
HANDOVER = END + 8


def step_recorded(arena, command):
    """Record the rate command the live rival actually supplies to the real motors."""
    name = '_aim' if arena._rival_attempt else '_command_to'
    original = getattr(arena, name)
    captured = []

    def record(*args):
        rates = original(*args)
        captured.append(np.asarray(rates).tolist())
        return rates

    setattr(arena, name, record)
    try:
        _, reward, done, cap, _ = arena.step(command, render=False)
    finally:
        setattr(arena, name, original)
    if len(captured) != 1:
        raise ValueError('expected one physical rival rate vector per control frame')
    return reward, done, cap, captured[0]


class MatchStation(Station):
    def __init__(self, profile, serve):
        super().__init__(profile['spread'], profile['pace'], profile['rival'])
        self.serve = np.asarray(serve, dtype=float)
        spec = self.arena.lens_intrinsics()['left']
        self.service_optics = NaturalViewTransport('left', lens_fov={'left': spec}, flight=False)

    def play(self, seed, role, memory):
        arena, player = self.arena, self.player
        arena.reset(seed=seed, server=role, render=False)
        player.reset()
        player.outbound = role == -1
        player.memory, player.update_memory = memory, True
        context = dict(before=len(memory.evidence), bias_before=memory.bias().tolist())
        tracker = self.optics.tracker()
        serving = self.service_optics.tracker() if role == 1 else None
        actions, rival_tape, sightings, toss_x = [], [], [], []
        serve_tape = None
        for tick in range(arena.max_steps):
            frames = arena.render_cameras()
            tracker.update(frames, tick=tick)
            ball = tracker.ball
            position = None if ball is None or not ball.observed or tracker.sources.get('ball_range') != 'stereo' else np.asarray(ball.position)
            if position is not None:
                sightings.append((tick, position.tolist()))
            if serving is not None and tick < HANDOVER:
                if tick in OPTICAL_TICKS:
                    serving.update({'left': frames['left']}, tick=tick)
                    toss = serving.ball
                    if toss is not None and toss.observed:
                        toss_x.append(float(toss.position[0]))
                if tick == OPTICAL_TICKS[-1]:
                    serve_tape = tape(self.serve, float(np.median(toss_x)) if toss_x else 0.)
                action = np.zeros(6) if serve_tape is None else serve_tape[tick]
                player.history.append((tick, position))
                if position is not None and position[1] > .07:
                    player.outbound = True
                player.sim.twin.advance(action)
            else:
                action = player.command(tick, position)
            actions.append(action.tolist())
            reward, done, cap, far_action = step_recorded(arena, action)
            rival_tape.append(far_action)
            if done or cap:
                break
        context.update(after=len(memory.evidence), bias_after=memory.bias().tolist())
        return dict(seed=seed, role=role, reward=reward, actions=actions, rival_actions=rival_tape,
                    longest=alternating(arena.events), optical=sightings, shots=player.shots,
                    outcome=arena.point_result(), events=arena.events.copy(), learning=context,
                    handover=HANDOVER if role == 1 and len(actions) > HANDOVER else 0 if role == -1 else None,
                    toss_x=toss_x, rgb=tracker.digest.hexdigest(),
                    service_rgb=None if serving is None else serving.digest.hexdigest(),
                    rival=self.rival.decisions.copy())


def summarize(rows):
    return dict(points=len(rows), lengths=[row['longest'] for row in rows],
                longest=max(row['longest'] for row in rows),
                ten_or_more=sum(row['longest'] >= 10 for row in rows),
                legal_returns={side: sum(row['outcome'][key] for row in rows)
                               for side,key in (('pupil','legal_returns'), ('rival','rival_returns'))},
                legal_serves={side: sum(row['outcome']['legal_serve'] for row in rows if row['role'] == role)
                              for side,role in (('pupil',1), ('rival',-1))},
                serve_attempts={side: sum(row['role'] == role for row in rows)
                                for side,role in (('pupil',1), ('rival',-1))})


def verify(report):
    arena = RallyArena(report['spread'], report['pace'])
    opponent(arena, report['rival'])
    score = ServiceScore(first_server=report['first_server'])
    schedule = np.random.default_rng(report['seed'])
    try:
        for row in report['points']:
            expected = int(schedule.integers(2**31)), score.server, list(score.score)
            if score.finished or (row['seed'], row['role'], row['score_before']) != expected:
                raise ValueError('the observed game changed its ordinary service schedule')
            arena.reset(seed=row['seed'], server=row['role'], render=False)
            live_aim, live_direct = arena._aim, arena._command_to
            try:
                for action, rival in zip(row['actions'], row['rival_actions'], strict=True):
                    rates = np.asarray(rival, dtype=float)
                    arena._aim = arena._command_to = lambda *args, rates=rates: rates
                    _, reward, _, _, _ = arena.step(action, render=False)
            finally:
                arena._aim, arena._command_to = live_aim, live_direct
            if (row['outcome'] != arena.point_result() or row['reward'] != reward
                    or json.dumps(row['events']) != json.dumps(arena.events)):
                raise ValueError(f"physical two-arm replay diverged at seed {row['seed']}")
            if list(score.record(arena.point_result())) != row['score_after']:
                raise ValueError('the replayed scoreboard changed')
        if not score.finished or list(score.score) != report['score']:
            raise ValueError('the final match score changed')
    finally:
        arena.close()
    return len(report['points'])


def run(output, seed, checkpoint):
    profile = _read(checkpoint)
    serve = _read(FACTORY)['programs'][STANDARD_PICK]
    station = MatchStation(profile, serve)
    memory = ReturnMemory(profile['evidence'])
    score = ServiceScore(first_server=-1)
    schedule = np.random.default_rng(seed)
    practice = {row['seed'] for row in profile['practice']}
    report = dict(schema='tt22-optical-stock-anchor-game-1', seed=seed, first_server=-1, target=11,
                  spread=profile['spread'], pace=profile['pace'], rival=profile['rival'], arena=CONFIG,
                  source_sha256=_hash(checkpoint), serve_sha256=_hash(FACTORY), serve_program=STANDARD_PICK,
                  checkpoint_before=memory.evidence.copy(), points=[], complete=False)
    try:
        for _ in range(90):
            point = int(schedule.integers(2**31))
            if point in practice:
                raise ValueError('the full game must use a fresh point schedule')
            before = list(score.score)
            row = station.play(point, score.server, memory)
            after = list(score.record(row['outcome']))
            report['points'].append(dict(row, score_before=before, score_after=after))
            report.update(score=after, checkpoint_after=memory.evidence.copy(), stats=summarize(report['points']))
            _write(output, report)
            print('point', len(report['points']), row['role'], row['outcome']['legal_serve'], row['longest'], after,
                  row['outcome']['reason'], flush=True)
            if score.finished:
                break
    finally:
        station.close()
    if not score.finished:
        raise RuntimeError('the fixed schedule exhausted its 90 point allowance')
    report.update(complete=True, verified=verify(report))
    _write(output, report)
    print('final', report['score'], report['stats'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=SEED)
    parser.add_argument('--checkpoint', type=Path, default=CHECKPOINT)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    args.folder.mkdir(parents=True, exist_ok=True)
    if args.verify:
        print('replayed', verify(_read(args.verify)))
    else:
        run(args.folder/f'match-{args.seed}.json.gz', args.seed, args.checkpoint)
