"""Play a scored match with the branch-practice motor selector and standard serves."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt8_service_arena import ServiceScore
from .tt19_flight import FlightLaw
from .tt19_view_full_game import digest_json
from .tt20_natural import NaturalViewTransport
from .tt21_service_drill import OPTICAL_TICKS, tape
from .tt22_branch_motor import BranchMotor, MOTORS, OutcomeScorer
from .tt22_branch_motor_student import PRACTICE
from .tt22_match import FACTORY, HANDOVER, STANDARD_PICK, step_recorded, summarize
from .tt22_opponent import RallyArena, install_rally_opponent
from .tt22_rally_trial import CONFIG, OUTPUT, alternating


SEED = 853884001
MAX_POINTS = 90


class MatchPoint:
    def __init__(self, profile):
        self.arena = RallyArena(**CONFIG)
        self.rival = install_rally_opponent(self.arena, target_profile=profile)
        lenses = self.arena.lens_intrinsics()
        self.optics = NaturalViewTransport('fixed', lens_fov=lenses, law=FlightLaw(), flight=False, associate=True)
        self.service_optics = NaturalViewTransport('left', lens_fov={'left': lenses['left']}, flight=False)
        self.serve = np.asarray(_read(FACTORY)['programs'][STANDARD_PICK], dtype=float)

    def play(self, seed, role, learner):
        arena = self.arena
        arena.reset(seed=seed, server=role, render=False)
        learner.reset()
        tracker = self.optics.tracker()
        serving = self.service_optics.tracker() if role == 1 else None
        actions, rival_actions, optical, toss = [], [], [], []
        serve_tape = None
        for tick in range(arena.max_steps):
            frames = arena.render_cameras()
            tracker.update(frames, tick=tick)
            ball = tracker.ball
            position = None
            if ball is not None and ball.observed and tracker.sources.get('ball_range') == 'stereo':
                position = np.asarray(ball.position)
                optical.append((tick, position.tolist()))
            if serving is not None and tick < HANDOVER:
                if tick in OPTICAL_TICKS:
                    serving.update({'left': frames['left']}, tick=tick)
                    observed = serving.ball
                    if observed is not None and observed.observed:
                        toss.append(float(observed.position[0]))
                    if tick == OPTICAL_TICKS[-1]:
                        serve_tape = tape(self.serve, float(np.median(toss)) if toss else 0.)
                command = np.zeros(6) if serve_tape is None else serve_tape[tick]
                learner.cue.observe(tick, position)
                learner.cue.advance(command)
                # This branch begins with the learner's own serve. Its first
                # incoming ball comes from a rally stroke, not a release.
                learner.cue.previous = 1
            else:
                command = learner.command(tick, position)
            actions.append(command.tolist())
            reward, done, capped, rival = step_recorded(arena, command)
            rival_actions.append(rival)
            if done or capped:
                break
        events = list(arena.events)
        return dict(seed=seed, role=role, reward=reward, actions=actions, rival_actions=rival_actions,
                    optical=optical, events=events, longest=alternating(events), shots=learner.shots,
                    outcome=arena.point_result(), rgb=tracker.digest.hexdigest(), toss=toss,
                    service_rgb=None if serving is None else serving.digest.hexdigest())

    def close(self):
        self.arena.close()


def scoreboard(rows, seeds):
    score = ServiceScore(first_server=-1)
    for row, seed in zip(rows, seeds[:len(rows)], strict=True):
        if score.finished or (row['seed'], row['role'], row['score_before']) != (seed, score.server, list(score.score)):
            raise ValueError(f"service schedule changed at {row['seed']}")
        if list(score.record(row['outcome'])) != row['score_after']:
            raise ValueError(f"score changed at {row['seed']}")
    return score


def verify(rows, seeds):
    if not scoreboard(rows, seeds).finished:
        raise ValueError('verification requires a completed game')
    arena = RallyArena(**CONFIG)
    original_aim, original_direct = arena._aim, arena._command_to
    try:
        for row in rows:
            arena._aim, arena._command_to = original_aim, original_direct
            arena.reset(seed=row['seed'], server=row['role'], render=False)
            for index, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True), start=1):
                rates = np.asarray(far)
                arena._aim = arena._command_to = lambda *args, rates=rates: rates
                _, reward, done, capped, _ = arena.step(near, render=False)
                if bool(done or capped) != (index == len(row['actions'])):
                    raise ValueError(f"point length changed at {row['seed']}:{index}")
            if (arena.point_result() != row['outcome'] or reward != row['reward']
                    or digest_json(arena.events) != digest_json(row['events'])):
                raise ValueError(f"physical result changed at {row['seed']}")
    finally:
        arena.close()
    return len(rows)


def run(checkpoint, output, schedule=SEED, *, scorer=OutcomeScorer, policy='contextual'):
    weights = _read(checkpoint)
    if weights['motors'] != [list(motor) for motor in MOTORS]:
        raise ValueError('practice motor library and live library disagree')
    seeds = np.random.default_rng(schedule).integers(2**31, size=MAX_POINTS).tolist()
    if len(set(seeds)) != len(seeds) or set(seeds) & {row[1] for row in weights['donors']}:
        raise ValueError('match schedule must contain unique nonpractice seeds')
    metadata = dict(schema='tt23-motor-game-1', schedule=schedule, first_server=-1, target=11,
                    weights=checkpoint.name, weights_hash=_hash(checkpoint), policy=policy,
                    profile=weights['profile'], coefficients=weights['source'], config=CONFIG,
                    service_program=STANDARD_PICK, service_hash=_hash(FACTORY))
    result = _read(output) if output.exists() else dict(metadata=metadata, points=[])
    if result['metadata'] != metadata:
        raise ValueError('the existing game uses different policies or settings')
    rows = result['points']
    score = scoreboard(rows, seeds)
    if not score.finished:
        station = MatchPoint(weights['profile'])
        learner = BranchMotor(weights['source'], scorer(weights['contexts']))
        try:
            for seed in seeds[len(rows):]:
                before = list(score.score)
                row = station.play(seed, score.server, learner)
                after = list(score.record(row['outcome']))
                rows.append(dict(row, score_before=before, score_after=after))
                result['score'] = after
                _write(output, result)
                print(policy, len(rows), row['role'], row['longest'], after, row['outcome']['reason'], flush=True)
                if score.finished:
                    break
        finally:
            station.close()
    result.update(verified=verify(rows, seeds), summary=summarize(rows))
    _write(output, result)
    print(policy, result['score'], result['summary'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, default=PRACTICE)
    parser.add_argument('--output', type=Path, default=OUTPUT/'tt23-motor-game-baseline.json.gz')
    parser.add_argument('--schedule', type=int, default=SEED)
    parser.add_argument('--label', default='contextual')
    args = parser.parse_args()
    run(args.checkpoint, args.output, args.schedule, policy=args.label)


if __name__ == '__main__':
    main()
