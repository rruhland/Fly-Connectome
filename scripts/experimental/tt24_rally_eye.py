"""Single-camera, continuous online execution and cross-camera observation transfer."""
import argparse
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt22_opponent import install_rally_opponent
from .tt22_opponent_trial import FarTape
from .tt23_coach_trial import PRACTICE, ReplyRanking
from .tt23_sequence import CONFIG, SequenceArena, course, verify
from .tt24_rally_policy import JointPolicy, ViewHistory, orange_center
from .tt24_rally_trial import (COACHED, STEP_SIZE, PublicBody, Station, archive_schedule,
                               demo_schedule, returns)


DIMS = (240, 180)
DATA = Path('artifacts/tt24/rally-two-eyes.npz')


class SingleEye(ViewHistory):
    def observe(self, frames, joints, issued_position):
        # The shared evaluator hashes two slots. They alias one actual RGB image.
        position = orange_center(frames['left'], self.latest[0])
        return self.observe_pixels((position, None), joints, issued_position)


class SingleStation(Station):
    def __init__(self, profile, eye):
        self.arena = SequenceArena(**dict(CONFIG, width=DIMS[0], height=DIMS[1]))
        self.arena.streams = (eye,)
        native = self.arena.render_cameras
        def render():
            frame = native()[eye]
            return dict(left=frame, right=frame)
        self.arena.render_cameras = render
        self.rival = install_rally_opponent(self.arena, target_profile=profile)
        self.rival.confirm = ReplyRanking(self.rival, _read(PRACTICE)).confirm
        self.tape = FarTape(self.arena)


def collect():
    schedule = demo_schedule(archive_schedule())
    arena = SequenceArena(**dict(CONFIG, width=DIMS[0], height=DIMS[1]))
    eyes = ('left', 'wrist')
    arena.streams = eyes
    values = {key: [] for key in (*eyes, 'postures', 'rates', 'seed', 'clip')}
    try:
        for clip, row in enumerate(schedule):
            arena.lane = row['lane']
            arena.reset(seed=row['seed'], server=-1, render=False)
            histories = {key: SingleEye() for key in eyes}
            body = PublicBody(arena)
            stop = returns(row, 1)[-1] + 1
            for near, far in zip(row['actions'][:stop], row['rival_actions'][:stop], strict=True):
                near, far = np.asarray(near), np.asarray(far)
                frames = arena.render_cameras()
                for key, h in histories.items():
                    values[key].append(h.observe({'left': frames[key]}, arena.joints, body.target))
                    h.issued(near)
                arena._aim = arena._command_to = lambda *args: far
                _, _, done, cap, _ = arena.step(near, render=False)
                body.issued(near)
                values['postures'].append(body.target.copy())
                values['rates'].append(near)
                values['seed'].append(row['seed'])
                values['clip'].append(clip)
                if done or cap:
                    break
            assert digest_json(arena.events) == digest_json([v for v in row['events'] if v['step'] <= arena.steps])
    finally:
        arena.close()
    np.savez_compressed(DATA, **values)


class Correspondence:
    """Use paired, unlabeled views to estimate the other eye from this one."""
    def __init__(self, source, target, actor):
        # Each row includes measured posture. Map optical channels only.
        self.scale = actor.scale
        self.x = target / self.scale
        self.y = source[:, :7]
        self.tree = cKDTree(self.x)
        self.actor = actor

    def command(self, state, issued):
        query = state / self.scale
        distance, ids = self.tree.query(query, k=min(24, len(self.x)))
        weight = np.exp(-.5*((distance-distance.min()) / max(.1, .5*distance.max()))**2)
        delta = self.x[ids] - query
        design = np.column_stack((np.ones(len(ids)), delta))
        weighted = design.T * weight
        reg = np.eye(design.shape[1]) * .25
        reg[0,0] = 0.
        converted = state.copy()
        converted[:7] = np.linalg.solve(weighted @ design + reg, weighted @ self.y[ids])[0]
        return self.actor.command(converted, issued)


def actors(data, eye):
    other = 'wrist' if eye == 'left' else 'left'
    all_rows = np.ones(len(data['clip']), dtype=bool)
    one = data['clip'] == 0
    # Keep the one calibration demonstration out of the other-eye source library.
    source = data['seed'] != data['seed'][0]
    def fit(view, rows):
        return JointPolicy(data[view][rows], data['postures'][rows], data['rates'][rows], STEP_SIZE)
    source_actor = fit(other, source)
    aligned = data[other][one]
    shuffled = aligned.copy()
    shuffled[:, :7] = np.roll(aligned[:, :7], len(aligned)//2, axis=0)
    return dict(full=fit(eye, all_rows), one=fit(eye, one),
                transferred=Correspondence(aligned, data[eye][one], source_actor),
                unpaired=Correspondence(shuffled, data[eye][one], source_actor))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--collect', action='store_true')
    p.add_argument('--limit', type=int, default=9)
    p.add_argument('--eye', choices=('wrist', 'left'), default='wrist')
    p.add_argument('--extra-controls', action='store_true')
    p.add_argument('--fresh', action='store_true', help='independent unselected e/f counterbalanced releases')
    p.add_argument('--out', type=Path)
    args = p.parse_args()
    if args.collect:
        collect()
        return
    if args.out is None:
        p.error('--out required for scoring')
    with np.load(DATA) as data:
        policy = actors(data, args.eye)
        budget = dict(full=len(data['clip']), one=int((data['clip']==0).sum()),
                      other=int((data['seed']!=data['seed'][0]).sum()))
    station = SingleStation(_read(COACHED)['metadata']['opening']['far_profile'], args.eye)
    cases = []
    try:
        starts = course({'e': 1442869013, 'f': 802550131}) if args.fresh else archive_schedule()
        with np.load(DATA) as data:
            assert not set(data['seed']).intersection(row['seed'] for row in starts)
        for row in starts[:args.limit]:
            results = {}
            modes = ('unpaired', 'transferred_delayed') if args.extra_controls else ('full', 'delayed', 'one', 'transferred')
            if args.fresh:
                modes = ('one', 'transferred', 'unpaired', 'transferred_delayed')
            for mode in modes:
                condition = 'sight_delay' if 'delayed' in mode else 'demo_full'
                chosen = policy['transferred'] if mode == 'transferred_delayed' else policy.get(mode, policy['full'])
                results[mode] = station.play(row, condition, chosen,
                                             new_history=SingleEye, live_opening=True)
            print(row['seed'], [len(returns(r, 1)) for r in results.values()], flush=True)
            cases.append(dict(seed=row['seed'], results=results))
    finally:
        station.arena.close()
    _write(args.out, dict(eye=args.eye, frames=budget, demonstration_sha256=_hash(DATA),
                          cases=cases, verified=verify(cases)))


if __name__ == '__main__':
    main()
