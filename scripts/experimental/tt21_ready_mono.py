"""Transfer a frozen stereo lateral-preparation fit to unmarked single-eye play."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from multiprocessing import get_context

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt20_camera_carry import EYES, SETTINGS
from .tt20_curriculum import OUTPUT as TT20, receiver
from .tt20_fixture import TT20Arena
from .tt20_natural import NaturalViewTransport
from .tt20_opponent import install_opponent
from .tt21_body_state import StateInitializedDrySimulator
from .tt21_ready_position import OPENING, OUTPUT, inbound_features, outbound_features, ready_metrics, rotated_poses
from .tt21_recovery_trial import observed


STEREO = OUTPUT / 'ready-position-incoming-held.json.gz'
RECEIPT = OUTPUT / 'ready-mono.json.gz'
SCHEDULES = {'left': 1259313371, 'wrist': 882071163}
COUNT = 12


class MonoPositionStation:
    def __init__(self, eye):
        self.eye = eye
        self.arena = TT20Arena(**SETTINGS, view_mode=eye)
        install_opponent(self.arena)
        self.model = receiver(False, TT20)
        self.stroke = self.model.player
        self.stroke.sim.close()
        self.sim = self.stroke.sim = StateInitializedDrySimulator(self.stroke.law)
        self.catalog = self.stroke.poses
        shoulder = self.sim.model.joint('learner_shoulder_pan').id
        self.pivot = self.sim.twin.data.xanchor[shoulder, :2].copy()
        self.axis_z = self.sim.twin.data.xaxis[shoulder, 2]
        self.radians = self.sim.increments[0] * self.sim.timestep
        self.lens = self.arena.lens_intrinsics()[eye]

    def close(self):
        self.stroke.close()
        self.arena.close()

    def point(self, seed, role, coefficients):
        arena = self.arena
        arena.reset(seed=seed, server=role, render=False)
        self.model.reset(role)
        self.model.serves.force = OPENING
        self.sim.twin.reset()
        self.stroke.poses = self.catalog
        self.sim.current = None
        transport = NaturalViewTransport(self.eye, lens_fov={self.eye: self.lens}, law=self.stroke.law,
                                         frame_seconds=arena.control_dt)
        tracker = transport.tracker()
        actions, reads, optical = [], [], []
        pan, peak_pan = 0., 0.
        decision = None
        outbound = False
        for tick in range(arena.max_steps):
            images = arena.render_cameras()
            if tuple(images) != (self.eye,):
                raise ValueError('the controller must receive exactly one native RGB stream')
            tracker.update(images, tick=tick)
            ball = observed(tracker.ball)
            velocity = tracker.ball_velocity if ball is not None else None
            position = None if ball is None else ball.tolist()
            optical.append(dict(tick=tick, position=position,
                                velocity_m_per_frame=None if velocity is None else list(map(float, velocity)),
                                range=tracker.sources.get('ball_range')))
            if ball is not None:
                reads.append((tick, position))
            opened = bool(self.stroke.plans) or self.model.actor is not None
            if opened and decision is None and ball is not None:
                outbound |= outbound_features(reads) is not None
                cue = inbound_features(reads) if outbound else None
                if cue is not None:
                    lane = float(cue @ coefficients)
                    desired = np.clip(lane / .24, -.98, .98) if abs(lane) > .105 else 0.
                    decision = dict(tick=tick, cue=cue.tolist(), lane=lane, goal=float(desired / self.radians))
            if pan:
                self.sim.current = self.sim.twin.state()
            command = self.model.command(tick, ball, estimated_velocity=velocity)
            if decision is not None and self.model.phase == 'receive':
                delta = np.clip(decision['goal'] - pan, -1. - command[0], 1. - command[0])
                if abs(delta) > 1.e-8:
                    command = command.copy()
                    command[0] += delta
                    pan += delta
                    peak_pan = max(peak_pan, abs(pan))
                    self.stroke.poses = rotated_poses(self.catalog, self.pivot, self.axis_z * pan * self.radians)
            actions.append(command.tolist())
            self.sim.twin.advance(command)
            _, reward, done, capped, _ = arena.step(command, render=False)
            if done or capped:
                break
        return dict(seed=seed, role=role, decision=decision, peak_pan_rad=float(peak_pan * self.radians), reward=reward,
                    actions=actions, optical=optical, approach=None, plans=self.stroke.plans,
                    outcome=arena.point_result(), events=arena.events.copy(), rgb=tracker.digest.hexdigest())


def audit_replay(arena, rows):
    for row in rows:
        arena.reset(seed=row['seed'], server=row['role'], render=False)
        for command in row['actions']:
            _, reward, _, _, _ = arena.step(command, render=False)
        if (arena.point_result() != row['outcome'] or reward != row['reward']
                or json.dumps(arena.events) != json.dumps(row['events'])):
            raise ValueError(f"physical replay diverged for {row['seed']}")
    return len(rows)


def batch(eye, seeds, coefficients):
    station = MonoPositionStation(eye)
    conditions = {'shift': [], 'center': []}
    equivalence = 0
    try:
        for index, seed in enumerate(seeds):
            role = (-1, 1)[index % 2]
            shifted = station.point(seed, role, coefficients)
            if shifted['peak_pan_rad'] > 0.:
                centered = station.point(seed, role, np.zeros_like(coefficients))
                first = next((i for i, (a, b) in enumerate(zip(shifted['actions'], centered['actions'])) if a != b), None)
                if first is not None and first < shifted['decision']['tick']:
                    raise ValueError(f'paired controls disagreed before the optical cue for {seed}')
            else:
                cue = shifted['decision']
                centered = dict(shifted, decision=None if cue is None else dict(cue, lane=0., goal=0.),
                                identical_zero_shift=True)
                first = None
                equivalence += 1
            conditions['shift'].append(dict(shifted, first_action_difference=first))
            conditions['center'].append(centered)
            print(eye, index + 1, seed, shifted['peak_pan_rad'],
                  [row['outcome']['legal_returns'] for row in (centered, shifted)],
                  [row['reward'] for row in (centered, shifted)], flush=True)
        verified = audit_replay(station.arena, conditions['shift'] + conditions['center'])
    finally:
        station.close()
    metrics = {name: ready_metrics(rows) for name, rows in conditions.items()}
    for name, rows in conditions.items():
        metrics[name].update(realized_shifts=sum(row['peak_pan_rad'] > 0 for row in rows),
                             optical_positions=sum(sample['position'] is not None for row in rows for sample in row['optical']),
                             optical_velocities=sum(sample['velocity_m_per_frame'] is not None for row in rows
                                                    for sample in row['optical']))
    paired_wins = {name: sum(a['reward'] > 0 >= b['reward'] for a, b in zip(conditions[name], conditions[other], strict=True))
                   for name, other in (('shift', 'center'), ('center', 'shift'))}
    return dict(eye=eye, schedule_seed=SCHEDULES[eye], seeds=seeds, verified=verified,
                zero_shift_equivalence=equivalence, metrics=metrics, exclusive_point_wins=paired_wins, conditions=conditions)


def run(eyes, count, workers):
    source = _read(STEREO)
    if source['cue_mode'] != 'incoming':
        raise ValueError('expected the frozen stereo incoming-cue fit')
    coefficients = np.asarray(source['coefficients'], dtype=float)
    schedule = {eye: np.random.default_rng(SCHEDULES[eye]).integers(2**31, size=count).tolist() for eye in eyes}
    prior = {point['seed'] for arm in source['conditions'].values() for point in arm['points']}
    prior.update(point['seed'] for point in source['observations'])
    held = [seed for seeds in schedule.values() for seed in seeds]
    if len(set(held)) != len(held) or set(held) & prior:
        raise ValueError('the held points overlap the stereo evaluation or teaching set')
    result = dict(schema='tt21-ready-mono-1', stereo_source=STEREO.name, stereo_sha256=_hash(STEREO),
                  coefficients=coefficients.tolist(), arena=SETTINGS, opponent='search', held_schedule=schedule,
                  live_refit_points=0, results={})
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        jobs = {pool.submit(batch, eye, schedule[eye], coefficients): eye for eye in eyes}
        for done in as_completed(jobs):
            eye = jobs[done]
            result['results'][eye] = done.result()
            _write(RECEIPT, result)
            print(eye, result['results'][eye]['metrics'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eyes', nargs='+', choices=EYES, default=EYES)
    parser.add_argument('--starts', type=int, default=COUNT)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    run(args.eyes, args.starts, args.workers)
