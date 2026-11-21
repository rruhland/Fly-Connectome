"""Audit online lateral forecasts with a rigid tabletop tilted in the native model."""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from .tt4_arena import TABLE_HEIGHT
from .tt4_transfer_trial import _hash, _read, _write
from .tt20_curriculum import OUTPUT as FACTORY, receiver
from .tt20_fixture import TT20Arena
from .tt20_natural import NaturalViewTransport
from .tt20_opponent import install_opponent
from .tt21_body_state import StateInitializedDrySimulator
from .tt21_ready_position import PRACTICE, fit, inbound_features, rotated_poses, shoulder_goal, training_pairs
from .tt21_recovery_trial import ARENA, OUTPUT


TILT_DEGREES = 4.
DEPTH = -.88
SCHEDULE = 479807264
CALIBRATION_SEED = 1896605105
PRACTICE_SIZE = 4
HELD_SIZE = 10
CV = np.array((0., 1., -DEPTH))
RESULT = 'changed-bounce-audit.json.gz'
HELD_RESULT = 'changed-bounce-strokes.json.gz'
STROKE_SEED = 1378826709
STROKE_COUNT = 8
CONTROL_FRAMES = 42
MIN_IMPROVEMENT_M = .025
MIN_ACTION_IMPROVEMENT = .12


def _vector(values):
    return ' '.join(f'{x:.12g}' for x in values)


class TiltedArena(TT20Arena):
    """Rotate the collision surface, painted lines, and net as one assembly."""

    def __init__(self, degrees):
        self.degrees = degrees
        super().__init__(**ARENA)

    def _model_document(self):
        doc = super()._model_document()
        if self.degrees == 0.:
            return doc
        world = doc.find('worldbody')
        angle = np.deg2rad(self.degrees)
        tilted = ET.SubElement(world, 'body', name='tt21_tilted_playing_surface',
                              pos=_vector((0., 0., TABLE_HEIGHT)),
                              quat=_vector((np.cos(angle/2), 0., np.sin(angle/2), 0.)))
        for geom in world.findall('geom'):
            if geom.get('name') == 'floor':
                continue
            pos = np.fromstring(geom.get('pos'), sep=' ')
            if pos[2] > TABLE_HEIGHT - .035:
                world.remove(geom)
                geom.set('pos', _vector(pos - (0., 0., TABLE_HEIGHT)))
                tilted.append(geom)
            elif geom.get('type') == 'box':
                # The four vertical legs keep their floor contacts and meet
                # the raised or lowered underside at the leg center.
                size = np.fromstring(geom.get('size'), sep=' ')
                height = TABLE_HEIGHT - .05 / np.cos(angle) - np.tan(angle) * pos[0]
                pos[2] = size[2] = .5 * height
                geom.set('pos', _vector(pos))
                geom.set('size', _vector(size))
        return doc


class FixedOptics(NaturalViewTransport):
    """Hold the image-calibrated fixed cameras in the arm's level reference frame."""

    def observe(self, frames, tick, prior, wrist_prior=None):
        fits = {key: self.registry.known[key] for key in frames}
        self.registry.active = fits
        self.registry.last_events = {key: 'verified' for key in fits}
        return fits


def camera_calibration(arena):
    arena.reset(seed=CALIBRATION_SEED, server=-1, render=False)
    natural = NaturalViewTransport('fixed', lens_fov=arena.lens_intrinsics(), flight=False)
    tracker = natural.tracker()
    tracker.update(arena.render_cameras(), tick=0)
    state = natural.state()
    if set(state) != {'left', 'right'}:
        raise ValueError('both fixed cameras must register from natural court lines')
    return state, tracker.digest.hexdigest()


def crossing(reads):
    """Interpolate a measured crossing between two adjacent optical sightings."""
    for (t, a), (u, b) in zip(reads, reads[1:]):
        if u - t <= 3 and a[1] > DEPTH >= b[1] and (b[1] - a[1]) / (u-t) < -.02:
            part = (DEPTH-a[1]) / (b[1]-a[1])
            return dict(x=float(a[0] + part*(b[0]-a[0])), tick=float(t + part*(u-t)))
    return None


def trial(arena, state, seed):
    tracker = FixedOptics('fixed', state=state, lens_fov=arena.lens_intrinsics(), flight=False).tracker()
    arena.reset(seed=seed, server=-1, render=False)
    optical, truth, cue = [], [], None
    for tick in range(CONTROL_FRAMES):
        tracker.update(arena.render_cameras(), tick=tick)
        if tracker.ball is not None and tracker.ball.observed and tracker.sources.get('ball_range') == 'stereo':
            position = list(tracker.ball.position)
            optical.append((tick, position))
            if cue is None:
                features = inbound_features(optical)
                if features is not None:
                    cue = dict(features=features.tolist(), tick=tick)
        # Evaluator truth is stored for scoring the audit. It cannot change
        # the images, observation window, optical teacher or predictor.
        truth.append((tick, (arena.ball - (0., 0., TABLE_HEIGHT)).tolist()))
        _, _, done, cap, _ = arena.step(np.zeros(6), render=False)
        if done or cap:
            break
    near = next((event for event in arena.events if event['kind'] == 'bounce' and event.get('side') == 1), None)
    post = None if near is None else next((event for event in arena.events if event['time'] > near['time']
                 and event['kind'] == 'separation' and event.get('collider') == 'table'), None)
    return dict(seed=seed, cue=cue, optical=optical, target=crossing(optical), rgb=tracker.digest.hexdigest(),
                evaluator_target=crossing(truth), evaluator_trace=truth, near_bounce=near,
                transverse_bounce_velocity_change=None if post is None else post['velocity'][0] - near['incoming'][0],
                events=arena.events)


def adapt_offset(practice, carried):
    """Use only later RGB sightings on earlier points to learn one offset."""
    residuals = [row['target']['x'] - np.dot(row['cue']['features'], carried)
                 for row in practice if row['cue'] is not None and row['target'] is not None]
    return float(np.sum(residuals) / (len(residuals) + 2.)), residuals


def metrics(rows, carried, intercept):
    usable = [row for row in rows if row['cue'] is not None and row['evaluator_target'] is not None]
    paired = []
    for row in usable:
        feature = row['cue']['features']
        x = row['evaluator_target']['x']
        predictions = dict(constant_velocity=float(CV @ feature), carried=float(carried @ feature),
                           adapted=float(carried @ feature + intercept))
        paired.append(dict(seed=row['seed'], truth=x, prediction=predictions,
                           optical=None if row['target'] is None else row['target']['x'],
                           lateral_error={key: value-x for key, value in predictions.items()},
                           arm_error={key: abs(shoulder_goal(value, 1.)-shoulder_goal(x, 1.))
                                      for key, value in predictions.items()},
                           lead_frames=row['evaluator_target']['tick']-row['cue']['tick']))
    median = {key: float(np.median([abs(point['lateral_error'][key]) for point in paired])) if paired else None
              for key in ('constant_velocity', 'carried', 'adapted')}
    actionable = sum(point['arm_error']['constant_velocity'] - point['arm_error']['adapted'] >= MIN_ACTION_IMPROVEMENT
                     for point in paired)
    return dict(usable=len(paired), median_abs_error_m=median, actionable_improvements=actionable, rows=paired,
                qualify_strokes=len(paired) >= 7 and actionable >= 2
                and median['constant_velocity'] - median['adapted'] >= MIN_IMPROVEMENT_M)


def run(output):
    starts = np.random.default_rng(SCHEDULE).integers(2**31, size=PRACTICE_SIZE+HELD_SIZE).tolist()
    starts_source = training_pairs('incoming')
    if len(set(starts)) != len(starts) or set(starts) & ({CALIBRATION_SEED}|{row['seed'] for row in starts_source}):
        raise ValueError('the probe, practice, and scoring points must be disjoint')
    carried = fit(starts_source)
    cases = {}
    for mode, degrees in (('level', 0.), ('tilted', TILT_DEGREES)):
        arena = TiltedArena(degrees)
        try:
            if mode == 'level':
                state, image_hash = camera_calibration(arena)
            rows = []
            offsets = []
            for i, seed in enumerate(starts):
                row = trial(arena, state, seed)
                rows.append(row)
                if i < PRACTICE_SIZE:
                    offset, updates = adapt_offset(rows, carried)
                    offsets.append(offset)
                print(mode, seed, row['cue'] is not None, row['target'] is not None, flush=True)
            surface_normal = arena.data.geom_xmat[arena.model.geom('table').id].reshape((3, 3))[:, 2].tolist()
            cases[mode] = dict(table_normal=surface_normal, degrees=degrees, optical_offset=offset, optical_offsets=offsets,
                               practice_residuals=updates, train=rows[:PRACTICE_SIZE], held=rows[PRACTICE_SIZE:],
                               summary=metrics(rows[PRACTICE_SIZE:], carried, offset))
        finally:
            arena.close()
    level, tilted = cases['level']['held'], cases['tilted']['held']
    shift = [b['evaluator_target']['x']-a['evaluator_target']['x'] for a,b in zip(level, tilted, strict=True)
             if a['evaluator_target'] is not None and b['evaluator_target'] is not None]
    velocity_key = 'transverse_bounce_velocity_change'
    changes = [b[velocity_key]-a[velocity_key] for a,b in zip(level, tilted, strict=True)
               if a[velocity_key] is not None and b[velocity_key] is not None]
    result = dict(schema='tt21-changed-bounce-audit-1', seed=SCHEDULE, starts=starts,
                  parent_sha256=_hash(PRACTICE), prior=carried.tolist(), prior_examples=len(starts_source),
                  calibration=state, calibration_rgb_sha256=image_hash, cases=cases,
                  held_evaluator_lateral_shift_m=shift, held_evaluator_velocity_change_m_s=changes,
                  qualification=dict(min_pairs=7, min_improvement_m=MIN_IMPROVEMENT_M,
                                     min_action_improvements=2, min_action_improvement_rad=MIN_ACTION_IMPROVEMENT))
    _write(output, result)
    print({key: {k: v for k, v in value['summary'].items() if k != 'rows'} for key, value in cases.items()}, flush=True)


class ReadyReceiver:
    """Use the same shoulder preparation on the first incoming arc of an ordinary point."""

    def __init__(self, camera_state):
        self.arena = TiltedArena(TILT_DEGREES)
        install_opponent(self.arena)
        self.optics = FixedOptics('fixed', state=camera_state, lens_fov=self.arena.lens_intrinsics(), flight=False)
        self.player = receiver(False, FACTORY)
        stroke = self.player.player
        stroke.sim.close()
        self.sim = stroke.sim = StateInitializedDrySimulator(stroke.law)
        self.catalogue = stroke.poses
        joint = self.sim.model.joint('learner_shoulder_pan').id
        self.pivot = self.sim.twin.data.xanchor[joint, :2].copy()
        self.axis = self.sim.twin.data.xaxis[joint, 2]
        self.radians = self.sim.increments[0] * self.sim.timestep

    def close(self):
        self.player.player.close()
        self.arena.close()

    def point(self, seed, coefficients):
        arena, player, sim = self.arena, self.player, self.sim
        arena.reset(seed=seed, server=-1, render=False)
        player.reset(-1)
        player.player.poses = self.catalogue
        sim.twin.reset()
        sim.current = None
        tracker = self.optics.tracker()
        optical, commands = [], []
        pan, decision = 0., None
        for tick in range(arena.max_steps):
            tracker.update(arena.render_cameras(), tick=tick)
            ball = tracker.ball
            position = None if ball is None or not ball.observed else np.asarray(ball.position)
            if position is not None and tracker.sources.get('ball_range') == 'stereo':
                optical.append((tick, position.tolist()))
                if decision is None:
                    features = inbound_features(optical)
                    if features is not None:
                        lane = float(coefficients @ features)
                        decision = dict(tick=tick, lane=lane, goal=shoulder_goal(lane, self.radians))
            if pan:
                sim.current = sim.twin.state()
            action = player.command(tick, position)
            if decision is not None:
                delta = np.clip(decision['goal'] - pan, -1.-action[0], 1.-action[0])
                if abs(delta) > 1e-8:
                    action[0] += delta
                    pan += delta
                    player.player.poses = rotated_poses(self.catalogue, self.pivot,
                                                       self.axis * pan * self.radians)
            commands.append(action.tolist())
            sim.twin.advance(action)
            _, reward, done, cap, _ = arena.step(action, render=False)
            if done or cap:
                break
        return dict(seed=seed, decision=decision, actions=commands, reward=reward, rgb=tracker.digest.hexdigest(),
                    optical=optical, result=arena.point_result(), events=arena.events.copy(), plans=player.player.plans)


def stroke_test(audit_path, output):
    audited = _read(audit_path)
    tilted = audited['cases']['tilted']
    if not tilted['summary']['qualify_strokes']:
        raise ValueError('the optical estimates did not qualify for a held stroke test')
    starts = np.random.default_rng(STROKE_SEED).integers(2**31, size=STROKE_COUNT).tolist()
    if set(starts) & set(audited['starts']) or len(set(starts)) != STROKE_COUNT:
        raise ValueError('stroke test must use new starts')
    adapted = np.asarray(audited['prior']) + np.array((tilted['optical_offset'], 0., 0.))
    options = dict(constant_velocity=CV, adapted=adapted)
    station = ReadyReceiver(audited['calibration'])
    pairs = []
    try:
        for i, seed in enumerate(starts):
            mode = tuple(options)
            rows = {name: station.point(seed, options[name]) for name in (mode if i % 2 == 0 else mode[::-1])}
            pairs.append(rows)
            print('strokes', seed, {key: row['result']['legal_returns'] for key, row in rows.items()}, flush=True)
        for rows in pairs:
            for row in rows.values():
                arena = station.arena
                arena.reset(seed=row['seed'], server=-1, render=False)
                for command in row['actions']:
                    _, reward, _, _, _ = arena.step(command, render=False)
                if (row['result'] != arena.point_result() or row['reward'] != reward
                        or json.dumps(row['events']) != json.dumps(arena.events)):
                    raise ValueError('held physical replay differs from the original point')
    finally:
        station.close()
    summary = {}
    for name in options:
        rows = [p[name] for p in pairs]
        summary[name] = dict(points=len(rows), legal_first_returns=sum(row['result']['legal_returns'] > 0 for row in rows),
                             contacts=sum(row['result']['contacts'] > 0 for row in rows),
                             wins=sum(row['reward'] > 0 for row in rows),
                             used_pan=sum(row['decision'] is not None and bool(row['decision']['goal']) for row in rows))
    _write(output, dict(schema='tt21-changed-bounce-strokes-1', seed=STROKE_SEED, starts=starts,
                        audit_sha256=_hash(audit_path), pairs=pairs, summary=summary, replays=2*len(pairs)))
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT/RESULT)
    parser.add_argument('--strokes', action='store_true')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.strokes:
        stroke_test(args.output, args.output.with_name(HELD_RESULT))
    else:
        run(args.output)
