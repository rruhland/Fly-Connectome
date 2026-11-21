"""Audit subpixel bearings, then run a matched optical rally comparison."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_arena import HOME
from .tt4_transfer_trial import _hash, _read, _write
from .tt18_view import CameraFit, triangulate_views
from .tt19_flight import optical_start
from .tt19_view_trial import CAMERAS
from .tt22_imaging import DetailArena, position
from .tt22_opponent import PLAN_FRAMES, _stroke
from .tt22_rally_trial import FRAME, OUTPUT, Station, alternating, rates


SOURCE = OUTPUT / 'rally-anchor-612423769.json.gz'
AUDIT = OUTPUT / 'imaging-offline.json.gz'
LIVE = OUTPUT / 'imaging-live.json.gz'
AUDIT_SEEDS = (419578354, 593435140)
SEED = 1618526033
MODES = ('stock', 'disk', 'detail')
PHASES = ('before_bounce', 'first_two_postbounce', 'later_postbounce')


class ArchivedOpponent:
    """Reconstruct selected far motor tapes without repeating the search."""

    def __init__(self, station, records):
        self.arena, self.rival = station.arena, station.rival
        self.records = {record['tick']: record for record in records if record['confirmed']}
        self.contacts = 0
        self.tape = None
        self.start = 0

    def __call__(self, side, *_):
        arena, rival = self.arena, self.rival
        if arena.contacts != self.contacts:
            self.contacts, self.tape = arena.contacts, None
        if arena._last_hitter == -1 or arena.velocity[1] <= .22:
            return arena._command_to(side, HOME, .97)
        if arena.steps in self.records:
            frames = min(PLAN_FRAMES, arena.max_steps-arena.steps)
            index, pace, lead, pitch, lateral, rise = self.records[arena.steps]['selected']
            eta, center, _ = rival.incoming.windows(frames)[index]
            goals = _stroke(arena, np.arange(frames)*arena.control_dt, eta, center, pace, lead, pitch, lateral, rise, None)
            low, high = arena._joint_low[rival.axes], arena._joint_high[rival.axes]
            target = arena.data.ctrl[rival.axes].copy()
            micros = np.arange(1, rival.micro+1)[:, None]
            self.tape = []
            for request in goals:
                rate = np.clip((np.clip(request, low, high)-target)/(rival.micro*rival.increment), -1., 1.)
                target = np.clip(target + micros*(rate*rival.increment), low, high)[-1]
                self.tape.append(rate)
            self.start = arena.steps
        if self.tape is not None and arena.steps-self.start < len(self.tape):
            return self.tape[arena.steps-self.start]
        return arena._command_to(side, HOME, .8)


def actual_cameras(arena):
    """Use simulator extrinsics to score bearings after the optical estimate."""
    look = np.diag((1., -1., -1.))
    specs = arena.lens_intrinsics()
    return {key: CameraFit(arena.data.camera(CAMERAS[key][0]).xpos.copy()-FRAME,
                           look @ arena.data.camera(CAMERAS[key][0]).xmat.reshape(3, 3).T,
                           specs[key]['focal'], arena.width, arena.height)
            for key in arena.streams}


def audit_frames(station, row):
    arena = station.arena
    arena.reset(seed=row['seed'], server=-1, render=False)
    arena._aim = ArchivedOpponent(station, row['rival'])
    tracker = station.optics.tracker()
    law = station.player.law
    modes = (*MODES, 'exact_pixels', 'exact_pose')
    history = {mode: [] for mode in modes}
    samples = []
    for tick, action in enumerate(row['actions']):
        truth = arena.ball.copy()-FRAME
        speed = arena.velocity.copy()
        capture = tick == 0 or (-.90 < truth[1] < .25 and speed[1] < -.55)
        if not capture:
            _, reward, _, _, _ = arena.step(action, render=False)
            continue
        tracker.update(arena.render_cameras(), tick=tick)
        fine = arena.render_detail() if tracker.sources.get('ball_range') == 'stereo' else None
        choices = {mode: position(tracker, mode, fine) for mode in MODES}
        cameras = actual_cameras(arena)
        expected = {key: camera.project(truth)[:2] for key, camera in cameras.items()}
        measured = choices['stock'][1]
        choices['exact_pixels'] = (triangulate_views(tracker.fits, {key: expected[key] for key in measured}), {})
        choices['exact_pose'] = (triangulate_views(cameras, measured), measured)
        estimates = {}
        for mode, (point, pixels) in choices.items():
            history[mode].append((tick, point))
            forecast = optical_start(history[mode], law.gravity)
            estimates[mode] = dict(position=None if point is None else point.tolist(),
                                   velocity=None if forecast is None else (30*forecast[1]).tolist(),
                                   pixel_residual={key: (pix-expected[key]).tolist() for key, pix in pixels.items()})
        active = bool(-.84 < truth[1] < -.30 and speed[1] < -.75 and .02 < truth[2] < .7)
        samples.append(dict(tick=tick, active=active, position=truth.tolist(), velocity=speed.tolist(), estimates=estimates))
        _, reward, _, _, _ = arena.step(action, render=False)
    if row['outcome'] != arena.point_result() or row['reward'] != reward or json.dumps(row['events']) != json.dumps(arena.events):
        raise ValueError(f"archived controls no longer reproduce {row['seed']}")
    return dict(seed=row['seed'], longest=row['longest'], sample_rgb_sha256=tracker.digest.hexdigest(), samples=samples)


def errors(rows):
    samples = [sample for row in rows for sample in row['samples'] if sample['active']]
    result = {}
    for mode in (*MODES, 'exact_pixels', 'exact_pose'):
        position_error, speed_error, pixel_error = [], [], []
        for sample in samples:
            seen = sample['estimates'][mode]
            if seen['position'] is not None:
                position_error.append(np.asarray(seen['position'])-sample['position'])
            if seen['velocity'] is not None:
                speed_error.append(np.asarray(seen['velocity'])-sample['velocity'])
            pixel_error.extend(seen['pixel_residual'].values())
        result[mode] = {}
        for label, values in (('position_m', position_error), ('velocity_m_s', speed_error), ('pixels', pixel_error)):
            result[mode][label] = dict(n=len(values), absolute_median=np.median(abs(np.asarray(values)), axis=0).tolist(),
                                      absolute_p90=np.quantile(abs(np.asarray(values)), .9, axis=0).tolist()) if values else dict(n=0)
    return dict(attack_frames=len(samples), modes=result)


def bounce_phases(rows, originals):
    """Label the audit only with the latest true near bounce in this flight."""
    for row in rows:
        events = originals[row['seed']]['events']
        far = [e['step'] for e in events if e['kind'] == 'contact' and e.get('side') == -1]
        near = [e['step'] for e in events if e['kind'] == 'bounce' and e.get('side') == 1]
        for sample in row['samples']:
            tick = sample['tick']
            far_contact = max((t for t in far if t <= tick), default=-1)
            bounce = max((t for t in near if far_contact < t <= tick), default=None)
            age = None if bounce is None else tick-bounce
            sample['near_bounce_age_frames'] = age
            sample['bounce_phase'] = PHASES[0] if age is None else PHASES[1] if age < 2 else PHASES[2]
    return {phase: errors([dict(samples=[sample for sample in row['samples'] if sample['bounce_phase'] == phase])
                           for row in rows]) for phase in PHASES}


def run_audit(count):
    source = _read(SOURCE)
    spread, pace, rival = (source[key] for key in ('spread', 'pace', 'rival'))
    station = Station(spread, pace, rival, arena=DetailArena(spread, pace))
    data = dict(schema='tt22-imaging-offline-1', source=SOURCE.name, source_sha256=_hash(SOURCE), rows=[])
    try:
        rows = {row['seed']: row for row in source['rows']}
        for seed in AUDIT_SEEDS[:count]:
            row = audit_frames(station, rows[seed])
            data['rows'].append(row)
            data['errors'] = errors(data['rows'])
            data['bounce_phases'] = bounce_phases(data['rows'], rows)
            _write(AUDIT, data)
            print('offline', seed, data['errors'], flush=True)
    finally:
        station.close()


def live_point(station, seed, mode):
    arena, player = station.arena, station.player
    arena.reset(seed=seed, server=-1, render=False)
    player.reset()
    tracker = station.optics.tracker()
    commands, optics = [], []
    started = perf_counter()
    fine_frames = 0
    for tick in range(arena.max_steps):
        tracker.update(arena.render_cameras(), tick=tick)
        fine = None
        if mode == 'detail' and tracker.sources.get('ball_range') == 'stereo':
            fine = arena.render_detail()
            fine_frames += 1
        estimate, _ = position(tracker, mode, fine)
        if estimate is not None:
            optics.append((tick, estimate.tolist()))
        action = player.command(tick, estimate)
        commands.append(action.tolist())
        _, reward, done, capped, _ = arena.step(action, render=False)
        if done or capped:
            break
    return dict(seed=seed, role=-1, reward=reward, actions=commands, optical=optics, shots=player.shots,
                longest=alternating(arena.events), events=arena.events.copy(), outcome=arena.point_result(),
                rival=station.rival.decisions.copy(), rgb=tracker.digest.hexdigest(), detail_frames=fine_frames,
                wall_seconds=perf_counter()-started)


def verify_live(data):
    spread, pace, rival = (data[key] for key in ('spread', 'pace', 'rival'))
    station = Station(spread, pace, rival)
    count = 0
    try:
        arena = station.arena
        for pair in data['pairs']:
            for row in pair.values():
                arena.reset(seed=row['seed'], server=-1, render=False)
                arena._aim = ArchivedOpponent(station, row['rival'])
                for command in row['actions']:
                    _, reward, _, _, _ = arena.step(command, render=False)
                if (arena.point_result() != row['outcome'] or reward != row['reward']
                        or json.dumps(arena.events) != json.dumps(row['events'])):
                    raise ValueError(f"physical replay changed on {row['seed']}")
                count += 1
    finally:
        station.close()
    return count


def run_live(mode, count, *, repeat):
    source = _read(SOURCE)
    spread, pace, rival = (source[key] for key in ('spread', 'pace', 'rival'))
    seeds = np.random.default_rng(SEED).integers(2**31, size=count).tolist()
    if repeat:
        seeds.insert(0, 593435140)
    data = dict(schema='tt22-imaging-live-1', source=SOURCE.name, source_sha256=_hash(SOURCE), mode=mode,
                spread=spread, pace=pace, rival=rival, schedule=seeds, repeated_development_seed=repeat,
                input='native stereo RGB', pairs=[])
    try:
        stations = {m: Station(spread, pace, rival, arena=DetailArena(spread, pace)) for m in ('stock', mode)}
        for i, seed in enumerate(seeds):
            order = ('stock', mode) if i % 2 == 0 else (mode, 'stock')
            pair = {name: live_point(stations[name], seed, name) for name in order}
            data['pairs'].append(pair)
            data['counts'] = {name: rates([p[name] for p in data['pairs']]) for name in ('stock', mode)}
            _write(LIVE, data)
            print('live', seed, {key: value['longest'] for key, value in pair.items()}, flush=True)
    finally:
        for station in stations.values():
            station.close()
    data['replayed'] = verify_live(data)
    _write(LIVE, data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', type=int, choices=(1, 2))
    parser.add_argument('--phase-audit', action='store_true')
    parser.add_argument('--verify-live', action='store_true')
    parser.add_argument('--live', choices=MODES[1:])
    parser.add_argument('--count', type=int, default=2)
    parser.add_argument('--repeat', action='store_true')
    args = parser.parse_args()
    if args.audit:
        run_audit(args.audit)
    elif args.phase_audit:
        data = _read(AUDIT)
        rows = {row['seed']: row for row in _read(SOURCE)['rows']}
        data['bounce_phases'] = bounce_phases(data['rows'], rows)
        _write(AUDIT, data)
        for phase, summary in data['bounce_phases'].items():
            print(phase, summary['attack_frames'], {mode: summary['modes'][mode]['velocity_m_s']
                                                   for mode in ('stock', 'disk', 'exact_pixels')})
    elif args.live:
        run_live(args.live, args.count, repeat=args.repeat)
    elif args.verify_live:
        data = _read(LIVE)
        data['replayed'] = verify_live(data)
        _write(LIVE, data)
        print('replayed', data['replayed'])
    else:
        parser.error('select --audit, --phase-audit, --live, or --verify-live')
