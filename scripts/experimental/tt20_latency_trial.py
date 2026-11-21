"""Bidirectional single-eye transfer of a learned actuator delay on the unmarked clamp."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import sha256
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from .tt4_arena import TABLE_HEIGHT
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt7_flight import FlightMemory
from .tt19_catalog import dry_catalog
from .tt19_direct_receive import previous_seeds
from .tt19_flight import FlightLaw
from .tt19_model_rollout import DrySimulator
from .tt19_so101 import mujoco
from .tt19_stroke_trial import DONOR
from .tt20_fixture import TT20Arena
from .tt20_latency import CommandPipe, LatencyPlayer, blade_motion, identify_delay, scene_motion
from .tt20_natural import NaturalViewTransport


EYES = ('left', 'wrist')
BRANCHES = ('unadapted', 'transferred', 'destination')
BENCH = dict(venue='hall', detail='preview', strict_endline=True, mount='clamp', marked=False,
             wrist_optics='outboard', width=480, height=360, max_steps=190)
PROBE_SEED = 1671947024
CHECK_SEED = 1938450927
FRESH_SEED = 3011576803
HIDDEN_DELAY = 4
ARCHIVE = 'latency-clamp-catalogue.npz'
RECEIPT = 'direct-receive-latency.json.gz'
FRESH_RECEIPT = 'direct-receive-latency-replication.json.gz'
PROBE = np.zeros((28, 6))
PROBE[2:5, :5] = (.72, -.08, 0., -.26, .13)
PROBE[7:10, :5] = -PROBE[2:5, :5]
PROBE[12:15, :5] = (-.18, .38, -.43, .58, .22)
PROBE[17:20, :5] = -PROBE[12:15, :5]


def factory_arena():
    return TT20Arena(**dict(BENCH, venue='studio', strict_endline=False), view_mode='left')


def single_image(arena, eye):
    frames = arena.render_cameras()
    if tuple(frames) != (eye,):
        raise ValueError('a second image stream entered the single-eye experiment')
    return frames[eye]


def probe(eye, delay):
    arena = TT20Arena(**BENCH, view_mode=eye)
    try:
        arena.reset(seed=PROBE_SEED, render=False, server=-1)
        ball = arena.model.geom('ball')
        ball.contype[0] = ball.conaffinity[0] = 0
        arena.ball[:] = (0., 0., -25.)
        arena.velocity[:] = 0.
        mujoco.mj_forward(arena.model, arena.data)
        queue = CommandPipe(delay)
        actuators = arena._arm_ids[1][2]
        low, high = arena._joint_low[actuators], arena._joint_high[actuators]
        steps = int(round(arena.control_dt / arena.model.opt.timestep))
        scales = np.arange(1, steps + 1)[:, None]
        frames = [single_image(arena, eye)]
        for action in PROBE:
            settings = np.clip(arena.data.ctrl[actuators] + scales * (arena._joint_increments[actuators] * queue.apply(action)),
                               low, high)
            for values in settings:
                arena.data.ctrl[actuators] = values
                mujoco.mj_step(arena.model, arena.data)
            frames.append(single_image(arena, eye))
        motion = blade_motion(frames) if eye == 'left' else scene_motion(frames)
        digest = sha256()
        for rgb in frames:
            digest.update(rgb.tobytes())
        return motion, digest.hexdigest()
    finally:
        arena.close()


def calibrate():
    eyes = {}
    for eye in EYES:
        factory_motion, template_hash = probe(eye, 0)
        observations, live_hash = probe(eye, HIDDEN_DELAY)
        result = identify_delay(factory_motion, observations)
        eyes[eye] = dict(result, charged_frames=len(PROBE), template_motion=factory_motion.tolist(),
                         measured_motion=observations.tolist(), factory_image_sha256=template_hash, live_image_sha256=live_hash)
        print('probe', eye, result, flush=True)
    return eyes


def policy(atlas, directory, lead):
    with np.load(atlas) as data:
        catalogue = dict(data)
    rule = FlightLaw.from_state(_read(directory / 'flight-source.json.gz')['source_law'])
    state = _read(DONOR)['series']
    response = FlightMemory('impulse', state['flight_history'], state['flight_impulses']).response_matrix
    player = LatencyPlayer(catalogue, rule, response, lead)
    player.sim.close()
    player.sim = DrySimulator(rule, arena=factory_arena())
    return player


def play(arena, eye, receiver, seed, frames, reference):
    arena.reset(seed=seed, render=False, server=-1)
    queue = CommandPipe(HIDDEN_DELAY)
    lens = arena.lens_intrinsics()[eye]
    camera = NaturalViewTransport(eye, lens_fov={eye: lens}).tracker()
    receiver.reset()
    commands, applied, observations = [], [], []
    same = reference is not None
    shared = 0
    for tick in range(arena.max_steps):
        if same and tick < len(frames):
            rgb = frames[tick]
            shared += 1
        else:
            rgb = single_image(arena, eye)
            if reference is None:
                frames.append(rgb)
        camera.update({eye: rgb})
        ball = camera.ball
        point = ball.position if ball is not None and ball.observed else None
        velocity = camera.ball_velocity
        action = receiver.command(tick, point, -1, estimated_velocity=velocity)
        physical = queue.apply(action)
        observations.append(dict(ball=None if point is None else np.asarray(point).tolist(),
                                 velocity=None if velocity is None else np.asarray(velocity).tolist(),
                                 range=camera.sources.get('ball_range')))
        commands.append(action.tolist())
        applied.append(physical.tolist())
        if same and (tick >= len(reference) or applied[-1] != reference[tick]):
            same = False
        _, _, terminal, capped, _ = arena.step(physical, render=False)
        if terminal or capped:
            break
    return dict(seed=seed, eye=eye, learned_delay=receiver.delay, requested=commands, applied=applied,
                optical=observations, rgb_sha256=camera.digest.hexdigest(), plans=receiver.plans,
                shared_rgb_frames=shared, result=arena.point_result(), events=arena.events)


def batch(atlas, previous, observed, eye, schedule):
    arena = TT20Arena(**BENCH, view_mode=eye)
    other = next(view for view in EYES if view != eye)
    distinct = {'unadapted': 0, 'transferred': observed[other]['delay_frames'], 'destination': observed[eye]['delay_frames']}
    selectable = {lead: policy(atlas, previous, lead) for lead in set(distinct.values())}
    rows = []
    try:
        for index, seed in schedule:
            order = BRANCHES if index % 2 == 0 else tuple(reversed(BRANCHES))
            outcomes = {}
            frames, reference = [], None
            for branch in order:
                lead = distinct[branch]
                if lead not in outcomes:
                    point = play(arena, eye, selectable[lead], seed, frames, None if reference is None else reference['applied'])
                    if reference is None:
                        reference = point
                    else:
                        shared = point['shared_rgb_frames']
                        if point['optical'][:shared] != reference['optical'][:shared]:
                            raise ValueError('distinct branches disagreed on identical RGB frames')
                    outcomes[lead] = point
            row = {branch: outcomes[lead] for branch, lead in distinct.items()}
            rows.append(dict(index=index, seed=seed, live_executions=len(outcomes), **row))
            print(eye, seed, {branch: data['result']['legal_returns'] for branch, data in row.items()}, flush=True)
    finally:
        for player in selectable.values():
            player.close()
        arena.close()
    return rows


def replay(report):
    count = 0
    for eye, rows in report['pairs'].items():
        arena = TT20Arena(**report['arena'], view_mode=eye)
        try:
            for row in rows:
                for branch in BRANCHES:
                    point = row[branch]
                    arena.reset(seed=row['seed'], server=-1, render=False)
                    queue = CommandPipe(report['evaluator_delay_frames'])
                    for requested, physical in zip(point['requested'], point['applied'], strict=True):
                        action = queue.apply(requested)
                        if physical != action.tolist():
                            raise ValueError('the actual command transport differs on replay')
                        arena.step(action, render=False)
                    if (point['result'] != arena.point_result() or
                            json.dumps(point['events']) != json.dumps(arena.events)):
                        raise ValueError(f"single-eye timing differs on replay for {row['seed']}")
                    count += 1
        finally:
            arena.close()
    return count


def summarize(rows):
    return {branch: dict(points=len(rows), first_returns=sum(bool(r[branch]['result']['legal_returns']) for r in rows),
                         total_returns=sum(r[branch]['result']['legal_returns'] for r in rows),
                         point_wins=sum(r[branch]['result']['points_for'] for r in rows),
                         hits=sum(bool(r[branch]['result']['contacts']) for r in rows),
                         plans=sum(len(r[branch]['plans']) for r in rows),
                         optical_frames=sum(sum(s['ball'] is not None for s in r[branch]['optical']) for r in rows))
            for branch in BRANCHES}


def audit_ranging(report):
    """Evaluator-only comparison of the saved estimates with replayed physical flight."""
    summary = {}
    origin = np.array((0., 0., TABLE_HEIGHT))
    for eye, rows in report['pairs'].items():
        arena = TT20Arena(**report['arena'], view_mode=eye)
        summary[eye] = {}
        try:
            for branch in ('unadapted', 'transferred'):
                windows = visible = 0
                within = []
                for row in rows:
                    point = row[branch]
                    arena.reset(seed=row['seed'], server=-1, render=False)
                    own_strikes = [event['step'] for event in point['events']
                                   if event['kind'] == 'contact' and event['side'] == 1]
                    for tick, (sample, command) in enumerate(zip(point['optical'], point['applied'], strict=True)):
                        if own_strikes and tick >= own_strikes[0]:
                            break
                        physical = arena.ball - origin
                        if tick >= 8 and -.81 < physical[1] < -.24 and arena.velocity[1] * arena.control_dt < -.031:
                            windows += 1
                            if sample['ball'] is not None:
                                visible += 1
                                within.append(np.asarray(sample['ball']) - physical)
                        arena.step(command, render=False)
                details = dict(physical_window_frames=windows, observed_frames=visible)
                if within:
                    error = np.asarray(within)
                    details.update(median_depth_bias_m=float(np.median(error[:, 1])),
                                   median_abs_depth_error_m=float(np.median(abs(error[:, 1]))),
                                   median_error_m=float(np.median(np.linalg.norm(error, axis=1))),
                                   p90_error_m=float(np.percentile(np.linalg.norm(error, axis=1), 90)))
                summary[eye][branch] = details
        finally:
            arena.close()
    return summary


def schedule_starts(previous, folder, path, rng_seed, count):
    fresh = np.random.default_rng(rng_seed).integers(2**31, size=count).tolist()
    earlier = previous_seeds(previous) | previous_seeds(folder) | {PROBE_SEED}
    for directory in (previous, folder):
        for receipt in directory.glob('direct-receive*.json.gz'):
            if receipt.resolve() == path.resolve():
                continue
            report = _read(receipt)
            earlier.update(report.get('held_starts', ()))
            pairs = report.get('pairs', ())
            for rows in (pairs.values() if isinstance(pairs, dict) else (pairs,)):
                earlier.update(row['seed'] for row in rows)
    if len(set(fresh)) != count or set(fresh) & earlier:
        raise ValueError('held starts overlap another experiment or the probe')
    return fresh


def run(previous, folder, count, workers, *, replicate=False):
    path = folder / (FRESH_RECEIPT if replicate else RECEIPT)
    rng_seed = FRESH_SEED if replicate else CHECK_SEED
    fresh = schedule_starts(previous, folder, path, rng_seed, count)
    atlas = folder / ARCHIVE
    if replicate:
        original = _read(folder / RECEIPT)
        if _hash(atlas) != original['catalogue_sha256']:
            raise ValueError('the receiving catalogue was modified between schedules')
        if (original['arena'] != BENCH or original['evaluator_delay_frames'] != HIDDEN_DELAY
                or original['source_priors_sha256'] != _hash(previous/'flight-source.json.gz')
                or original['donor_sha256'] != _hash(DONOR)):
            raise ValueError('the receiving fixture or inherited priors changed between schedules')
        report = {key: val for key, val in original.items()
                  if key not in ('pairs', 'metrics', 'replayed_points', 'ranging_audit')}
        report.update(held_seed=rng_seed, held_starts=fresh, preceding_receipt_sha256=_hash(folder / RECEIPT),
                      additional_target_calibration_frames=0)
    else:
        report = prepare_report(atlas, previous, rng_seed, fresh)
    _write(path, report)
    schedule = list(enumerate(fresh))
    with ProcessPoolExecutor(max_workers=workers, mp_context=get_context('spawn')) as pool:
        jobs = {eye: [pool.submit(batch, atlas, previous, report['calibration'], eye, schedule[i::workers])
                      for i in range(min(workers, len(schedule)))] for eye in EYES}
        pairs = {eye: sorted([row for job in tasks for row in job.result()], key=lambda row: row['index'])
                 for eye, tasks in jobs.items()}
    report.update(pairs=pairs, metrics={eye: summarize(rows) for eye, rows in pairs.items()})
    report['replayed_points'] = replay(report)
    report['ranging_audit'] = audit_ranging(report)
    _write(path, report)
    print(report['metrics'], flush=True)


def prepare_report(atlas, previous, rng_seed, fresh):
    seen = calibrate()
    model = dry_catalog(atlas, 2048, arena=factory_arena())
    return dict(schema='tt20-single-eye-command-delay-v1', arena=BENCH, evaluator_delay_frames=HIDDEN_DELAY,
                probe_seed=PROBE_SEED, probe_commands=PROBE.tolist(), calibration=seen,
                protocol=dict(transferred='only delay_frames from other eye', destination='delay_frames from same eye',
                              unadapted='zero-frame estimate'), held_seed=rng_seed, held_starts=fresh,
                catalogue_sha256=_hash(atlas), dry_model=model, source_priors_sha256=_hash(previous/'flight-source.json.gz'),
                donor_sha256=_hash(DONOR), matched_camera_policy='natural single-eye temporal flight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous', type=Path, default=ROOT/'artifacts/tt19')
    parser.add_argument('--folder', type=Path, default=ROOT/'artifacts/tt20')
    parser.add_argument('--starts', type=int, default=8)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--fresh', action='store_true')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--verify', action='store_true')
    action.add_argument('--audit-ranging', action='store_true')
    args = parser.parse_args()
    if args.verify or args.audit_ranging:
        saved = _read(args.folder / (FRESH_RECEIPT if args.fresh else RECEIPT))
        if _hash(args.folder/ARCHIVE) != saved['catalogue_sha256']:
            raise ValueError('the receiving catalogue was modified')
        if args.verify:
            print('replayed', replay(saved))
        else:
            print(json.dumps(audit_ranging(saved), indent=2))
    elif not 1 <= args.workers <= (20 if args.fresh else args.starts):
        parser.error('expected at least as many paired starts as workers, with a positive count')
    else:
        args.folder.mkdir(parents=True, exist_ok=True)
        run(args.previous, args.folder, 20 if args.fresh else args.starts, args.workers, replicate=args.fresh)
