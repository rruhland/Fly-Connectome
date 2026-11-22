"""Test direct Pong gains with anonymous RGB and a physically varied far arm."""

import argparse
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_arena import TABLE_HEIGHT
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_flight import FlightLaw
from .tt19_view_full_game import digest_json
from .tt22_motion_object import MotionObject, OPENING, playable
from .tt22_opponent import ANCHOR, VARIED, install_rally_opponent
from .tt22_opponent_learn import JitteredOpening, PACE, SPREAD
from .tt22_opponent_trial import FarTape
from .tt22_rally_trial import CONFIG, alternating
from .tt22_spatial import SpatialChoice, SpatialPlayer, broad_hypotheses, portable_contact
from .tt22_spatial_trial import DRAW, OUTPUT, SOURCE, SOURCE_REFINEMENT, metrics


RIVAL_SOURCE = OUTPUT/'tt22_opponent-learned.json.gz'
DURATION_SOURCE = OUTPUT/'duration-pong-player.json'
GENERATOR = 1472065101
DURATION_HELD = 1764237199
EXPLORE = 1111385982
MODES = ('native_fixed', 'family', 'scratch', 'score_permuted', 'geometry')
LEARNED_MODES = ('family', 'scratch', 'score_permuted')
FRAME = np.array((0., 0., TABLE_HEIGHT))


class Station:
    def __init__(self, delta):
        self.arena = JitteredOpening(**CONFIG)
        self.rival = install_rally_opponent(self.arena, lane='varied')
        fixed_targets = self.rival.target
        self.rival.target = lambda: tuple(np.asarray(fixed_targets()) + delta)
        self.far = FarTape(self.arena)
        self.player = SpatialPlayer(FlightLaw(), direct=True)
        self.lenses = self.arena.lens_intrinsics()

    def close(self):
        self.player.close()
        self.arena.close()

    def point(self, seed, coefficients, *, geometry=False, switch=None):
        arena, player = self.arena, self.player
        arena.reset(seed=seed, server=-1, render=False)
        self.far.steps.clear()
        player.reset()
        player.coefficients = coefficients
        player.geometric_override = geometry
        observer = MotionObject(self.lenses)
        digest = sha256()
        actions, optical, receive = [], [], {}
        host = perf_counter()
        for tick in range(arena.max_steps):
            if switch is not None and tick == switch['tick']:
                player.coefficients = switch['coefficients']
                player.geometric_override = switch['geometry']
            frames = arena.render_cameras()
            for eye, image in frames.items():
                digest.update(eye.encode())
                digest.update(image)
            estimate = observer.update(frames, tick)
            # Forward kinematics are based only on the independent command twin.
            twin = player.sim.twin
            center, _, _, _ = player.factory.pose(twin.data.ctrl[twin.axes[:5]])
            player.observe_actor(tick, center)
            if tick < OPENING:
                action = player.command(tick, None)
                if tick == OPENING - 1:
                    prefix = dict(observer.prefix_positions)
                    player.history[:] = [(time, prefix.get(time)) for time in range(OPENING)]
            else:
                action = player.command(tick, estimate)

            # Scorer-side diagnostics run after command selection and never feed policy.
            truth = arena.ball.copy()-FRAME
            record = dict(tick=tick, mover=None if estimate is None else estimate.tolist(),
                          actor_command_fk=center.tolist(), eligible=playable(truth),
                          error=None if estimate is None else float(np.linalg.norm(estimate-truth)))
            if tick == OPENING-1:
                for time, guess in observer.prefix_positions:
                    if time < tick:
                        optical[time]['mover'] = guess.tolist()
                        optical[time]['error'] = float(np.linalg.norm(guess-np.asarray(optical[time]['reference'])))
            if tick < OPENING:
                record['reference'] = truth.tolist()
            optical.append(record)
            reply = arena.rival_returns
            if reply and arena._last_hitter == -1 and -.99 < truth[1] < -.80 and arena.velocity[1] < -.3:
                gap = abs(truth[1]+.91)
                if reply not in receive or gap < receive[reply][0]:
                    receive[reply] = gap, dict(reply_index=reply, xyz=truth.tolist(),
                                               velocity=arena.velocity.tolist(), tick=tick)
            actions.append(action.tolist())
            _, reward, done, capped, _ = arena.step(action, render=False)
            if done or capped:
                break
        for note in player.shots:
            row = optical[note['tick']]
            actor = row['actor_command_fk'][0]
            nominal = note['anchor'][0]
            mover = row['mover']
            current = switch if switch is not None and note['tick'] >= switch['tick'] else dict(
                coefficients=coefficients, geometry=geometry)
            x = nominal if current['geometry'] else portable_contact(actor, mover[0], nominal, current['coefficients'])
            note['contact_x'] = dict(nominal=float(nominal), requested=float(x))
        return dict(seed=seed, role=-1, coefficients=list(map(float, coefficients)), geometry=geometry, switch=switch,
                    actions=actions, rival_actions=self.far.tape(len(actions)), events=arena.events.copy(),
                    outcome=arena.point_result(), reward=reward, longest=alternating(arena.events),
                    optical=optical, rgb_sha256=digest.hexdigest(), bootstrap_score=observer.seed_score,
                    pose_origin='nominal-command-fk', changed_proposals=player.applied,
                    maximum_request_shift_m=player.max_correction, shots=player.shots,
                    rival_plans=self.rival.decisions.copy(), receiving=[value for _, value in receive.values()],
                    wall_seconds=perf_counter()-host, control_seconds=len(actions)*arena.control_dt)


def verify(rows, *, rgb=False):
    arena = JitteredOpening(**CONFIG)
    seen = set()
    try:
        for row in rows:
            key = row['seed'], digest_json(row['actions']), digest_json(row['rival_actions'])
            if key in seen:
                continue
            seen.add(key)
            arena.reset(seed=row['seed'], server=-1, render=False)
            digest = sha256()
            for tick, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True)):
                if rgb:
                    for eye, image in arena.render_cameras().items():
                        digest.update(eye.encode())
                        digest.update(image)
                arena._rival_attempt = True
                arena._aim = lambda *args: np.asarray(far)
                _, reward, done, capped, _ = arena.step(near, render=False)
                if bool(done or capped) != (tick+1 == len(row['actions'])):
                    raise ValueError('the two physical arms changed point duration')
            if (reward != row['reward'] or arena.point_result() != row['outcome'] or
                    digest_json(arena.events) != digest_json(row['events']) or
                    (rgb and digest.hexdigest() != row['rgb_sha256'])):
                raise ValueError(f'physical replay changed {row["seed"]}')
    finally:
        arena.close()
    return dict(condition_points=len(rows), independently_replayed=len(seen), rgb=rgb)


def insights(rows):
    eligible = [frame for row in rows for frame in row['optical'] if frame['eligible']]
    errors = [frame['error'] for frame in eligible if frame['error'] is not None]
    incoming_contacts = []
    for row in rows:
        pending = False
        for event in row['events']:
            if event['kind'] == 'legal_return' and event.get('side') == -1:
                pending = True
            if pending and event['kind'] == 'contact' and event.get('side') == 1:
                incoming_contacts.append(event['ball'][0])
                pending = False
    return dict(metrics(rows), anonymous_visible=sum(row['bootstrap_score'] is not None for row in rows),
                mover_frames=len(errors), possible_mover_frames=len(eligible),
                mover_within_15mm=sum(error <= .015 for error in errors),
                received_x_m=[later['xyz'][0] for row in rows for later in row['receiving']],
                contacted_incoming_x_m=incoming_contacts,
                planned_lanes=[plan['target'][0] for row in rows for plan in row['rival_plans']
                               if 'target' in plan])


def ending(row):
    """Audit scored openings, physical later replies and whether the learner hit."""
    opening = next((event for event in row['events'] if event['kind'] == 'legal_serve'), None)
    far_index = next((i for i in range(len(row['events'])-1, -1, -1)
                      if row['events'][i]['kind'] == 'legal_return' and row['events'][i].get('side') == -1), None)
    result = dict(seed=row['seed'], legal_opening=opening is not None, reason=row['outcome']['reason'],
                  won=row['reward'] > 0, last_legal_rival_reply=None, near_after_reply=None)
    if far_index is not None:
        result['last_legal_rival_reply'] = row['events'][far_index]['step']
        result['near_after_reply'] = any(event['kind'] == 'contact' and event.get('side') == 1
                                         for event in row['events'][far_index+1:])
        reach = next((item for item in row['receiving']
                      if item['reply_index'] == row['outcome']['rival_returns']), None)
        if reach is not None:
            result['late_reach'] = reach
            result['anonymous_error_at_reach'] = row['optical'][reach['tick']]['error']
    return result


def try_duration(path, previous):
    base = _read(previous)
    if base['verified']['condition_points'] != len(base['seeds'])*len(MODES):
        raise ValueError('the comparative source score trials are still pending')
    source = _read(DURATION_SOURCE)
    best = int(np.argmax(np.asarray(source['totals'])/source['counts']))
    coefficients = source['coefficients'][best]
    delta = np.asarray(base['learned_far_shift'])
    data = dict(schema='tt22-anonymous-native-duration-1', paired_sha256=_hash(previous),
                duration_source_sha256=_hash(DURATION_SOURCE), seeds=base['seeds'], coefficients=coefficients,
                arena=base['arena'], requested_lanes=base['requested_lanes'], rows=[])
    station = Station(delta)
    try:
        for index, seed in enumerate(base['seeds']):
            row = station.point(seed, coefficients)
            data['rows'].append(row)
            # With the shared opening, later feed differences originate after
            # the learner's changed stroke. Record any first-serve differences.
            serve = lambda r: next((event for event in r['events'] if event['kind']=='legal_serve'), None)
            same_serve = {name: digest_json(serve(rows[index])) == digest_json(serve(row))
                          for name, rows in base['branches'].items()}
            row['same_legal_opening'] = same_serve
            _write(path, data)
            print('duration', index+1, seed, row['longest'], row['reward'], row['outcome']['legal_returns'],
                  same_serve, flush=True)
    finally:
        station.close()
    data['summary'] = insights(data['rows'])
    data['endings'] = {name: [ending(row) for row in rows]
                       for name, rows in (*base['branches'].items(), ('duration', data['rows']))}
    data['verified'] = verify(data['rows'])
    data['rgb_verified'] = verify([max(data['rows'], key=lambda row: row['longest'])], rgb=True)
    _write(path, data)
    print('duration verified', data['verified'], flush=True)


def probe_late_contact(path, previous):
    """Hold one real late rival stroke and its entire causal prefix fixed."""
    base = _read(previous)
    original = max(base['branches']['family'], key=lambda point: point['longest'])
    late_landing = next(event for event in reversed(original['events'])
                        if event['kind']=='legal_return' and event.get('side') == -1)
    hit = next(event for event in reversed(original['events'])
               if event['kind']=='contact' and event.get('side') == -1 and event['step'] < late_landing['step'])
    cutoff = hit['step']
    source = _read(DURATION_SOURCE)
    winner = int(np.argmax(np.asarray(source['totals'])/source['counts']))
    variants = {'duration': (source['coefficients'][winner], False), 'geometry': ((0., 1., 1.), True)}
    data = dict(schema='tt22-anonymous-shared-late-1', paired_sha256=_hash(previous), original=original,
                cut_after_far_contact=cutoff, prior_legal_rival_reply=late_landing, branches={})
    station = Station(np.asarray(base['learned_far_shift']))
    try:
        for name, (coefficients, geometry) in variants.items():
            row = station.point(original['seed'], original['coefficients'],
                                switch=dict(tick=cutoff, coefficients=coefficients, geometry=geometry))
            for key in ('actions', 'rival_actions', 'optical'):
                if row[key][:cutoff] != original[key][:cutoff]:
                    raise ValueError(f'the original late incoming feed changed before {name}')
            row['same_legal_rival_landing'] = any(digest_json(event) == digest_json(late_landing)
                                                   for event in row['events'])
            data['branches'][name] = row
            _write(path, data)
            print('late contact', name, cutoff, row['longest'], row['reward'], row['outcome']['reason'], flush=True)
    finally:
        station.close()
    data['verified'] = verify([original, *data['branches'].values()])
    _write(path, data)


def duration_held(path, previous):
    base = _read(previous)
    seeds = np.random.default_rng(DURATION_HELD).integers(2**31, size=6).tolist()
    if len(set(seeds)) != len(seeds) or set(seeds).intersection(base['seeds']):
        raise ValueError('duration held seeds overlap the previous comparison')
    source = _read(DURATION_SOURCE)
    best = int(np.argmax(np.asarray(source['totals'])/source['counts']))
    original = base['source']
    old_winner = int(np.argmax(np.asarray(original['totals'])/original['counts']))
    arms = dict(duration=source['coefficients'][best],
                native_fixed=original['coefficients'][old_winner], geometry=(0., 1., 1.))
    data = dict(schema='tt22-anonymous-duration-held-1', arena=CONFIG, seed=DURATION_HELD, seeds=seeds,
                paired_sha256=_hash(previous), duration_source_sha256=_hash(DURATION_SOURCE),
                coefficients=arms, branches={name: [] for name in arms})
    station = Station(np.asarray(base['learned_far_shift']))
    order = list(arms)
    try:
        for index, seed in enumerate(seeds):
            for name in order[index % 3:] + order[:index % 3]:
                row = station.point(seed, arms[name], geometry=name == 'geometry')
                data['branches'][name].append(row)
                _write(path, data)
                print('duration held', index+1, seed, name, row['longest'], row['reward'],
                      row['outcome']['legal_returns'], flush=True)
    finally:
        station.close()
    data['summary'] = {name: insights(rows) for name, rows in data['branches'].items()}
    data['endings'] = {name: [ending(row) for row in rows] for name, rows in data['branches'].items()}
    rows = [row for branch in data['branches'].values() for row in branch]
    data['verified'] = verify(rows)
    data['rgb_verified'] = verify([max(rows, key=lambda row: row['longest'])], rgb=True)
    _write(path, data)
    print('duration held verified', data['verified'], flush=True)


def run(path, count):
    source = _read(SOURCE)
    qualification = _read(SOURCE_REFINEMENT)
    if source != qualification['portable_memory'] or _hash(SOURCE) != qualification['numeric_state_sha256']:
        raise ValueError('the score-selected native Pong prior changed')
    profiles = _read(RIVAL_SOURCE)['profiles']
    if profiles['fixed'] != list(ANCHOR):
        raise ValueError('the native far-arm reference profile changed')
    delta = np.asarray(profiles['learned'])-profiles['fixed']
    seeds = np.random.default_rng(GENERATOR).integers(2**31, size=count).tolist()
    common = broad_hypotheses(DRAW)
    exploration = np.random.default_rng(EXPLORE)
    ties = exploration.random(len(common)+len(source['coefficients']))
    shared_probe = int(exploration.integers(1,count))
    common_option = int(exploration.integers(len(common)))
    options = {name: SpatialChoice(common, source, 'pong' if name == 'family' else name, ties)
               for name in LEARNED_MODES}
    best = int(np.argmax(np.asarray(source['totals'])/source['counts']))
    data = dict(schema='tt22-anonymous-varied-lateral-1', arena=CONFIG, seed=GENERATOR, seeds=seeds,
                source_sha256=_hash(SOURCE), far_training_sha256=_hash(RIVAL_SOURCE), source=source,
                learned_far_shift=delta.tolist(), requested_lanes=(np.asarray(VARIED)+delta).tolist(),
                spread=SPREAD, pace=PACE, common=common.tolist(), shared_probe_index=shared_probe,
                shared_probe_option=common_option, branches={name: [] for name in MODES})
    station = Station(delta)
    memo = {}
    try:
        for ordinal, seed in enumerate(seeds):
            rotation = ordinal % len(MODES)
            for name in MODES[rotation:]+MODES[:rotation]:
                learner = options.get(name)
                selected = None
                if learner is not None:
                    selected = common_option if ordinal == shared_probe else learner.choose(explore=True)
                if name == 'geometry':
                    coefficients = (0., 1., 1.)
                elif learner is None:
                    coefficients = source['coefficients'][best]
                else:
                    coefficients = learner.coefficients[selected]
                key = seed, tuple(coefficients), name == 'geometry'
                if key not in memo:
                    row = station.point(seed, coefficients, geometry=name == 'geometry')
                    memo[key] = name, deepcopy(row)
                else:
                    previous, raw = memo[key]
                    row = dict(deepcopy(raw), cached_from=previous)
                if learner is not None:
                    learner.update(selected, row['reward'], row['outcome']['legal_returns'])
                    row.update(selected=selected, checkpoint=learner.state())
                data['branches'][name].append(row)
                _write(path, data)
                print('paired', ordinal+1, seed, name, selected, row['longest'], row['reward'],
                      row['outcome']['legal_returns'], len(row['receiving']), row['bootstrap_score'], flush=True)
    finally:
        station.close()
    data['summary'] = {name: insights(rows) for name, rows in data['branches'].items()}
    pairs = [row for rows in data['branches'].values() for row in rows]
    data['verified'] = verify(pairs)
    witness = max(pairs, key=lambda row: row['longest'])
    data['rgb_verified'] = verify([witness], rgb=True)
    _write(path, data)
    print('verified', data['verified'], 'RGB witness', witness['seed'], witness['longest'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    parser.add_argument('--count', type=int, choices=(4, 5, 6), default=4)
    parser.add_argument('--verify', action='store_true')
    parser.add_argument('--duration', action='store_true')
    parser.add_argument('--late-contact', action='store_true')
    parser.add_argument('--duration-held', action='store_true')
    args = parser.parse_args()
    args.folder.mkdir(exist_ok=True, parents=True)
    path = args.folder/'spatial-anonymous-varied.json.gz'
    if args.verify:
        if args.duration_held:
            saved = _read(args.folder/'spatial-anonymous-duration-held.json.gz')
            print(verify([row for rows in saved['branches'].values() for row in rows]))
        elif args.late_contact:
            data = _read(args.folder/'spatial-anonymous-late.json.gz')
            print(verify([data['original'], *data['branches'].values()]))
        elif args.duration:
            print(verify(_read(args.folder/'spatial-anonymous-duration.json.gz')['rows']))
        else:
            data = _read(path)
            print(verify([row for rows in data['branches'].values() for row in rows]))
    elif args.duration_held:
        duration_held(args.folder/'spatial-anonymous-duration-held.json.gz', path)
    elif args.late_contact:
        probe_late_contact(args.folder/'spatial-anonymous-late.json.gz', path)
    elif args.duration:
        try_duration(args.folder/'spatial-anonymous-duration.json.gz', path)
    else:
        run(path, args.count)
