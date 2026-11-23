"""Identify remapped wrist actuation using causal dynamics learned during table tennis."""

import argparse
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt23_sequence import CONFIG, SequenceArena
from .tt24_bounce import BounceArena
from .tt24_donor_trial import Native
from .tt24_effects import transmission
from .tt24_forward_wiring import BallFree
from .tt24_public_body import FUSED_OUTPUT, load_public
from .tt24_rally_trial import PublicBody, STEP_SIZE


ROOT = Path('artifacts/tt24')
TABLE = ROOT / 'rally-improve-combined-held.json.gz'
PUPIL = ROOT / 'tt24_donor-refined.npz'
SOURCE_OUT = ROOT / 'forehand-carry-source.npz'
REPORT = ROOT / 'forehand-carry.json.gz'
RNG = 2027796531
SHUFFLE = 123876901
REMAPPINGS = 6
DROPS_PER_WIRING = 2
SOURCE_DT = 1. / 30.
STEP = STEP_SIZE[:5]
AMPLITUDE = .43
IDENTITY = np.eye(5)
MODES = ('nominal', 'hold', 'frozen', 'table_1', 'table_2', 'shuffled_1', 'shuffled_2', 'random_id_1',
         'random_id_2', 'known')


def _paddle(observer, joints):
    return np.stack([observer.body(position)[12:17] for position in joints])


def _source(path):
    corpus = _read(path)
    if (corpus['verified_tapes'] != sum(len(case['results']) for case in corpus['cases']) or
            corpus['metadata']['teacher_commands_in_assessment'] != 0):
        raise ValueError('expected verified full live points')
    arena = SequenceArena(**CONFIG)
    boards = []
    try:
        for entry in corpus['cases']:
            row = entry['results']['combined']
            arena.lane = row['lane']
            arena.reset(seed=row['seed'], server=-1, render=False)
            body = PublicBody(arena)
            near_actuators = arena._arm_ids[1][2][:5]
            own_rates = arena.model.jnt_dofadr[arena.model.actuator_trnid[near_actuators, 0]]
            q, velocity, requested, targets = [arena.joints[:5]], [arena.data.qvel[own_rates].copy()], [], []
            for count, (near, far) in enumerate(zip(row['actions'], row['rival_actions'], strict=True), 1):
                before = body.target[:5].copy()
                targets.append(before)
                body.issued(np.asarray(near))
                requested.append((body.target[:5] - before) / STEP)
                rates = np.asarray(far)
                arena._aim = arena._command_to = lambda *args, rates=rates: rates
                _, reward, done, cap, _ = arena.step(near, render=False)
                q.append(arena.joints[:5])
                velocity.append(arena.data.qvel[own_rates].copy())
                if bool(done or cap) != (count == len(row['actions'])):
                    raise ValueError(f'archived TT point changed its terminal tick on {row["seed"]}')
            if (reward != row['reward'] or digest_json(arena.events) != digest_json(row['events']) or
                    arena.point_result() != row['outcome']):
                raise ValueError(f'archived TT commands change referee events on {row["seed"]}')
            boards.append(dict(seed=row['seed'], q=np.asarray(q), velocity=np.asarray(velocity),
                               nominal_targets=np.asarray(targets), effective_requests=np.asarray(requested),
                               actual_actions=np.asarray(row['actions'])[:, :5]))
    finally:
        arena.close()
    return boards


def _features(positions, velocities, targets, action, previous, home):
    return np.c_[np.ones(len(positions)), velocities * SOURCE_DT, positions - home, targets - positions,
                 action * STEP, previous * STEP]


def _data(clips, *, shuffle=False):
    q, v, ctrl, u, previous, homes, future = ([] for _ in range(7))
    for clip in clips:
        now = clip['q']
        command = clip['effective_requests']
        q.extend(now[:-1])
        v.extend(clip['velocity'][:-1])
        ctrl.extend(clip['nominal_targets'])
        u.extend(command)
        previous.extend(np.vstack((np.zeros((1, 5)), command[:-1])))
        homes.extend(np.broadcast_to(now[0], command.shape))
        future.extend(now[1:])
    q, v, ctrl, u, previous, homes, future = map(np.asarray, (q, v, ctrl, u, previous, homes, future))
    if shuffle:
        order = np.random.default_rng(SHUFFLE).permutation(len(q))
        ctrl, u, previous = ctrl[order], u[order], previous[order]
    return _features(q, v, ctrl, u, previous, homes), future - q, q


def _fit(clips, *, shuffle=False):
    x, y, _ = _data(clips, shuffle=shuffle)
    scale = np.maximum(.008, x.std(axis=0))
    scale[0] = 1.
    scaled = x / scale
    penalties = np.r_[.4, [1.5]*10, [.55]*15]
    fit = np.linalg.solve(scaled.T @ scaled + np.diag(penalties), scaled.T @ y) / scale[:, None]
    noise = np.maximum(.0018, np.sqrt(np.mean((x @ fit-y)**2, axis=0)))
    return fit, noise


def _source_audit(clips, observer):
    # Hold out entire live points, including poor returns, for prediction only.
    excluded = [value for i, value in enumerate(clips) if i % 4 == 3]
    training = [value for i, value in enumerate(clips) if i % 4 != 3]
    x, y, q = _data(excluded)
    real = _paddle(observer, q + y) - _paddle(observer, q)
    baseline = np.vstack([np.vstack((np.zeros((1, 5)), np.diff(row['q'][:-1], axis=0))) for row in excluded])
    guess = {'persist_ip': baseline}
    for name, shuffled in [('table', False), ('shuffled', True)]:
        fit, _ = _fit(training, shuffle=shuffled)
        guess[name] = x @ fit
    details = {}
    for name, value in guess.items():
        predicted = _paddle(observer, q + value) - _paddle(observer, q)
        details[name] = dict(encoder_rms_mrad=1000.*float(np.sqrt(np.mean((y-value)**2))),
                             public_pose_displacement_rms_mm=1000.*float(np.sqrt(np.mean((real[:, :3]-predicted[:, :3])**2))),
                             public_normal_change_rms=float(np.sqrt(np.mean((real[:, 3:]-predicted[:, 3:])**2))))
    return dict(training_points=[row['seed'] for row in training], checked_points=[row['seed'] for row in excluded],
                checked_30_hz_frames=len(y), prediction=details)


def _save_source(path, clips, models):
    payload = {key: np.concatenate([clip[key] for clip in clips])
               for key in ('q', 'velocity', 'nominal_targets', 'effective_requests', 'actual_actions')}
    payload |= dict(offsets=np.r_[0, np.cumsum([len(x['q']) for x in clips])],
                    action_offsets=np.r_[0, np.cumsum([len(x['actual_actions']) for x in clips])],
                    seeds=np.array([entry['seed'] for entry in clips]))
    for name, (fit, noise) in models.items():
        payload[name], payload[name+'_noise'] = fit, noise
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + '.tmp.npz')
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def _probe_plan(rng):
    orthogonal, _ = np.linalg.qr(rng.normal(size=(5, 5)))
    signs = np.array((1., 1., -1., -1.))
    movements = (orthogonal.T[:, None, :] * (signs[:, None]*AMPLITUDE)[None, :, :]).reshape(-1, 5)
    return np.r_[np.zeros((4, 5)), movements, np.zeros((2, 5))]


def _probe(arena, seed, wiring, commands):
    obs = arena.reset(seed=seed, split='held', render=False)
    q, v = [obs['joints'].astype(float)], [obs['joint_velocity'].astype(float)]
    for desired in commands:
        for _ in range(2):
            obs, _, _ = arena.step(wiring @ desired, render=False)
        q.append(obs['joints'].astype(float))
        v.append(obs['joint_velocity'].astype(float))
    return dict(positions=np.asarray(q), velocities=np.asarray(v), commands=commands)


def _linear_equation(clip, fit, noise):
    q = clip['positions']
    v, u = clip['velocities'][:-1], clip['commands']
    integrated = np.vstack((np.zeros((1, 5)), np.cumsum(u * STEP, axis=0)))
    before = integrated[:-1]
    action = u * STEP
    previous = np.vstack((np.zeros((1, 5)), action[:-1]))
    nominal = np.broadcast_to(q[0], u.shape)
    baseline = _features(q[:-1], v, nominal, np.zeros_like(u), np.zeros_like(u), q[0]) @ fit
    # A fixed known role transform puts the matrix's rows in physical-joint
    # coordinates. Each requested axis remains unknown under the interface.
    jacobian = sum(np.einsum('jo,nk->nojk', layer, drive)
                   for layer, drive in ((fit[11:16], before), (fit[16:21], action), (fit[21:26], previous)))
    goal = np.diff(q, axis=0) - baseline
    return jacobian.reshape(len(u), 5, 25)/noise[None, :, None], goal/noise[None, :]


def _identification(probes, model):
    equations = [_linear_equation(probe, *model) for probe in probes]
    if len(probes) == 2:
        # Opposite matched launches cancel common unmodelled free drift.
        a, b = equations
        equations = [(a[0] * .6, a[1] * .6), (b[0] * .6, b[1] * .6),
                     ((a[0]-b[0]) / np.sqrt(2.), (a[1]-b[1]) / np.sqrt(2.))]
    x = np.concatenate([entry[0] for entry in equations]).reshape(-1, 25)
    y = np.concatenate([entry[1] for entry in equations]).ravel()
    penalty = 2.4
    return np.linalg.solve(x.T @ x + penalty*np.eye(25), x.T @ y + penalty*IDENTITY.ravel()).reshape(5, 5)


def _random_identification(probes):
    # Same random orthogonal probes. Fit a generic common servo delay from
    # the measured encoders, with no table-tennis or nominal plant data.
    best = None
    for unit in (0., .25, .45, .60, .72, .81, .88):
        histories, measured = [], []
        for probe in probes:
            target = np.cumsum(probe['commands'] * STEP, axis=0)
            filtered = np.zeros_like(target)
            prior = np.zeros(5)
            for index, value in enumerate(target):
                prior = unit*prior + (1. - unit)*value
                filtered[index] = prior
            histories.append(filtered)
            measured.append(probe['positions'][1:] - probe['positions'][0])
        if len(probes) == 2:
            histories.append((histories[0]-histories[1])/np.sqrt(2.))
            measured.append((measured[0]-measured[1])/np.sqrt(2.))
        x, y = np.concatenate(histories), np.concatenate(measured)
        ridge = .00085
        fit = np.linalg.solve(x.T @ x + ridge*IDENTITY, x.T @ y + ridge*IDENTITY)
        error = float(np.mean((y-x @ fit)**2))
        if best is None or error < best[0]:
            best = error, fit.T
    return best[1]


def _decoding(matrix):
    u, s, v = np.linalg.svd(matrix)
    return (v.T * (s/(s*s + .020))) @ u.T


def _play(arena, engine, weights, seed, wiring, decoder, *, hold=False, tape=None):
    live = not (hold or tape is not None)
    obs = arena.reset(seed=seed, split='held', render=live)
    observer = load_public(FUSED_OUTPUT) if live else None
    issued = np.zeros(5)
    request = np.zeros(5)
    recorded, actual = [], []
    rgb = sha256() if live else None
    for tick in range(arena.max_steps):
        if live:
            rgb.update(obs['rgb'])
            state = np.ascontiguousarray(observer.observe(obs), dtype=float)
            engine.command(state.ctypes.data, weights.ctypes.data, issued.ctypes.data, request.ctypes.data)
            physical = np.clip(wiring @ np.clip(decoder @ request, -1., 1.), -1., 1.)
            recorded.append(request.tolist())
        else:
            physical = np.zeros(5) if tape is None else np.asarray(tape[tick])
        actual.append(physical.tolist())
        obs, _, done = arena.step(physical, render=live)
        if tape is not None and bool(done) != (tick + 1 == len(tape)):
            raise ValueError('a physical replay changed its ending tick')
        if done:
            break
    return dict(metrics=arena.judge_metrics(), requested=recorded, actions=actual,
                rgb_sha256=rgb.hexdigest() if live else None)


def _summary(cases):
    values = {name: np.array([trial['conditions'][name]['metrics']['longest_consecutive_rebounds']
                              for trial in cases]) for name in MODES}
    table = {name: dict(mean=float(numbers.mean()), counts=numbers.tolist(), max=int(numbers.max()),
                        reaching_ten=int((numbers >= 10).sum())) for name, numbers in values.items()}
    differences = {}
    for count in (1, 2):
        for name in ('shuffled', 'random_id'):
            delta = values[f'table_{count}'] - values[f'{name}_{count}']
            differences[f'table_{count}_vs_{name}_{count}'] = dict(mean=float(delta.mean()), wins=int((delta>0).sum()),
                                                                  ties=int((delta==0).sum()), losses=int((delta<0).sum()))
    return dict(conditions=table, comparisons=differences, verified_tapes=len(cases)*len(MODES))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=REPORT)
    parser.add_argument('--source-output', type=Path, default=SOURCE_OUT)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    before = perf_counter()
    clips = _source(TABLE)
    calibration = load_public(FUSED_OUTPUT).decoder
    audit = _source_audit(clips, calibration)
    models = {key: _fit(clips, shuffle=key=='shuffled') for key in ('table', 'shuffled')}
    _save_source(args.source_output, clips, models)
    rng = np.random.default_rng(RNG)
    schedule = [dict(wiring=transmission(rng), probe_seed=int(rng.integers(2**31-1)),
                     commands=_probe_plan(rng), releases=rng.integers(2**31-1,size=DROPS_PER_WIRING).tolist())
                for _ in range(REMAPPINGS)]
    report = dict(schema='tt24-causal-forehand-carry-1', source_sha256=_hash(TABLE), source_archive=_hash(args.source_output),
                  source_live_points=len(clips), source_frames=sum(len(row['actual_actions']) for row in clips),
                  source_audit=audit, trained_before_target=True, dimensions=[192,144], target_motor_practice_drops=0,
                  policies_frozen=_hash(PUPIL), public_pose_and_runtime_sha256=_hash(FUSED_OUTPUT),
                  supplied_motor_encoder_role_order=['pan','shoulder','elbow','wrist flex','wrist roll'],
                  calibration_probe_frames_60_hz=2*len(schedule[0]['commands']),
                  regime='same stock geometry and physics, unseen dense actuator mixing', rng=RNG,
                  schedule=[{key: value.tolist() if isinstance(value, np.ndarray) else value
                             for key,value in case.items()} for case in schedule], cases=[], wiring_audit=[])
    _write(args.output, report)
    print('source', report['source_live_points'], report['source_frames'], audit, flush=True)
    with np.load(PUPIL, allow_pickle=False) as checkpoint:
        weights = np.ascontiguousarray(checkpoint['weights'], dtype=float)
    engine = Native()
    arena = BounceArena(view='wrist', width=192, height=144)
    probe_arena = BallFree('stock', report['calibration_probe_frames_60_hz'])
    try:
        for index, setup in enumerate(schedule):
            wire, cmd, seed = setup['wiring'], setup['commands'], setup['probe_seed']
            observed = [_probe(probe_arena, seed, wire, cmd*sign) for sign in (1., -1.)]
            estimated = {}
            for count in (1, 2):
                for source, model in models.items():
                    estimated[f'{source}_{count}'] = _identification(observed[:count], model)
                estimated[f'random_id_{count}'] = _random_identification(observed[:count])
            errors = {name: float(np.linalg.norm(estimate-wire)/np.linalg.norm(wire))
                      for name, estimate in estimated.items()}
            report['wiring_audit'].append(dict(index=index, estimates={k:v.tolist() for k,v in estimated.items()},
                                               relative_errors=errors,
                                               observations=[{k:v.tolist() for k,v in row.items()} for row in observed]))
            print('remap', index+1, errors, flush=True)
            decoders = {name: _decoding(matrix) for name,matrix in estimated.items()}
            decoders |= dict(known=np.linalg.inv(wire), nominal=IDENTITY, frozen=IDENTITY, hold=IDENTITY)
            for launch in setup['releases']:
                rows = {}
                for mode in MODES:
                    active = IDENTITY if mode=='nominal' else wire
                    outcome = _play(arena, engine, weights, launch, active, decoders[mode], hold=mode=='hold')
                    strict = _play(arena, engine, weights, launch, active, decoders[mode], tape=outcome['actions'])
                    if digest_json(outcome['metrics']) != digest_json(strict['metrics']):
                        raise ValueError(f'the {mode} physical replay changed its final result on {launch}')
                    rows[mode] = outcome
                report['cases'].append(dict(seed=launch, remap=index, conditions=rows))
                _write(args.output, report)
                print('held', launch, [row['metrics']['longest_consecutive_rebounds'] for row in rows.values()], flush=True)
    finally:
        arena.close()
        probe_arena.close()
        engine.working.cleanup()
    if _hash(PUPIL) != report['policies_frozen']:
        raise ValueError('the wrist motor policy changed during the test')
    report['summary'] = _summary(report['cases'])
    report['wall_seconds'] = perf_counter()-before
    _write(args.output, report)
    print(report['summary'], flush=True)


if __name__ == '__main__':
    main()
