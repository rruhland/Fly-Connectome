"""Validate an encoder map with a new movement and fit redundant probes on disagreement."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read, _write
from .tt24_bounce import BounceArena
from .tt24_donor_trial import ROOT
from .tt24_effects import transmission
from .tt24_integrated_stress import InstalledArm, schedule
from .tt24_motor_bank import I, STEP, decoder, sample
from .tt24_ready import prepare_step


BENCH = ROOT / 'validated-motor-bench.json.gz'
RELOCATION = ROOT / 'validated-motor-relocation.json.gz'
DEVELOPMENT = ROOT / 'validated-motor-development.json.gz'
CONTACTS = ROOT / 'validated-motor-contacts.json.gz'
VALIDATION_LIMIT = .125
AFTER_SEED = 449829455


class Probe:
    """Track cumulative requested increments and public encoder samples."""

    def __init__(self, joints, send):
        self.send = send
        self.joints = np.asarray(joints, float)
        self.request = np.zeros(5)
        self.history = []
        self.commands = []

    def step(self, command):
        self.joints = np.asarray(self.send(command), float)
        self.request += STEP*command
        self.history.append(self.joints.copy())
        self.commands.append(command.copy())
        return self.joints

    def go(self, endpoint):
        delta = endpoint-self.request
        moves = max(4, int(np.ceil(np.max(abs(delta))/(.49*STEP))))
        for _ in range(moves):
            self.step(delta/(STEP*moves))
        stable = 0
        for tick in range(30):
            previous = self.joints.copy()
            current = self.step(np.zeros(5))
            stable = stable+1 if np.max(abs(current-previous)) < .00045 else 0
            if tick >= 4 and stable >= 3:
                break
        return self.joints.copy()


def robust_affine(commands, joints, rng):
    """Trim endpoint errors independently by encoder channel, then refit inliers."""
    baseline = joints[0]
    x = np.column_stack((commands/.10, np.ones(len(commands))))
    y = joints-baseline
    indices = np.asarray([rng.choice(len(x), 6, replace=False) for _ in range(320)])
    designs = x[indices]
    spectrum = np.linalg.svd(designs, compute_uv=False)
    good = spectrum[:, -1] > .12
    minimum = int(np.ceil(.68 * len(x)))
    fits = [np.linalg.lstsq(x, y, rcond=None)[0]]
    if np.any(good):
        fits.extend(np.linalg.solve(designs[good], y[indices[good]]))
    fits = np.asarray(fits)
    loss = np.abs(np.einsum('nd,kdo->kno', x, fits) - y)
    trimmed = np.sort(loss, axis=1)[:, :minimum]
    best = np.argmin(np.mean(trimmed**2, axis=1), axis=0)
    result = np.empty((6, 5))
    for output, index in enumerate(best):
        fitted = fits[index, :, output]
        residual = abs(x@fitted-y[:, output])
        cutoff = max(.0018, 2.5 * float(np.median(residual)))
        inlier_weight = np.minimum(1., cutoff/np.maximum(residual,1e-12))**2
        w = np.sqrt(inlier_weight)
        result[:, output] = np.linalg.lstsq(w[:,None]*x, w*y[:,output], rcond=None)[0]
    return result[:5].T/.10


def probe_validation(motion, center, baseline, move):
    q_start = motion.joints.copy()
    q_out = motion.go(center+move)
    q_back = motion.go(center)
    prediction = baseline@move
    magnitude = np.linalg.norm(move)
    return [float(np.linalg.norm(q_out-q_start-prediction)/magnitude),
            float(np.linalg.norm(q_out-q_back-prediction)/magnitude),
            float(np.linalg.norm(q_back-q_start)/magnitude)]


def fits_locally(errors, spectrum):
    return max(errors) <= VALIDATION_LIMIT and spectrum[-1] >= .38 and spectrum[0] <= 1.65


def relocate(motion, baseline, rng):
    """Find a reachable pose with motion along the weak encoder direction."""
    output, _, inputs = np.linalg.svd(baseline)
    weak_joint, weak_request = output[:, -1], inputs[-1]
    center = motion.go(np.zeros(5))
    anchors = np.stack((.306*weak_request, -.306*weak_request))
    observed = np.stack([motion.go(point) for point in anchors])
    excursions = abs((observed-center)@weak_joint)
    index = int(np.argmax(excursions))
    anchor = anchors[index]
    samples = [motion.go(anchor)]
    basis = np.linalg.qr(rng.normal(size=(5,5)))[0].T
    offsets = np.stack((.068*basis, -.068*basis), axis=1).reshape(-1,5)
    points = np.vstack((np.zeros(5), offsets))
    samples.extend(motion.go(anchor+point) for point in offsets)
    recovered = robust_affine(points, np.asarray(samples), rng)
    center = motion.go(anchor)
    vector = rng.normal(size=5)
    errors = probe_validation(motion, anchor, recovered, .079*vector/np.linalg.norm(vector))
    spectrum = np.linalg.svd(recovered, compute_uv=False)
    trusted = bool(fits_locally(errors, spectrum))
    motion.go(np.zeros(5))
    return (recovered if trusted else baseline), dict(anchor=anchor.tolist(), anchor_q=center.tolist(),
                excursions=excursions.tolist(), selected_side=index, validation_errors=errors,
                singular_values=spectrum.tolist(), accepted=trusted)


def validated(joints, send, seed, known=(), *, recovery='local'):
    """Calibrate using only the requested command stream and observed joints."""
    rng = np.random.default_rng(int(seed)^AFTER_SEED)
    motion = Probe(joints, send)
    baseline, report = sample(joints, motion.step, seed, known)
    original_ticks = len(motion.history)
    prefix = np.cumsum(report['segment_ticks'])-1
    x = list(np.vstack((np.zeros(5), np.cumsum(report['x'], axis=0))))
    y = [motion.history[tick] for tick in prefix]
    center = motion.request.copy()
    diagonal = rng.normal(size=5)
    move = .091 * diagonal/np.linalg.norm(diagonal)
    errors = probe_validation(motion, center, baseline, move)
    spectrum = np.linalg.svd(baseline, compute_uv=False)
    unstable = not fits_locally(errors, spectrum)
    validated_ticks = len(motion.history)
    relocation = None
    if unstable and recovery == 'relocate':
        learned, relocation = relocate(motion, baseline, rng)
        label = None if relocation['accepted'] else report['known']
    elif unstable:
        basis = np.linalg.qr(rng.normal(size=(5,5)))[0].T
        # Independent mirrors free saturated directions while keeping excursions local.
        goals = np.vstack(([np.zeros(5)], .139*basis, -.139*basis,
                           [move, -move, np.zeros(5)]))
        for point in goals:
            x.append(point)
            y.append(motion.go(point))
        learned = robust_affine(np.asarray(x), np.asarray(y), rng)
        label = None
    else:
        learned = baseline
        label = report['known']
    return learned, dict(ticks=len(motion.history), known=label, fit=learned.tolist(), baseline=report,
                         validation_errors=errors, validation_ticks=validated_ticks-original_ticks,
                         refit=bool(unstable), refit_ticks=len(motion.history)-validated_ticks,
                         requested_offset=motion.request.tolist(), relocation=relocation,
                         x=np.asarray(x).tolist(), y=np.asarray(y).tolist())


def relocated(joints, send, seed, known=()):
    return validated(joints, send, seed, known, recovery='relocate')


def measurement(arm, location, plan, actuator, method, known=(I,)):
    arena = arm.arena
    obs = arm.reset(plan, location, render=False)
    q0 = obs['joints'].astype(float)
    actual, requested = [], []
    def send(command):
        drive = np.clip(actuator @ command, -1., 1.)
        requested.append(command.tolist())
        actual.append(drive.tolist())
        prepare_step(arena, drive, location)
        return arena._observation(False)['joints']
    fitted, info = method(q0, send, plan['calibration_rng'], known)
    initial_inspection_ticks = len(actual)
    net = np.sum(requested, axis=0)
    if np.linalg.norm(net) > 1e-7:
        for _ in range(4):
            send(-net/4)
        previous = arena._observation(False)['joints'].copy()
        settle = 0
        for _ in range(30):
            now = send(np.zeros(5))
            settle = settle+1 if np.max(abs(now-previous)) < .00045 else 0
            previous = now
            if settle >= 3:
                break
    final = arena._observation(False)['joints']
    raw = arena._observation(False)
    arm.reset(plan, location, render=False)
    for action in actual:
        prepare_step(arena, np.asarray(action), location)
    if not np.array_equal(raw['joints'], arena._observation(False)['joints']):
        raise ValueError('unscored raw mechanical replay changed the terminal encoders')
    inverse_error = float(np.linalg.norm(actuator@decoder(fitted)-I)/np.sqrt(5))
    return dict(info, relative_map_error=float(np.linalg.norm(fitted-actuator)/np.linalg.norm(actuator)),
                inverse_error=inverse_error, initial_q=q0.tolist(), final_q=final.tolist(),
                inspection_ticks=initial_inspection_ticks, all_ticks=len(actual),
                raw_commands=actual)


def holdouts(roster=195681240):
    rng = np.random.default_rng(roster+170240207)
    cases = schedule(seed=roster, per_body=28)
    for i, plan in enumerate(cases):
        plan['calibration_rng'] = int(rng.integers(1,2**31-1))
        actuator = transmission(rng)
        if i%3:
            actuator = actuator @ (I+rng.uniform(-.19,.19,(5,5)))
        yield plan, actuator


def summarize(rows):
    table = {}
    for mode in ('baseline','validated'):
        values = [row['modes'][mode] for row in rows]
        err = np.asarray([v['relative_map_error'] for v in values])
        inverse = np.asarray([v['inverse_error'] for v in values])
        charge = [v['all_ticks'] for v in values]
        table[mode] = dict(count=len(values), error_mean=float(err.mean()), error_median=float(np.median(err)),
                           inverse_mean=float(inverse.mean()), worst=float(err.max()),
                           errors_over_10pct=int((err>.1).sum()), errors_over_20pct=int((err>.2).sum()),
                           charge_mean=float(np.mean(charge)), charge_max=max(charge),
                           refits=sum(v.get('refit',False) for v in values))
    return table


def bench(output, recovery='local', *, development=False):
    arms = {body:InstalledArm(body) for body in ('stock','alternate')}
    server = BounceArena(view='none')
    rows = []
    try:
        if development:
            failures = [r for r in _read(BENCH)['cases'] if r['modes']['validated']['refit']]
            cases = [(r['plan'], np.asarray(r['judge_wiring'])) for r in failures]
        else:
            cases = holdouts(195681240 if recovery=='local' else 785091620)
        for plan, actuator in cases:
            server.reset(seed=plan['launcher_seed'],split='held',render=False)
            location = server._ball_position.copy()
            arm = arms[plan['body']]
            original = measurement(arm,location,plan,actuator,sample)
            improved = measurement(arm,location,plan,actuator,validated if recovery=='local' else relocated)
            rows.append(dict(plan=plan, judge_wiring=actuator.tolist(), modes=dict(baseline=original,validated=improved)))
            print('mechanical validation',plan['body'],plan['learner_seed'],
                  round(original['relative_map_error'],4),round(improved['relative_map_error'],4),
                  improved['refit'],original['all_ticks'],improved['all_ticks'],flush=True)
    finally:
        server.close()
        for arm in arms.values():
            arm.arena.close()
    _write(output, dict(cases=rows, summary=summarize(rows), method=recovery, development=development))
    print(summarize(rows), flush=True)


def contact_evidence(output):
    """Audit recorded commands after fit selection, without invoking the learner."""
    arms = {body:InstalledArm(body) for body in ('stock','alternate')}
    launcher = BounceArena(view='none')
    records = []
    try:
        for cohort, path in (('local',BENCH),('relocated',RELOCATION)):
            for row in _read(path)['cases']:
                if row['modes']['baseline']['relative_map_error']<=.2:
                    continue
                plan = row['plan']
                launcher.reset(seed=plan['launcher_seed'],split='held',render=False)
                position = launcher._ball_position.copy()
                arm = arms[plan['body']]
                arena = arm.arena
                modes = {}
                for tag,result in row['modes'].items():
                    arm.reset(plan,position,render=False)
                    touched = {}
                    paddle_table_frames = []
                    for tick,action in enumerate(result['raw_commands']):
                        prepare_step(arena,np.asarray(action),position)
                        active = {tuple(sorted(arena._model.geom(int(g)).name for g in contact.geom))
                                  for contact in arena._data.contact if contact.efc_address>=0}
                        if ('learner_racket','table') in active:
                            paddle_table_frames.append(tick)
                        for pair in sorted(active):
                            name = ' / '.join(pair)
                            touched[name] = touched.get(name,0)+1
                    if not np.array_equal(arena._observation(False)['joints'],result['final_q']):
                        raise ValueError('mechanical forensic replay changed the encoder endpoint')
                    modes[tag] = dict(contact_frame_counts=touched, paddle_table_frames=paddle_table_frames)
                records.append(dict(cohort=cohort, seed=plan['learner_seed'], body=plan['body'], modes=modes))
    finally:
        launcher.close()
        for arm in arms.values():
            arm.arena.close()
    _write(output,dict(evaluation_only=True,cases=records))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    options = parser.add_mutually_exclusive_group()
    options.add_argument('--relocate', action='store_true')
    options.add_argument('--development', action='store_true')
    options.add_argument('--contacts', action='store_true')
    args = parser.parse_args()
    if args.development:
        bench(args.output or DEVELOPMENT, 'relocate', development=True)
    elif args.relocate:
        bench(args.output or RELOCATION, 'relocate')
    elif args.contacts:
        contact_evidence(args.output or CONTACTS)
    else:
        bench(args.output or BENCH)


if __name__ == '__main__':
    main()
