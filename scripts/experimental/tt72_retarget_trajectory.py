"""Retarget earned native paddle trajectories from the +12 cm clamp onto stock."""

import argparse
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
from scipy.optimize import least_squares, lsq_linear
from scipy.spatial.transform import Rotation, Slerp
from scipy.stats import qmc

from . import tt63_lateral_clamp as lateral
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import BladeIK, NOMINAL_NO_LOAD, Rollout
from .tt4_transfer_trial import _hash, _read, _write
from .tt42_learning_course import OPTIC

ROOT = Path('artifacts/tt72_retarget_trajectory')
PLAN, DONORS, STOCK = (ROOT / name for name in ('protocol.json', 'donors.json.gz', 'stock.json.gz'))
mpc = lateral.BASE
qual = lateral.coach
tape = qual.tape
IDS = (9254, 8831, 15456)
ZERO = [0.] * 6
DT = mpc.DT
SEEDS = (1821, 7252, 11210)


def contract():
    return dict(schema='tt72-retarget-1', source={str(p): _hash(p) for p in (
        lateral.DEVELOPMENT, lateral.PLAN, lateral.GEOMETRY, OPTIC)},
        ids=IDS, offset_m=.12, interior_m=.058, contact_offset_ticks=(-4, -2, 0, 2, 4),
        preparation_gains=(1., 1.45), remap_depth_m=(0.,),
        scope='Training-only native-state diagnosis. Keep original world delivery and original stock body. '
              'Retarget attained world blade center, signed full rotation and angular/linear twist throughout '
              'preparation, swing and immediate follow-through. Test native stock rates and actual landing and '
              'ordinary answer. Reserved releases are untouched unless a new repeatable stock interior succeeds.')


def receipt():
    dev, g = _read(lateral.DEVELOPMENT), _read(lateral.GEOMETRY)
    if not dev['complete'] or not g['complete']:
        raise ValueError('Complete lateral development before selecting teacher tapes')
    index = {(r['case']['id'], r['dx']): r for r in dev['rows']}
    ready = {r['dx']: r['ready'] for r in g['rows']}
    return ready, index


def blade(a):
    k = a._racket_ids[1]
    twist = np.empty(6)
    mujoco.mj_objectVelocity(a.model, a.data, mujoco.mjtObj.mjOBJ_GEOM, k, twist, 0)
    return dict(time=float(a.data.time), p=a.data.geom_xpos[k].tolist(),
                R=a.data.geom_xmat[k].reshape(3, 3).tolist(), twist=twist.tolist(),
                q=a.joints.tolist(), qd=a.data.qvel[a._near_dofs].tolist(),
                ref=a.data.ctrl[a._arm_ids[1][2]].tolist())


def capture(a, cam, rig, c, row, anchor, actions):
    tape.setup(a, c, row, anchor)
    eye = mpc.Vision(rig, a.control_dt, raw=True)
    eye.eye = mpc.LearnedOptics(eye.eye, OPTIC)
    original = a._racket_contact
    frames, impacts = [], []

    def hit(side, p, incoming):
        if side == 1:
            x = blade(a)
            impacts.append(dict(**x, ball=a.ball.tolist(), incoming=np.asarray(incoming).tolist(),
                                location=np.asarray(p).tolist()))
        return original(side, p, incoming)

    with patch.object(a, '_racket_contact', hit):
        for tick, rate in enumerate(actions):
            x = mpc.observed(eye, tick, cam.capture() if tick % 2 == 0 else None, a.joints.copy())
            frames.append(dict(**blade(a), rgb_pv=None if x is None else x.tolist()))
            _, _, end, cap, _ = a.exchange(rate, mpc.FAR_COMMAND)
            if end or cap or a.rival_returns >= 2:
                break
    return frames, impacts


def harvest():
    ROOT.mkdir(parents=True, exist_ok=True)
    if not PLAN.exists():
        _write(PLAN, contract())
    p = _read(PLAN)
    if dict(p, ids=tuple(p['ids']), contact_offset_ticks=tuple(p['contact_offset_ticks']),
            preparation_gains=tuple(p['preparation_gains']), remap_depth_m=tuple(p['remap_depth_m'])) != contract():
        raise ValueError('TT72 teacher contract changed')
    ready, index = receipt()
    d = _read(DONORS) if DONORS.exists() else dict(protocol=_hash(PLAN), rows=[], complete=False)
    if d['protocol'] != _hash(PLAN):
        raise ValueError('TT72 donor contract changed')
    with lateral.fixture.school.enabled(), lateral.slid(.12), mpc.centered(ready[.12]):
        a = mpc.arena()
        cam, rig, *_ = mpc.resources(a)
        try:
            for release in IDS[len(d['rows']):]:
                entry = index[release, .12]
                ref = next(r for r in entry['motion']['physical'] if qual.received(r, True))
                spec = entry['case']
                source = _read(Path(spec['source']))
                c, row, anchor = qual.get(source, spec)
                frames, impacts = capture(a, cam, rig, c, row, anchor, ref['commands'])
                impact = next(x for x in impacts if abs(x['time'] - ref['contact']['time_s']) < 1e-8)
                actual = digest_json(a.events)
                if actual != digest_json(ref['events']) or actual != ref['repeat_sha256']:
                    raise ValueError(f'Original shifted physical tape changed on {release}')
                d['rows'].append(dict(case=spec, fork=entry['motion']['fork'], capture=frames, impact=impact,
                                      contact=ref['contact'], event_sha256=actual,
                                      motor=a.motor_audit(), source_sha256=_hash(Path(spec['source']))))
                d['complete'] = len(d['rows']) == len(IDS)
                _write(DONORS, d)
                print('donor', release, 'frames', len(frames), 'RGB', sum(x['rgb_pv'] is not None for x in frames),
                      'contact mm', round(1000 * ref['contact']['radial_m'], 3), flush=True)
        finally:
            cam.close()
            a.close()


class FullPose(BladeIK):
    """Stock articulated geometry, with the actual signed disc coordinate frame."""

    def __init__(self, arena):
        super().__init__(arena)
        blade_id = self.model.geom('learner_racket')
        weld = self.model.eq_data[self.model.equality('tt19_grasp').id, 3:10]
        local = np.empty((2, 9))
        mujoco.mju_quat2Mat(local[0], weld[3:])
        mujoco.mju_quat2Mat(local[1], blade_id.quat)
        self.local_rotation = local[0].reshape(3, 3) @ local[1].reshape(3, 3)

    def frame(self, q, jac=False):
        d = self.data
        d.qpos[self.joints] = q
        mujoco.mj_kinematics(self.model, d)
        R = d.xmat[self.gripper].reshape(3, 3)
        center = d.xpos[self.gripper] + R @ self.center
        frame = R @ self.local_rotation
        if not jac:
            return center, frame
        mujoco.mj_comPos(self.model, d)
        mujoco.mj_jac(self.model, d, self.linear, self.angular, center, self.gripper)
        return center, frame, self.linear[:, self.dofs].copy(), self.angular[:, self.dofs].copy()

    def fit_frame(self, p, R, start, bounds=None, tangent=.022, stay=.0025, n=30):
        low, high = (self.low, self.high) if bounds is None else bounds
        weight = np.array([tangent, tangent, .108])
        last = None

        def state(q):
            nonlocal last
            if last is None or not np.array_equal(last[0], q):
                last = q.copy(), self.frame(q, True)
            return last[1]

        def error(q):
            center, frame, *_ = state(q)
            return np.r_[center - p, ((frame - R) * weight).T.ravel(), stay * (q - start)]

        def jac(q):
            _, frame, J, W = state(q)
            dR = np.concatenate([np.cross(W.T, f).T * w for f, w in zip(frame.T, weight)], axis=0)
            return np.vstack([J, dR, stay * np.eye(5)])

        result = least_squares(error, np.clip(start, low + 1e-8, high - 1e-8), jac=jac,
                               bounds=(low, high), max_nfev=n, ftol=2e-4, gtol=2e-5)
        return result.x, float(np.linalg.norm(result.fun))

    def velocity(self, q, twist, limit=NOMINAL_NO_LOAD):
        _, _, J, W = self.frame(q, True)
        target, jac = np.r_[twist[3:], .055 * twist[:3]], np.vstack([J, .055 * W])
        lo, hi = -np.broadcast_to(limit, (5,)).copy(), np.broadcast_to(limit, (5,)).copy()
        lo[q < self.low + .001], hi[q > self.high - .001] = 0., 0.
        return lsq_linear(jac, target, bounds=(lo, hi), tol=1e-7).x

    def errors(self, q, p, R, twist, qd=None):
        center, frame, J, W = self.frame(q, True)
        swing = np.rad2deg(Rotation.from_matrix(frame @ R.T).magnitude())
        face = np.rad2deg(np.arccos(np.clip(frame[:, 2] @ R[:, 2], -1., 1.)))
        if qd is None:
            # Optimistic stock bound, not a claim of loaded motor attainability.
            qd = self.velocity(q, twist)
        return [float(np.linalg.norm(p - center)), float(face), float(swing),
                float(np.linalg.norm(twist[3:] - J @ qd)), float(np.linalg.norm(twist[:3] - W @ qd))]


def ramp(progress):
    x = np.clip(progress, 0., 1.)
    return x * x * (3. - 2. * x)


class Course:
    """Retimed full world-space demonstration; terminal score not in the target."""

    def __init__(self, donor):
        d = donor['impact']
        contact_time = d['time']
        cut = int(round(d['time'] / DT))
        samples = [x for x in donor['capture'][donor['fork']:cut + 8] if abs(x['time'] - contact_time) > 1e-8]
        samples = sorted([*samples, d], key=lambda q: q['time'])
        self.ref_times = np.asarray([x['time'] for x in samples])
        self.position = np.asarray([x['p'] for x in samples])
        self.rotation = Rotation.from_matrix([x['R'] for x in samples])
        self.rotation = Slerp(self.ref_times, self.rotation)
        self.twist = np.asarray([x['twist'] for x in samples])
        self.impact = d
        self.contact_time = contact_time

    def target(self, t, offset, gain):
        original = t - offset
        lead = np.maximum(self.contact_time - original, 0.)
        # Time freedom for preparation while the last 120 ms retains the
        # measured disc speed and contact presentation.
        original = original - (gain - 1.) * lead * ramp((lead - .12) / .22)
        original = np.clip(original, self.ref_times[0], self.ref_times[-1])
        R = self.rotation(original).as_matrix()
        p = np.stack([np.interp(original, self.ref_times, q) for q in self.position.T], axis=1)
        T = np.stack([np.interp(original, self.ref_times, q) for q in self.twist.T], axis=1)
        # Clock variation follows the same freely incoming physical path over
        # a few ticks. The native rollout still adjudicates real collisions.
        delta = offset * np.asarray(self.impact['incoming']) + offset**2 * np.array([0., 0., -4.905])
        p += ramp((original - self.contact_time + .19) / .15)[:, None] * delta
        return p, R, T


def contact_fits(ik, course, home, seed):
    d = course.impact
    starts = ((ik.high - ik.low) * qmc.Sobol(5, scramble=True, seed=seed).random_base2(4) + ik.low)
    starts = np.vstack([home, d['q'][:5], starts])
    result = []
    for shift in (-4, -2, 0, 2, 4):
        tc = course.contact_time + DT * shift
        p, R, twist = (z[0] for z in course.target(np.asarray([tc]), DT * shift, 1.))
        for tangent in (.034, .008):
            fits = [ik.fit_frame(p, R, s, tangent=tangent, stay=.0001, n=48) for s in starts]
            q, _ = min(fits, key=lambda k: k[1])
            e = ik.errors(q, p, R, twist)
            result.append(dict(offset_ticks=shift, tangent=tangent, q=q.tolist(), error=e,
                               joint_saturation=[int(i) for i in np.flatnonzero(
                                   np.minimum(q - ik.low, ik.high - q) < .003)]))
    return result


def ribbon(ik, course, home, start, fit, gain, speed):
    offset = fit['offset_ticks'] * DT
    hit = course.contact_time + offset
    ticks = np.arange(start, int(np.ceil((hit + .13) / DT)) + 1)
    ts = ticks * DT
    contact = int(np.argmin(abs(ts - hit)))
    p, R, T = course.target(ts, offset, gain)
    q = np.empty((len(ticks), 5))
    # Fit both sides of the anchored swing. The lead-in must be reachable from
    # the actual stock encoder state as well as from the attained contact pose.
    q[contact] = fit['q']
    phase = fit.get('phase_deg')
    if phase is not None:
        R = R @ Rotation.from_euler('z', phase, degrees=True).as_matrix()
        desire = ik.velocity(q[contact], T[contact], speed)
    eps = 5e-6
    for j in (*range(contact - 1, -1, -1), *range(contact + 1, len(q))):
        k = j + 1 if j < contact else j - 1
        prev = q[k]
        lo = np.maximum(ik.low, np.maximum(prev - speed * DT, home - speed * DT * j))
        hi = np.minimum(ik.high, np.minimum(prev + speed * DT, home + speed * DT * j))
        if np.any(hi - lo < eps):
            # If that backward branch is time-infeasible, use earliest
            # feasible forward joint pose instead of widening motor bounds.
            q[j] = np.clip(prev, np.maximum(ik.low, home - speed * DT * j),
                           np.minimum(ik.high, home + speed * DT * j))
            continue
        start_at = prev
        regularize = .0025
        if phase is not None and abs(j - contact) < 5:
            start_at = q[contact] + (j - contact) * DT * desire
            regularize = .060 * (1. - abs(j - contact) / 5.)
        q[j], _ = ik.fit_frame(p[j], R[j], start_at, bounds=(lo, hi),
                               tangent=fit['tangent'], stay=regularize, n=13)
    q[0] = home
    return q, contact, p, R, T


def goals_for(ik, course, a, fits, gains=(1., 1.45), leads=(.023, .055, .086)):
    axes = a._arm_ids[1][2]
    q0 = a.joints[:5].copy()
    start = int(a.steps)
    horizon = int(np.ceil((course.contact_time - start * DT + .22) / DT)) + 62
    speed = np.minimum(a._joint_increments[axes[:5]] * a.rate_scale / a.model.opt.timestep, NOMINAL_NO_LOAD)
    candidates, notes = [], []
    for fit in fits:
        for gain in gains:
            q, hit, p, R, T = ribbon(ik, course, q0, start, fit, gain, speed)
            dq = np.gradient(q, DT, axis=0)
            e = np.array([ik.errors(row, p[j], R[j], T[j], dq[j]) for j, row in enumerate(q)])
            summary = dict(fit=dict(fit), preparation_gain=gain, hit=hit, frames=len(q),
                           ideal_preparation_m=float(np.median(e[:max(hit - 7, 1), 0])),
                           ideal_last120_ms_m=float(np.median(e[max(hit - 7, 0):hit + 1, 0])),
                           ideal_pose_at_hit=e[hit].tolist())
            for servo_lead in leads:
                ix = np.minimum(np.arange(1, horizon + 1), len(q) - 1)
                goal = np.c_[q[ix] + servo_lead * dq[ix], np.full(horizon, a.data.ctrl[axes[-1]])]
                goal[len(q) - 1:, :5] = q[-1]
                notes.append(dict(**summary, servo_lead_s=servo_lead))
                candidates.append(goal)
    return np.asarray(candidates), notes


def trial(roll, goals, contact_clock, a, ik, course, notes):
    n, frames = goals.shape[:2]
    fb = np.zeros((n, 9))
    trace = np.full((n, frames, 17), np.nan)
    nearest = np.full((n, 19), np.nan)
    cutoff = course.contact_time - a.steps * DT + .20
    actions, scores = roll.trajectories(goals, qual.AIM, cutoff, feedback=fb, traces=trace, approaches=nearest)
    loss, legal = mpc.cost(scores, fb)
    touch = nearest[:, 18] > .50
    radius = np.where(legal, scores[:, 12], fb[:, 8])
    interior = legal & (radius <= qual.INTERIOR)
    loss += np.where(fb[:, 1] > .5, 42. * np.maximum(radius - .050, 0.), 0.)
    pose_logs = []
    for j in range(n):
        ix = min(contact_clock[j], frames - 2)
        q = trace[j, :, 6:11]
        ok = np.isfinite(q[:, 0])
        end = min(np.sum(ok), notes[j]['frames'])
        select = np.unique(np.r_[np.arange(2, end - 1, 4), ix])
        select = select[(select > 0) & (select < end - 1)]
        if len(select):
            offset = notes[j]['fit']['offset_ticks'] * DT
            p, R, T = course.target((a.steps + select) * DT, offset, notes[j]['preparation_gain'])
            if notes[j]['fit'].get('phase_deg') is not None:
                R = R @ Rotation.from_euler('z', notes[j]['fit']['phase_deg'], degrees=True).as_matrix()
            errors = np.asarray([ik.errors(q[k], p[u], R[u], T[u], (q[k + 1] - q[k - 1]) / (2 * DT))
                                 for u, k in enumerate(select)])
            prep, play = errors[select < ix - 7], errors[select >= ix - 7]
            last = errors[np.argmin(abs(select - ix))]
            p0 = None if not len(prep) else np.median(prep, axis=0).tolist()
            p1 = None if not len(play) else np.median(play, axis=0).tolist()
            pose_logs.append(dict(preparation=p0, swing=p1, closest_clock=last.tolist()))
            loss[j] += 2. * (last[0] + .003 * last[1])
        else:
            pose_logs.append(None)
    # Distinguish a nearly faithful no-touch from a non-rebounding touch.
    rank = np.lexsort((loss, ~touch, ~legal, ~interior))
    return dict(actions=actions, scores=scores, fb=fb, nearest=nearest, losses=loss, legal=legal,
                interior=interior, touch=touch, rank=rank, pose=pose_logs)


def search(a, donor, seed, equiv=False):
    stock_pose = FullPose(a)
    course = Course(donor)
    if equiv:
        fits = rotated_fits(stock_pose, course, a.joints[:5], seed)
        goals, labels = goals_for(stock_pose, course, a, fits, gains=(.60, 1.0), leads=(.044, .076))
    else:
        fits = contact_fits(stock_pose, course, a.joints[:5], seed)
        goals, labels = goals_for(stock_pose, course, a, fits)
    roll = Rollout(a, 1)
    result = []
    iterations = []
    orig = goals.copy()
    rng = np.random.default_rng(seed)
    for stage in range(2 if equiv else 3):
        fs = np.asarray([r['hit'] for r in labels])
        z = trial(roll, goals, fs, a, stock_pose, course, labels)
        rank, legal, touch, interior, radius = (z[k] for k in ('rank', 'legal', 'touch', 'interior', 'scores'))
        log = dict(stage=stage, trials=len(goals), touch=int(touch.sum()), legal=int(legal.sum()),
                   legal_interior=int(interior.sum()),
                   minimum_legal_radius_m=float(min(radius[legal, 12])) if legal.any() else None)
        iterations.append(log)
        print('planned', donor['case']['id'], log, flush=True)
        record = list(rank[:12])
        if equiv:
            for shift in (-4, -2, 0, 2, 4):
                subset = [j for j in rank if labels[j]['fit']['offset_ticks'] == shift]
                if subset and subset[0] not in record:
                    record.append(subset[0])
        for j in record:
            notes = dict(**labels[j], round=stage, hit_in_model=bool(touch[j]), loss=float(z['losses'][j]),
                         legal=bool(legal[j]), interior=bool(interior[j]), attained=z['pose'][j],
                         closest=(z['nearest'][j].tolist() if np.isfinite(z['nearest'][j]).all() else None),
                         predicted_radius_m=float(radius[j, 12]) if legal[j] else None)
            result.append((z['actions'][j].copy(), notes))
        if stage == (1 if equiv else 2):
            break
        elite = rank[:14]
        if equiv:
            # Retain the best paired pose AND contact representation from
            # every clock, even if one oblique rim touch dominates proxy loss.
            for shift in (-4, -2, 0, 2, 4):
                subset = [j for j in rank if labels[j]['fit']['offset_ticks'] == shift]
                if subset:
                    best_track = min(subset, key=lambda j: float('inf') if z['pose'][j] is None else
                                     z['pose'][j]['closest_clock'][0] + .004 * z['pose'][j]['closest_clock'][1])
                    elite = np.unique(np.r_[elite, subset[0], best_track])
        sel = np.r_[elite, rng.choice(elite, size=56, replace=True)]
        goals, labels = goals[sel].copy(), [labels[j] for j in sel]
        # Smooth perturbations preserve the whole learned early posture
        # instead of constructing a new instantaneous paddle strike.
        base = np.arange(goals.shape[1])
        scale = np.array([.090, .105, .116, .123, .145]) * (1. if stage == 0 else .64)
        for j in range(len(elite), len(goals)):
            k = labels[j]['hit']
            knots = np.array([0, max(1, k - 16), k - 9, k - 4, k + 2, k + 8, len(base) - 1])
            raw = rng.normal(size=(len(knots), 5)) * scale
            raw[[0, -1]] = 0.
            goals[j, :, :5] += np.stack([np.interp(base, knots, r) for r in raw.T], axis=1)
    candidates = []
    for cmd, info in sorted(result, key=lambda c: (not c[1]['interior'], not c[1]['legal'],
                                                       not c[1]['hit_in_model'], c[1]['loss'])):
        if any(np.sqrt(np.mean((old[:50, :5] - cmd[:50, :5]) ** 2)) < .024 for old, _ in candidates):
            continue
        candidates.append((cmd, info))
        if len(candidates) >= 5:
            break
    if equiv:
        for shift in (-4, -2, 0, 2, 4):
            if any(m['fit']['offset_ticks'] == shift for _, m in candidates):
                continue
            subset = [(u, m) for u, m in result if m['fit']['offset_ticks'] == shift]
            if subset:
                candidates.append(min(subset, key=lambda c: c[1]['loss']))
    return fits, iterations, candidates, int(len(orig))


def first_ball(actual, expected):
    first = actual['placements']['delivered'][0]
    keep = ('far_contact', 'near_bounce')
    return all(digest_json(first[k]) == digest_json(expected[k]) for k in keep)


def rotated_fits(ik, course, home, seed):
    """Use a constant in-plane phase that preserves center and world twist."""
    d = course.impact
    starts = ((ik.high - ik.low) * qmc.Sobol(5, scramble=True, seed=seed).random_base2(3) + ik.low)
    starts = np.vstack([home, d['q'][:5], starts])
    fits = []
    for shift in (-4, -2, 0, 2, 4):
        tc = course.contact_time + DT * shift
        p, R, T = (z[0] for z in course.target(np.asarray([tc]), DT * shift, 1.))
        for phase in (-90, 0, 90, 180):
            axis = R @ Rotation.from_euler('z', phase, degrees=True).as_matrix()
            local = [ik.fit_frame(p, axis, start, tangent=.015, stay=.0001, n=40) for start in starts]
            q, _ = min(local, key=lambda v: v[1])
            fits.append(dict(offset_ticks=shift, phase_deg=phase, q=q.tolist(), tangent=.015,
                             error=ik.errors(q, p, axis, T),
                             joint_saturation=[int(i) for i in np.flatnonzero(
                                 np.minimum(q - ik.low, ik.high - q) < .003)]))
    return fits


def run(equiv=False):
    ready, index = receipt()
    sources = _read(DONORS)
    if not sources['complete'] or sources['protocol'] != _hash(PLAN):
        raise ValueError('Complete exact RGB/encoder harvest first')
    signatures = dict(protocol=_hash(PLAN), donors=_hash(DONORS))
    dest = STOCK
    if equiv:
        plan = ROOT / 'equivalence.protocol.json'
        alt = dict(primary_sha256=_hash(STOCK), donor_sha256=_hash(DONORS), phase_deg=[-90, 0, 90, 180],
                   preparation_gains=[.60, 1.0], servo_leads_s=[.044, .076],
                   trials='80 full paths and 56 smooth perturbations per case plus unchanged diverse elites',
                   boundary='Constant rotational phase about the actual blade normal preserves outer disc, target center '
                            'and world linear/angular twist but moves the handle. Fit local inverse twist for the last '
                            'four ticks on both sides of the impact. Include a motion from every timing family in '
                            'the native shortlist. Still diagnostic and still exact matched incoming.')
        if not plan.exists():
            _write(plan, alt)
        if _read(plan) != alt:
            raise ValueError('Functional phase comparison contract changed')
        signatures['phase_equivalence_protocol'] = _hash(plan)
        dest = ROOT / 'equivalence.json.gz'
    report = _read(dest) if dest.exists() else dict(**signatures, rows=[], complete=False)
    if any(report.get(k) != v for k, v in signatures.items()):
        raise ValueError('Changed teacher or receiver')
    with lateral.fixture.school.enabled(cameras=False), mpc.centered(ready[0.]):
        a, cad = mpc.arena(width=64, height=48), mpc.arena(width=64, height=48, detail='cad')
        guard = qual.GripClearance(cad.model, stock=True)
        try:
            for donor in sources['rows'][len(report['rows']):]:
                s = donor['case']
                doc = _read(Path(s['source']))
                c, old, anchor = qual.get(doc, s)
                neutral = [ZERO] * donor['fork']
                tape.setup(a, c, old, anchor)
                for cmd in neutral:
                    a.exchange(cmd, mpc.FAR_COMMAND)
                archived = [e for e in old['events'] if e['time'] <= a.data.time + 1e-8]
                if digest_json(a.events) != digest_json(archived):
                    raise ValueError(f'Stock does not reproduce exact prefix on {s["id"]}')
                fit, evolution, options, paths = search(a, donor, SEEDS[IDS.index(s['id'])], equiv=equiv)
                tests = []
                baseline = index[s['id'], 0.]['motion']
                ref = index[s['id'], .12]['motion']
                d = next(t for t in ref['physical'] if qual.received(t, True))
                archived_control = tape.matched_motion(a, c, old, anchor, d['commands'] + [ZERO] * 110, guard)
                if not first_ball(archived_control, d['placements']['delivered'][0]):
                    raise ValueError('The native stock word-for-word replay changed first impact or bounce')
                for cmd, info in options:
                    useful = min(len(cmd), info['hit'] + 13)
                    outcome = tape.matched_motion(a, c, old, anchor, neutral + cmd[:useful].tolist() + [ZERO] * 155, guard)
                    if not outcome['same'] or not first_ball(outcome, d['placements']['delivered'][0]):
                        raise ValueError('Whole-path test changed the physical initial offer')
                    outcome.update(prediction=info, identical_far_contact_and_bounce=True, motor=a.motor_audit())
                    score = qual.received(outcome, True)
                    tests.append(outcome)
                    print('stock physical', s['id'], 'legal', outcome['landed'], 'answered', outcome['answered'],
                          'contact', None if not outcome['contact'] else outcome['contact']['radial_m'],
                          'CAD', outcome['clearance']['penetration_samples'], flush=True)
                    if score:
                        replay = tape.matched_motion(a, c, old, anchor, outcome['commands'], guard)
                        if not qual.received(replay, True) or digest_json(outcome['events']) != digest_json(replay['events']):
                            raise ValueError('Unexpected interior result did not verify')
                        outcome['repeat_sha256'] = digest_json(replay['events'])
                        break
                report['rows'].append(dict(case=s, errors=['center_m','signed_face_deg','full_frame_deg','linear_m_s','angular_rad_s'],
                                           stock_qualification=[dict(landed=o['landed'], answered=o['answered'],
                                                                     contact=o['contact']) for o in baseline['physical']],
                                           signed_impact_fits=fit, planning=evolution, nominal_paths=paths,
                                           raw_shifted_command_control=archived_control, physically_tested=tests))
                report['complete'] = len(report['rows']) == len(IDS)
                _write(dest, report)
        finally:
            a.close()
            cad.close()


def measure():
    """Re-observe one clean real stock contact per release with the RGB/encoder apparatus."""
    docs = [STOCK, ROOT / 'equivalence.json.gz']
    data = [_read(p) for p in docs]
    if not all(x['complete'] for x in data):
        raise ValueError('Finish both motor comparisons before auditing achieved motion')
    path = ROOT / 'stock-witnesses.json.gz'
    checks = [_hash(p) for p in (*docs, DONORS)]
    report = _read(path) if path.exists() else dict(sources=checks, rows=[], complete=False)
    if report['sources'] != checks:
        raise ValueError('Stock witness tape provenance changed')
    ready_by_offset, model_cases = receipt()
    ready = ready_by_offset[0.]
    donors = _read(DONORS)['rows']
    with lateral.fixture.school.enabled(), mpc.centered(ready):
        a = mpc.arena()
        cam, rig, *_ = mpc.resources(a)
        try:
            for i in range(len(report['rows']), len(donors)):
                teacher = donors[i]
                spec = teacher['case']
                orig = next(t for t in model_cases[spec['id'], .12]['motion']['physical'] if qual.received(t, True))
                outcomes = [x for d in data for x in d['rows'][i]['physically_tested'] if
                            x['contact'] is not None and x['depart'] is not None and not x['clearance']['penetration_samples']]
                stop = np.asarray(orig['depart'])
                def key(x):
                    c = x['contact']
                    earned = x['landed'] and x['answered']
                    face = c['radial_m'] <= qual.INTERIOR and abs(c['face_alignment']) >= .75
                    return not earned, not face, np.linalg.norm(np.asarray(x['depart']) - stop)
                if not outcomes:
                    report['rows'].append(dict(case=spec, no_clean_contact=True))
                    continue
                win = min(outcomes, key=key)
                doc = _read(Path(spec['source']))
                card, row, anchor = qual.get(doc, spec)
                capture_rows, strikes = capture(a, cam, rig, card, row, anchor, win['commands'])
                if digest_json(a.events) != digest_json(win['events']):
                    raise ValueError('Native stock observation changed actual event tape')
                at = next(x for x in strikes if abs(x['time'] - win['contact']['time_s']) < 1e-8)
                report['rows'].append(dict(case=spec, impact=at, capture=capture_rows, contact=win['contact'],
                                           departure=win['depart'], ordinal_outgoing=orig['depart'],
                                           teacher_contact=orig['contact'], event_sha256=digest_json(a.events),
                                           radial_m=win['contact']['radial_m'], predictions=win['prediction']))
                report['complete'] = len(report['rows']) == len(donors)
                _write(path, report)
                print('measured stock', spec['id'], win['contact'], 'outgoing', win['depart'], flush=True)
        finally:
            cam.close()
            a.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('harvest', 'stock', 'equiv', 'measure'))
    args = p.parse_args()
    {'harvest': harvest, 'stock': run, 'equiv': lambda: run(equiv=True), 'measure': measure}[args.stage]()


if __name__ == '__main__':
    main()
