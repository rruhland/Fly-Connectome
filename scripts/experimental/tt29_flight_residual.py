"""Learn incoming-flight errors from early and later external stereo RGB."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from .tt4_arena import TABLE_HEIGHT
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_partner_fast import NativeBallFlight
from .tt29_rally_physics import RallyArena, measured_launches
from .tt29_vision import Rig, VERTICAL, ball_pixels, flight_terms


PLANE = -.95
EARLY_LIMIT = -.20
SOURCE_DAMPING = 34.
UNSEEN_DAMPING = 42.
SEED = 29057917
OUTPUT = Path('artifacts/tt29/flight-residual.json.gz')
MOTOR_CHECKPOINT = Path('artifacts/tt29/camera-action-memory.json.gz')
LIVE_IDS = (4527, 11292)
NAMES = ('x_m', 'z_m', 'time_s', 'vx_mps', 'vy_mps', 'vz_mps')


def triangulate(rgb, cameras):
    """Measure one ball center from two synchronized ordinary camera frames."""
    left, right = (ball_pixels(rgb[key]) for key in ('left', 'right'))
    if left is None or right is None or left.clipped or right.clipped:
        return None
    a, b = cameras['left'], cameras['right']
    u, v = a.ray(left.center), b.ray(right.center)
    distances = np.linalg.lstsq(np.column_stack((u, -v)), b.eye - a.eye, rcond=None)[0]
    if min(distances) <= 0:
        return None
    p, q = a.eye + distances[0] * u, b.eye + distances[1] * v
    if np.linalg.norm(p - q) > .018:
        return None
    return (p + q) / 2


def early_state(samples):
    """Fit the still-air flight model to a sequence of pre-bounce stereo positions."""
    elapsed = samples[:, 0] - samples[-1, 0]
    travel, fall = flight_terms(elapsed, .69)
    design = np.column_stack((np.ones(len(travel)), travel))
    target = samples[:, 1:] - fall[:, None] * VERTICAL
    fit = np.linalg.lstsq(design, target, rcond=None)[0]
    if np.max(np.linalg.norm(design @ fit - target, axis=1)) > .032 or fit[1, 1] >= -2.:
        return None
    return fit


def nominal_crossing(path):
    if not len(path):
        return None
    crossing = np.flatnonzero((path[:-1, 2] > PLANE) & (path[1:, 2] <= PLANE))
    if not len(crossing):
        return None
    first, second = path[crossing[0]:crossing[0] + 2]
    alpha = (PLANE - first[2]) / (second[2] - first[2])
    point = (first + alpha * (second - first))
    return point[[1, 3, 0, 4, 5, 6]]


def observed_crossing(samples, before_time):
    """Interpolate the crossing from later direct stereo positions around it."""
    crossing = np.flatnonzero((samples[:-1, 2] > PLANE) & (samples[1:, 2] <= PLANE))
    if len(crossing) == 0:
        return None
    pivot = int(crossing[0])
    local = samples[max(0, pivot - 1):pivot + 4]
    if len(local) < 4 or local[0, 0] - before_time < .035 or np.max(np.diff(local[:, 0])) > .050:
        return None
    if local[:, 3].min() < .796 or not np.all(np.diff(local[:, 2]) < 0):
        return None
    # Use depth as the independent variable; estimate its crossing time and the
    # flight tangent with a local quadratic on the same observed samples.
    depth = local[:, 2] - PLANE
    design = np.stack((np.ones(len(depth)), depth, depth**2), axis=1)
    target = local[:, [1, 3, 0]].copy()
    target[:, 2] -= before_time
    fit = np.linalg.lstsq(design, target, rcond=None)[0]
    if np.max(np.linalg.norm(design @ fit - target, axis=1)) > .021 or fit[1, 2] >= -.05:
        return None
    velocity = np.r_[fit[1, 0], 1., fit[1, 1]] / fit[1, 2]
    return np.r_[fit[0], velocity]


def features(position, velocity, predicted):
    return np.array((1., position[0] / .32, (velocity[1] + 5.) / 2.,
                     velocity[2] / 2.2, (predicted[2] - .13) / .085, predicted[-1] / 2.))


def optical_trial(arena, cameras, nominal, shot):
    arena.launch(shot)
    observed = []
    initial = None
    attempted = False
    action = np.zeros(arena.action_size)
    rival = np.zeros(5)
    for tick in range(68):
        if tick % 2 == 0:
            p = triangulate(arena.render_cameras(), cameras)
            t = tick * arena.control_dt
            if p is not None:
                observed.append(np.r_[t, p])
                if not attempted and -.52 < p[1] < EARLY_LIMIT and len(observed) >= 4:
                    attempted = True
                    recent = np.asarray(observed[-5:])
                    if np.all(recent[:, 3] > .80) and recent[-1, 0] - recent[0, 0] <= .18:
                        now = early_state(recent)
                        if now is not None:
                            predicted = nominal_crossing(nominal.path(*now, np.zeros(3), 1))
                            if predicted is not None and predicted[2] >= .065:
                                initial = t, predicted, features(*now, predicted)
                if p[1] < PLANE - .42:
                    break
        _, _, done, truncated, _ = arena.exchange(action, rival)
        if done or truncated:
            break
    if initial is None:
        return dict(id=shot['id'], outcome='no_early_forecast', frames=len(observed))
    t, predicted, design = initial
    later = observed_crossing(np.asarray(observed), t)
    row = dict(id=shot['id'], outcome='paired' if later is not None else 'no_late_measurement', frames=len(observed),
               early_capture_s=t, forecast=predicted.tolist(), features=design.tolist())
    if later is not None:
        row['observed'] = later.tolist()
    return row


class FlightCorrection:
    """Regularized, robust increments trained on completed RGB measurements."""

    def __init__(self):
        self.rows = []
        self.coefficients = np.zeros((6, len(NAMES)))

    def learn(self, rows):
        self.rows.extend(row for row in rows if row['outcome'] == 'paired')
        if not self.rows:
            return
        design = np.array([row['features'] for row in self.rows])
        residual = np.array([np.subtract(row['observed'], row['forecast']) for row in self.rows])
        prior = np.diag(np.sqrt([1.5, 12., 10., 10., 8., 10.]))
        scale = np.array((.035, .035, .014, .25, .30, .38))
        for axis in range(len(NAMES)):
            observed = residual[:, axis]
            weight = np.ones(len(observed))
            for _ in range(3):
                root = np.sqrt(weight)
                value = np.linalg.lstsq(np.vstack((design * root[:, None], prior)),
                                        np.r_[observed * root, np.zeros(len(prior))], rcond=None)[0]
                weight = np.minimum(1., scale[axis] / np.maximum(1e-5, np.abs(observed - design @ value)))
            self.coefficients[:, axis] = value

    def delta(self, design):
        cap = np.array((.075, .075, .065, .55, .80, .85))
        return np.clip(design @ self.coefficients, -cap, cap)


def score(rows, correction=None):
    paired = [row for row in rows if row['outcome'] == 'paired']
    if not paired:
        return dict(paired=0, total=len(rows))
    residual = np.array([np.subtract(row['forecast'], row['observed']) for row in paired])
    if correction is None:
        delta = np.array([row['online_before'] for row in paired])
    else:
        delta = np.array([correction.delta(np.asarray(row['features'])) for row in paired])
    baseline = np.mean(np.abs(residual), axis=0)
    adjusted = np.mean(np.abs(residual + delta), axis=0)
    return dict(paired=len(paired), total=len(rows),
                nominal_mae=dict(zip(NAMES, baseline.tolist(), strict=True)),
                corrected_mae=dict(zip(NAMES, adjusted.tolist(), strict=True)))


def held_shift(matched, unseen):
    first = {row['id']: row['observed'] for row in matched if row['outcome'] == 'paired'}
    second = {row['id']: row['observed'] for row in unseen if row['outcome'] == 'paired'}
    common = sorted(first.keys() & second.keys())
    if not common:
        return dict(paired=0)
    changes = np.array([np.subtract(second[sid], first[sid]) for sid in common])
    return dict(paired=len(common),
                median_absolute_change=dict(zip(NAMES, np.median(np.abs(changes), axis=0).tolist(), strict=True)),
                median_signed_change=dict(zip(NAMES, np.median(changes, axis=0).tolist(), strict=True)))


def adjusted_flight(flight, nominal, correction):
    """Match the learned crossing change inside a small optical-state trust region."""
    if flight.bounced or not -.53 < flight.position[1] < -.13:
        return flight
    initial = np.r_[flight.position, flight.velocity]

    def forecast(state):
        return nominal_crossing(nominal.path(state[:3], state[3:], np.zeros(3), 1))

    base = forecast(initial)
    if base is None or not .06 < base[2] < .28:
        return flight
    delta = correction.delta(features(flight.position, flight.velocity, base))
    # The paired holdouts supported the vertical channels. Keep the other
    # crossing objectives at their nominal values for live planning.
    delta[[0, 2, 3, 4]] = 0.
    limits = np.array((.025, .035, .032, .24, .28, .34))
    scale = np.array((.015, .023, .012, .22, .30, .35))
    columns = []
    for step in np.diag(limits * .15):
        point = forecast(initial + step)
        if point is None:
            return flight
        columns.append((point - base) / (.15 * scale))
    derivative = np.column_stack(columns)
    weight = .35 * np.eye(len(limits))
    movement = np.linalg.lstsq(np.vstack((derivative, weight)), np.r_[delta / scale, np.zeros(len(limits))],
                               rcond=None)[0]
    adjusted = initial + limits * np.clip(movement, -1., 1.)
    check = forecast(adjusted)
    if check is None or np.linalg.norm((check - base - delta) / scale) >= np.linalg.norm(delta / scale):
        return flight
    return replace(flight, position=adjusted[:3], velocity=adjusted[3:])


def live(path, output, checkpoint):
    """Compare frozen versus residual-corrected optical planning on the unseen table."""
    from .tt29_action_memory import RecallPlanner
    from .tt29_endurance import audit
    from .tt29_material import memory_from
    from .tt29_partner import FOREHAND_LANES
    from .tt29_partner_fast import FastPartnerArena, FastPracticePartner
    from .tt29_vision import FlightPractice, OpticalFlight
    from .tt29_vision_twin import OpticalPlanner

    class Controller(OpticalPlanner):
        def __init__(self, twin, correction, active):
            super().__init__(twin, seed=290110)
            self.correction = correction
            self.active = active
            self.corrected = 0

        def _refresh(self, step, flight, angles, rate, target):
            if self.active:
                adjusted = adjusted_flight(flight, self.nominal.flight, self.correction)
                self.corrected += int(adjusted is not flight)
                flight = adjusted
            super()._refresh(step, flight, angles, rate, target)

    result = _read(output)
    for name in ('held_matched', 'held_unseen'):
        values = result['branches'][name]['summary']
        if any(values['corrected_mae'][metric] >= values['nominal_mae'][metric]
               for metric in ('z_m', 'vz_mps')):
            raise ValueError('The held RGB forecast did not pass the live-play gate')
    correction = FlightCorrection()
    correction.coefficients[:] = result['coefficients']
    learned = _read(checkpoint)
    practice_ids = {row['id'] for row in result['branches']['source']['rows']}
    motor_ids = {row['shot']['id'] for row in learned['rows']}
    if set(LIVE_IDS) & (practice_ids | motor_ids):
        raise ValueError('A live opening overlaps the practice course')
    launches = measured_launches(path)
    settings = dict(strike_ready='lateral', view_mode='fixed', width=256, height=192,
                    clearance_guard=True, fast_search=True, max_steps=220)
    twin = RallyArena(**settings)
    arena = FastPartnerArena(**settings, targets=FOREHAND_LANES, guard_table=True)
    arena.model.geom('table').solref[1] = -UNSEEN_DAMPING
    arena.partner = FastPracticePartner(arena, targets=FOREHAND_LANES, guard_table=True)
    fixed = Rig.from_arena(arena)
    rival = np.zeros(5)
    result['live'] = dict(checkpoint_sha256=_hash(checkpoint), return_cap=4,
                          corrected_targets=('z_m', 'vz_mps'), rows=[])
    try:
        for sid in LIVE_IDS:
            for active in (False, True):
                arena.launch(launches[sid])
                twin.reset(seed=sid, server=-1, render=False)
                vision = OpticalFlight(fixed, FlightPractice())
                policy = Controller(twin, correction, active)
                policy.nominal = RecallPlanner(twin, memory_from(learned), seed=290110)
                policy.nominal.flight = NativeBallFlight(twin)
                previous = arena.joints.copy()
                target = previous.copy()
                low, high = twin._joint_low[policy.actuators], twin._joint_high[policy.actuators]
                actions = []
                for step in range(arena.max_steps):
                    angle = arena.joints.copy()
                    t = step * arena.control_dt
                    vision.observe(t, arena.render_cameras(), angle)
                    flight = vision.latest
                    if (flight is not None and not policy.incoming and flight.velocity[1] < -.90
                            and flight.position[1] > -.16):
                        flight = None
                    action = policy.command(step, flight, angle, (angle - previous) / arena.control_dt, target)
                    previous = angle
                    target = np.clip(target + action * policy.nominal.rates * arena.control_dt, low, high)
                    _, _, done, capped, _ = arena.exchange(action, rival)
                    actions.append(action.tolist())
                    if done or capped or arena.legal_returns + arena.rival_returns >= 5:
                        break
                physical = audit(arena.events)
                result['live']['rows'].append(dict(id=sid, corrected=active, adjustments=policy.corrected,
                                                  score=physical, actions=actions, events=arena.events.copy()))
                _write(output, result)
                print('live', sid, active, policy.corrected, physical['returns'], flush=True)
    finally:
        arena.close()
        twin.close()


def run(path, output):
    launches = measured_launches(path)
    population = [s for s in launches.values() if s['pos_y'] > .18 and s['vel_y'] < -2.
                  and abs(s['pos_x']) < .58 and 1.0 < s['pos_z'] + TABLE_HEIGHT < 1.62]
    rng = np.random.default_rng(SEED)
    roster = rng.choice(len(population), 86, replace=False)
    source = [population[i] for i in roster[:46]]
    held = [population[i] for i in roster[46:]]
    arena = RallyArena(strike_ready='lateral', view_mode='fixed', width=256, height=192, max_steps=80)
    result = dict(dataset_sha256=_hash(path), seed=SEED, plane_y_m=PLANE, capture_hz=30,
                  nominal_damping=24., source_damping=SOURCE_DAMPING, unseen_damping=UNSEEN_DAMPING,
                  eligible_launches=len(population), branches={})
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        nominal = NativeBallFlight(arena)
        cameras = Rig.from_arena(arena).fixed
        correction = FlightCorrection()
        for name, shots, damping in (('source', source, SOURCE_DAMPING),
                                     ('held_matched', held, SOURCE_DAMPING), ('held_unseen', held, UNSEEN_DAMPING)):
            arena.model.geom('table').solref[1] = -damping
            rows = []
            for shot in shots:
                trial = optical_trial(arena, cameras, nominal, shot)
                rows.append(trial)
                if name == 'source':
                    if trial['outcome'] == 'paired':
                        trial['online_before'] = correction.delta(np.asarray(trial['features'])).tolist()
                    correction.learn([trial])
            summary = score(rows, None if name == 'source' else correction)
            result['branches'][name] = dict(rows=rows, summary=summary)
            result['coefficients'] = correction.coefficients.tolist()
            _write(output, result)
            print(name, summary, flush=True)
        result['held_optical_shift'] = held_shift(result['branches']['held_matched']['rows'],
                                                 result['branches']['held_unseen']['rows'])
        _write(output, result)
    finally:
        arena.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('feeds', type=Path)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--checkpoint', type=Path, default=MOTOR_CHECKPOINT)
    args = parser.parse_args()
    if args.live:
        live(args.feeds, args.output, args.checkpoint)
    else:
        run(args.feeds, args.output)
