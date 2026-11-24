"""Test local wrist-only sight recovery and stroke correction on matched hard openings."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _write
from .tt29_action_memory import context
from .tt29_rally_physics import desired_shot, measured_launches
from .tt29_vision import FlightPractice
from .tt29_wrist_only import BRACKET, OPPONENT, FastWristPartner, WristReceiver, capture_player, load_memory


OUT = Path('artifacts/tt29/tt29_wrist_local.json.gz')
MEMORY = OUT.with_name('tt29_wrist_only_carried.json.gz')
OPENS = (7345, 4197, 4230, 7271)
TARGET = np.array([0., 1.32, 1.08, 3.65])
WEIGHTS = ((.85, .20), (.70, .55), (.90, .85), (1.20, .50), (.0, .65), (.80, 0.))


def requested_angles(rates, start, speeds, dt, bounds):
    low, high = bounds
    target = start.copy()
    goals = []
    for command in rates:
        target = np.clip(target + command * speeds * dt, low, high)
        goals.append(target.copy())
    return np.stack(goals)


class LocalCorrection:
    """Try small pose corrections to an issued stroke using the wrist track and an isolated model."""

    def __init__(self):
        self.leg = None
        self.checks = 0
        self.last_check = -500
        self.previous = None
        self.trace = []
        self.sweeps = []
        self.last_seen = True
        self.search_start = None

    def retarget(self, pupil, step, flight, angles, rates, target):
        dt = pupil.twin.control_dt
        now = step * dt
        age = step - pupil.first
        has_plan = pupil.tape is not None and 0 < age < len(pupil.tape)
        if (flight is None or not pupil.incoming or now - flight.time > .020 or flight.depth_sigma > .073 or
                flight.rms > 1.35 or step - self.last_check < 5):
            return False
        if self.leg is not None and step - self.leg > 36:
            self.checks, self.leg, self.previous = 0, None, None
        if self.checks >= 2:
            return False
        p, v = flight.advance(now)
        if v[1] > -1.4 or p[1] > -.26 or p[1] < -1.03 or (not has_plan and p[1] > -.39):
            return False
        observed_bounce = pupil.bounced or (flight.bounced and p[1] < -.20)
        planner = pupil.nominal
        path = planner.flight.path(p, v, np.zeros(3), 1, observed_bounce, horizon=.48)
        available = path[(path[:, 0] >= .082) & (path[:, 0] < .31) & (path[:, 2] > -1.39) &
                         (path[:, 2] < -1.185) & (path[:, 3] > .87) & (path[:, 3] < 1.28)]
        if not len(available):
            return False
        model = pupil.twin
        bounds = model._joint_low[pupil.actuators], model._joint_high[pupil.actuators]
        if has_plan:
            base = requested_angles(pupil.tape[age:], target, planner.rates, dt, bounds)
        else:
            base = np.tile(target, (round(.86 / dt), 1))
        samples = np.linspace(0, len(available) - 1, min(6, len(available))).round().astype(int)
        shortlist = []
        for arrival in available[samples]:
            eta, point, velocity = arrival[0], arrival[1:4], arrival[4:7]
            k = min(len(base)-1, max(0, round(eta / dt)))
            center, normal, linear, angular = planner.ik.pose(base[k, :5], jacobian=True)
            miss = point - center - .026 * normal
            cost = np.linalg.norm(miss * [1.0, 1.75, 1.0]) + .10 * abs(point[1] + 1.29)
            shortlist.append((cost, eta, point, velocity, normal, linear, angular))
        miss, eta, point, velocity, normal, linear, angular = min(shortlist, key=lambda row: row[0])
        forecast = np.r_[point[[0, 2]], eta + now]
        if self.previous is not None and np.linalg.norm((forecast - self.previous) / [.018, .018, .014]) < 1.:
            return False
        state = round(eta / dt)
        nominal_center = planner.ik.pose(base[min(len(base)-1, state), :5])[0]
        desired = desired_shot(point, velocity, 1, (0., .98), 5.15)
        movement = np.clip(point - nominal_center - .026 * desired, -.080, .080)
        tilt = np.clip(desired - normal, -.26, .26)
        if (not has_plan and miss > .155) or (np.linalg.norm(movement) < .014 and np.linalg.norm(tilt) < .09):
            return False
        comp = np.concatenate((linear, .16 * angular, .046 * np.eye(5)))
        ramp = np.clip((np.arange(len(base)) + 1.) * dt / eta, 0., 1.)
        ramp = ramp * ramp * (3. - 2. * ramp)
        ramp *= np.clip((eta + .20 - (np.arange(len(base)) + 1.) * dt) / .13, 0., 1.)
        options = [base]
        for amount, yaw in WEIGHTS:
            dq = np.linalg.lstsq(comp, np.r_[amount * movement, .16 * yaw * tilt, np.zeros(5)], rcond=None)[0]
            candidate = base.copy()
            candidate[:, :5] += ramp[:, None] * np.clip(dq, -.24, .24)
            options.append(np.clip(candidate, *bounds))
        pupil._refresh(step, flight, angles, rates, target)
        model._received_bounce = observed_bounce
        began = perf_counter()
        commands, measures = planner.predict.trajectories(np.ascontiguousarray(options), TARGET, eta + .105)
        elapsed = perf_counter() - began
        scores = measures[:, 0]
        winner = int(np.argmin(np.where(np.isfinite(scores), scores, np.inf)))
        improve = bool(winner and np.isfinite(scores[winner]) and
                       (not np.isfinite(scores[0]) or scores[winner] < scores[0] - .06))
        self.leg = step if self.leg is None else self.leg
        self.last_check, self.previous = step, forecast
        self.checks += 1
        self.trace.append(dict(tick=step, eta_s=float(eta), point=point.tolist(), mismatch_m=float(miss),
                               proposed_delta=movement.tolist(), aim_delta=tilt.tolist(),
                               candidate_scores=[float(s) if np.isfinite(s) else None for s in scores],
                               selected=winner if improve else 0, verification_wall_s=elapsed,
                               from_issued_stroke=has_plan))
        if improve:
            pupil.first = step
            pupil.tape = commands[winner]
            # Credit the command that will actually be executed after this retarget.
            reference = context(path, -1.29)
            planner.pending = None if reference is None else (reference[0], options[winner], reference[1])
        return improve

    def search(self, pupil, eye, step, flight, target, command):
        now = step * pupil.twin.control_dt
        if pupil.tape is not None or (flight is not None and flight.velocity[1] < -1.):
            self.search_start = None
            return command
        if eye.last_frame_views:
            self.last_seen = True
            self.search_start = None
            return command
        if now - eye.last_read < .055 or (now - eye.last_read > .30 and step > 20 and not self.last_seen):
            return command
        if self.search_start is None:
            self.search_start = step
        if step - self.search_start >= 8:
            self.last_seen = False
            return command
        # A small inward pan opens the same forward sightline while waiting.
        goal = pupil.home[0] - .19
        command = command.copy()
        command[0] = np.clip((goal - target[0]) / (pupil.nominal.rates[0] * pupil.twin.control_dt), -.36, .36)
        self.sweeps.append(step)
        return command

    def report(self):
        return dict(checks=self.trace, gaze_ticks=self.sweeps,
                    replacements=sum(row['selected'] > 0 for row in self.trace))


def run(feeds, output, ids, memory, *, target, envelope, width, height, full_search=False):
    shots = measured_launches(feeds)
    options = dict(board=BRACKET, strike_ready='lateral', width=width, height=height, max_steps=360,
                   clearance_guard=True, fast_search=True, clearance_margin=.003,
                   robust_contact=True, motor_envelope=envelope)
    arena = FastWristPartner(**options, targets=OPPONENT, guard_table=True)
    cad = FastWristPartner(**options, targets=OPPONENT, guard_table=True, detail='cad')
    twin = WristReceiver(**options)
    report = dict(schema='tt29-wrist-local-paired-1', camera_size=[width, height], measured_sha256=_hash(feeds),
                  motor_sha256=_hash(memory), motor_envelope=envelope, sensed_streams=['wrist'],
                  full_search=full_search,
                  target=target, ids=list(ids), paired=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        for sid in ids:
            cases = {}
            for name in ('baseline', 'local'):
                row = capture_player(arena, twin, cad.model, shots[sid], course=FlightPractice(),
                                     memory=load_memory(memory), confidence=.112, scan=False, seed=sid,
                                     max_rally=target, min_frames=7, refine=False, late_retry=True,
                                     adjust=LocalCorrection() if name == 'local' else None,
                                     recall_only=not full_search)
                cases[name] = row
                print(sid, name, row['summary']['returns'], row['result']['reason'],
                      row['local_adjustment'], row['CAD']['penetration_samples'], flush=True)
            report['paired'].append(dict(id=sid, **cases))
            _write(output, report)
    finally:
        arena.close()
        cad.close()
        twin.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('feeds', type=Path)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--ids', nargs='+', type=int, default=OPENS)
    parser.add_argument('--memory', type=Path, default=MEMORY)
    parser.add_argument('--target', type=int, default=1)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--legacy-motor', action='store_true')
    parser.add_argument('--full-search', action='store_true')
    a = parser.parse_args()
    run(a.feeds, a.output, a.ids, a.memory, target=a.target, envelope=not a.legacy_motor,
        width=a.width, height=a.height, full_search=a.full_search)
