"""Test a physical face change using native wrist RGB and public arm encoders."""

import argparse
from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import numpy as np

from .tt4_arena import BALL_RADIUS, mujoco
from .tt4_transfer_trial import _read, _write
from .tt19_so101 import BLADE, WRIST_CAMERA, _GRASP_POS
from .tt19_view_full_game import digest_json
from .tt29_arm_clearance import ArmClearance
from .tt29_conditions import Conditions, Delay
from .tt29_rally_physics import NOMINAL_NO_LOAD
from .tt29_vision import EncoderCamera, Lens, ball_pixels
from .tt29_wrist_only_warm import SoloBoard


OUTPUT = Path('artifacts/tt31/flip-practice.json.gz')
# Surveyed near-horizontal blade. The first face points toward the offset board.
READY = (.01689, -.49254, -.48437, .965259, 2.404858, -.1)
DONORS = tuple(range(310451, 310455))
HELD = tuple(range(310501, 310509))
MAX_TICKS = 125
ROLL_STEP = 5.2 / 60
REVERSED_ROLL = READY[4] - np.pi


class FlipBench(SoloBoard):
    """Use a stock clamp and board with the regulation motor torque-speed curve."""

    def __init__(self, *, detail='preview'):
        self.flip_detail = detail
        super().__init__()
        self.max_steps = MAX_TICKS

    def _document(self, width, height, detail, arm):
        return super()._document(width, height, self.flip_detail, arm)

    def sample_drop(self, seed, split, regime):
        rng = np.random.default_rng(seed)
        base = super().sample_drop(seed, split, regime)
        low = .19 if self.low_drop else 1.055
        height = rng.uniform(low, low + (.08 if self.low_drop else .09))
        spread = .005 if split == 'practice' else .011
        return replace(base, pose=READY, offset=(*rng.uniform(-spread, spread, 2), height),
                       velocity=(*rng.uniform(-.014, .014, 2), rng.uniform(-.055, .020)),
                       spin=tuple(rng.uniform(-1.7, 1.7, 3)), drag=rng.uniform(.95, 1.05),
                       damping=rng.uniform(28., 32.))

    def reset(self, *, seed, split, render, low_drop=False, check_state=None, no_ball=False,
              motor_envelope=True):
        self.low_drop = low_drop
        self.check_state = check_state
        self.no_ball = no_ball
        self.motor_envelope = motor_envelope
        self.faces = {}
        self.state_hash = sha256()
        self.roll_trace = []
        return super().reset(seed=seed, split=split, render=render)

    def _limit_motor(self):
        if not self.motor_envelope:
            return
        normalized = self._data.qvel[self._joint_velocity] / NOMINAL_NO_LOAD
        self._model.actuator_forcerange[:] = self._standard_force * np.column_stack((
            np.clip(1. + normalized, 0., 1.), np.clip(1. - normalized, 0., 1.)
        ))

    def _judge_substep(self, incoming):
        pre = self._paddle_contacts
        reward = 0. if self.no_ball else super()._judge_substep(incoming)
        normal = self._data.geom_xmat[self._blade].reshape(3, 3)[:, 2]
        roll = float(self._data.qpos[self._joint_position[4]])
        self.roll_trace.append(roll)
        if self._paddle_contacts > pre:
            self.faces[self._paddle_contacts] = dict(face=1 if normal[2] > 0 else -1,
                                                    normal_z=float(normal[2]), roll_rad=roll)
        self.state_hash.update(self._data.qpos.tobytes() + self._data.qvel.tobytes() +
                               self._data.ctrl.tobytes())
        if self.check_state is not None:
            self.check_state(self._data)
        self._limit_motor()
        return reward

    def step(self, action, *, render=True):
        self._limit_motor()
        try:
            return super().step(action, render=render)
        finally:
            self._model.actuator_forcerange[:] = self._standard_force

    def evidence(self):
        metrics = self.judge_metrics()
        events = [event | self.faces[event['impact_index']] for event in metrics.pop('rebound_events')]
        return dict(metrics=metrics, rebounds=events, contacts=list(self.faces.values()),
                    state_sha256=self.state_hash.hexdigest())


class FaceChange:
    """Estimate the first impact from the visible drop and roll while it is airborne."""

    def __init__(self, camera, blade_offset, *, advance=0, switch=True):
        self.camera = camera
        self.blade = camera.remount(blade_offset, np.eye(3)).at(READY).eye
        self.contact_height = self.blade[2] + BALL_RADIUS + .006
        self.advance = advance
        self.switch = switch
        self.sightings = []
        self.trigger = None
        self.target = READY[4]

    def locate(self, observation):
        sight = ball_pixels(observation['rgb'])
        if sight is None or sight.clipped:
            return None
        camera = self.camera.at(np.r_[observation['joints'], READY[-1]])
        direction = camera.ray(sight.center)
        horizontal = direction[:2]
        span = horizontal @ horizontal
        if span < .001:
            return None
        distance = (self.blade[:2] - camera.eye[:2]) @ horizontal / span
        if distance <= 0:
            return None
        pos = camera.eye + distance * direction
        if np.linalg.norm(pos[:2] - self.blade[:2]) > .022:
            return None
        return float(pos[2])

    def command(self, tick, observation):
        command = np.zeros(5)
        if self.trigger is None:
            height = self.locate(observation)
            if height is not None:
                self.sightings.append((tick, height))
                samples = np.asarray(self.sightings[-5:])
                if len(samples) >= 4 and .033 < height - self.contact_height < .25:
                    speed = np.polyfit(samples[:, 0] - tick, samples[:, 1], 1)[0]
                    if -.095 < speed < -.024:
                        impact = tick + (self.contact_height - height) / speed
                        self.trigger = int(np.ceil(impact)) + 1 - self.advance
        if self.switch and self.trigger is not None and tick >= self.trigger:
            command[4] = np.clip((REVERSED_ROLL - self.target) / ROLL_STEP, -1., 1.)
            self.target += command[4] * ROLL_STEP
        return command


def dry_turn(arena, *, motor_envelope=True):
    """Measure one half-turn under the same controller and motor limits."""
    arena.reset(seed=DONORS[0], split='practice', render=False, no_ball=True,
                motor_envelope=motor_envelope)
    # Keep the ball outside the disk while measuring a recipe that has no vision loop.
    arena._ball_position[:] = 3., 3., 4.
    mujoco.mj_forward(arena._model, arena._data)
    issued = []
    target = READY[4]
    for _ in range(100):
        u = np.zeros(5)
        u[4] = np.clip((REVERSED_ROLL - target) / ROLL_STEP, -1., 1.)
        target += u[4] * ROLL_STEP
        issued.append(u.tolist())
        _, _, done = arena.step(u, render=False)
        if done:
            break
    arrivals = {}
    for degrees in (15, 5, 2):
        arrivals[str(degrees)] = next(
            ((k + 1) * arena._model.opt.timestep for k, roll in enumerate(arena.roll_trace)
             if abs(roll - REVERSED_ROLL) < np.deg2rad(degrees)), None)
    return dict(target_change_deg=180., travel_limit_rad_s=5.2, nominal_bound_s=np.pi / 5.2,
                arrival_within_deg_s=arrivals,
                peak_measured_rad_s=float(np.max(np.abs(np.diff(arena.roll_trace))) / arena._model.opt.timestep),
                actions=issued)


def rollout(arena, camera, blade_offset, seed, split, *, delay=0, advance=0, switch=True,
            low_drop=False, tape=None, callback=None, motor_envelope=True):
    controller = FaceChange(camera, blade_offset, advance=advance, switch=switch) if tape is None else None
    queue = Delay(Conditions(command_delay=delay), seed)
    obs = arena.reset(seed=seed, split=split, render=tape is None, low_drop=low_drop,
                      check_state=callback, motor_envelope=motor_envelope)
    if callback is not None:
        callback(arena._data)
    issued, applied = [], []
    encoders = [obs['joints'].tolist()]
    for tick in range(arena.max_steps):
        if tape is None:
            command = controller.command(tick, obs)
            issued.append(command.tolist())
            actual = queue.command(command)
        else:
            actual = tape[tick]
        applied.append(np.asarray(actual).tolist())
        obs, _, done = arena.step(actual, render=tape is None)
        encoders.append(obs['joints'].tolist())
        if done:
            break
    result = arena.evidence() | dict(applied=applied, encoders=encoders)
    if controller is not None:
        result.update(issued=issued, trigger=controller.trigger, sightings=controller.sightings)
    return result


def infer_delay(reference, delayed, maximum=4):
    """Compare command-aligned encoder motion windows using no ball or referee values."""
    window = 26

    def response(row):
        first = next((t for t, action in enumerate(row['issued']) if abs(action[4]) > .5), None)
        if first is None or len(row['encoders']) < first + window:
            raise ValueError('roll identification requires a complete response window')
        q = np.asarray(row['encoders'])[first:first + window, 4]
        return q - q[0]

    if not reference or len(reference) != len(delayed) or not 0 <= maximum < window:
        raise ValueError('expected matched episodes and a bounded lag search')
    before = np.stack([response(row) for row in reference])
    after = np.stack([response(row) for row in delayed])
    if np.max(np.ptp(before, axis=1)) < .15 or np.max(np.ptp(after, axis=1)) < .15:
        raise ValueError('roll identification requires observable motor response')
    count = window - maximum
    losses = [float(np.mean(np.square(before[:, :count] - after[:, lag:lag + count])))
              for lag in range(maximum + 1)]
    return int(np.argmin(losses)), losses


def summary(rows):
    consecutive = []
    first_airborne = []
    for row in rows:
        rebounds = row['rebounds']
        consecutive.append(sum(first['face'] != second['face'] and
                               second['impact_index'] == first['impact_index'] + 1
                               for first, second in zip(rebounds, rebounds[1:])))
        if rebounds and rebounds[0]['airborne_s'] is not None:
            first_airborne.append(rebounds[0]['airborne_s'])
    return dict(drops=len(rows), qualified_rebounds=sum(len(row['rebounds']) for row in rows),
                with_opposite_face=sum(count > 0 for count in consecutive),
                alternating_pairs=sum(consecutive),
                median_first_airborne_s=float(np.median(first_airborne)) if first_airborne else None)


def run(output):
    arena = FlipBench()
    try:
        model = arena._model
        lens = Lens(arena.width, arena.height, float(model.camera(WRIST_CAMERA).fovy[0]))
        camera = EncoderCamera.from_model(model, lens)
        offset = _GRASP_POS + model.geom(BLADE).pos
        dry = dict(coupled=dry_turn(arena), independent=dry_turn(arena, motor_envelope=False))
        # A known immediate channel provides the transmitted-command response reference.
        reference = [rollout(arena, camera, offset, seed, 'practice') for seed in DONORS]
        hidden_delay = 2
        delayed = [rollout(arena, camera, offset, seed, 'practice', delay=hidden_delay)
                   for seed in DONORS]
        lag, losses = infer_delay(reference, delayed)
        fixed = dict(stock=dict(), fixed_face=dict(switch=False), short_arc=dict(low_drop=True),
                     independent_limit=dict(motor_envelope=False),
                     delayed=dict(delay=hidden_delay),
                     carried=dict(delay=hidden_delay, advance=lag))
        held = {label: [rollout(arena, camera, offset, seed, 'held', **settings) for seed in HELD]
                for label, settings in fixed.items()}
        result = dict(schema='tt31-face-practice-1', joint_range_rad=model.joint('learner_wrist_roll').range.tolist(),
                      control_dt_s=arena.dt, physics_dt_s=model.opt.timestep, board_fovy_deg=lens.fovy,
                      no_load_rad_s=NOMINAL_NO_LOAD, dry_turn=dry, issued_delay_ticks=hidden_delay,
                      carried_lag=lag, delay_losses_rad2=losses, source_reference=reference,
                      source_delayed=delayed, held=held, summary={label: summary(rows) for label, rows in held.items()})
        _write(output, result)
        print({k: v['arrival_within_deg_s'] for k, v in dry.items()}, lag, losses, result['summary'], flush=True)
    finally:
        arena.close()


def verify(output):
    """Replay every command and check the full CAD along switched trajectories."""
    report = _read(output)
    arena = FlipBench()
    cad = FlipBench(detail='cad')
    guard = ArmClearance(cad._model, tolerance=0.)
    shadow = mujoco.MjData(cad._model)
    samples = violations = 0
    nearest = None

    def check(state):
        nonlocal samples, violations, nearest
        shadow.qpos[:] = state.qpos
        observed = min(guard.distances(shadow, horizon=.06), default=None, key=lambda pair: pair[2])
        samples += 1
        if observed is not None:
            violations += observed[2] < 0.
            if nearest is None or observed[2] < nearest[2]:
                nearest = observed

    groups = (
        ('practice', report['source_reference'], 'stock'),
        ('practice', report['source_delayed'], 'delayed'),
        *(('held', rows, label) for label, rows in report['held'].items()),
    )
    count = 0
    try:
        for split, rows, label in groups:
            for row in rows:
                replay = rollout(arena, None, None, row['metrics']['seed'], split,
                                 low_drop=label == 'short_arc', motor_envelope=label != 'independent_limit',
                                 tape=row['applied'], callback=check)
                for name in ('metrics', 'rebounds', 'contacts', 'state_sha256', 'encoders', 'applied'):
                    if digest_json(replay[name]) != digest_json(row[name]):
                        raise ValueError(f'Solo replay changed {name} at seed {row["metrics"]["seed"]}')
                count += 1
        report.update(exact_replays=count, CAD=dict(samples=samples, penetration_samples=violations,
                                                    nearest=nearest))
        _write(output, report)
        print('exact replays', count, report['CAD'], flush=True)
    finally:
        arena.close()
        cad.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.verify_only:
        run(args.output)
    verify(args.output)
