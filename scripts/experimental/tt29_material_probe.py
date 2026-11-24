"""Measure effective contact response from RGB-observed controlled drops."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from .tt4_arena import TABLE_HEIGHT, mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_so101 import BLADE
from .tt24_bounce import _HOME
from .tt29_conditions import CONDITIONS, apply
from .tt29_fast_camera import CachedStereoRenderer
from .tt29_flight_residual import triangulate
from .tt29_rally_physics import RallyArena, measured_launches
from .tt29_vision import Rig


OUTPUT = Path('artifacts/tt29/rgb-material-probe.json.gz')
MOTOR = Path('artifacts/tt29/camera-action-memory.json.gz')
SURFACES = {'table': 'table', 'paddle': BLADE}
HYPOTHESES = (18., 24., 31., 39., 47.)
SOURCE_HEIGHTS = (.20, .32)
HELD_HEIGHTS = (.17, .25, .36)
LIVE = (('materials', 4527), ('compound', 11292))
STEREO = ('left', 'right')
SIZE = (384, 288)


class ProbeArena(RallyArena):
    """Park the near arm in its horizontal solo posture for a commanded drop."""

    def _arm_home(self, side):
        if side == 1:
            return _HOME, ()
        return super()._arm_home(side)


def observed_ratio(samples):
    """Fit and intersect long optical arcs either side of the first rebound."""
    rows = np.asarray(samples)
    if len(rows) < 16:
        return None
    time, z = rows[:, 0], rows[:, 3]
    for i in range(7, len(z) - 8):
        if not (z[i] <= z[i - 1] and z[i] < z[i + 1] and z[i - 5] > z[i] + .035
                and z[i + 5] > z[i] + .028):
            continue
        left = np.arange(i - 7, i - 1)
        right = np.arange(i + 2, i + 8)
        if (np.max(np.diff(time[np.r_[left, i]])) > .055
                or np.max(np.diff(time[np.r_[i, right]])) > .055):
            continue
        t = time - time[i]
        before = np.polyfit(t[left], z[left], 2)
        after = np.polyfit(t[right], z[right], 2)
        roots = np.roots(after - before)
        different = [float(value.real) for value in roots
                     if abs(value.imag) < 1e-6 and abs(value.real) < .028]
        if not different:
            continue
        impact = min(different, key=abs)
        incoming = np.polyval(np.polyder(before), impact)
        outgoing = np.polyval(np.polyder(after), impact)
        ratio = -outgoing / incoming
        if incoming < -1. and .50 < ratio < 1.18:
            return dict(impact_speed_mps=float(-incoming), optical_restitution=float(ratio),
                        impact_time_s=float(time[i] + impact), center_height_m=float(np.polyval(before, impact)))
    return None


def probe(arena, render, cameras, surface, height):
    arena.reset(seed=16191, render=False)
    if surface == 'table':
        release = np.array((.26, -.65, TABLE_HEIGHT + .020))
    else:
        release = arena.data.geom(BLADE).xpos.copy() + (0., 0., .026)
    # A repeatable physical launch is the test input. Only its native camera
    # frames and capture times go to the fit below.
    arena.ball[:] = release + (0., 0., height)
    arena.velocity[:] = (0., 0., -.03)
    arena.data.qvel[arena._ball_dof + 3:arena._ball_dof + 6] = 0.
    mujoco.mj_forward(arena.model, arena.data)
    substeps = round(arena.control_dt / arena.model.opt.timestep)
    samples = []
    for step in range(41):
        position = triangulate(render.capture(), cameras)
        if position is not None:
            samples.append(np.r_[step * arena.control_dt, position].tolist())
        for _ in range(substeps):
            arena._aerodynamics()
            mujoco.mj_step(arena.model, arena.data)
    return dict(height_m=height, surface=surface, samples=samples, measured=observed_ratio(samples))


def profile(name):
    material = CONDITIONS['materials']
    if name == 'compound':
        return replace(material, ball_mass=.92, table_damping=38., paddle_damping=39.,
                       paddle_friction=.39, paddle_mass=1.12, elbow_mass=1.14,
                       outside_shift_m=.06, camera_yaw_deg=-4.)
    return CONDITIONS[name]


def estimate(probes, library, surface):
    source = [row for row in probes if row['surface'] == surface and row['measured'] is not None]
    course = [row for row in library if row['surface'] == surface and row['measured'] is not None]
    candidates = np.asarray([value for value in HYPOTHESES if any(row['damping'] == value for row in course)])
    if len(source) < 2 or len(candidates) < 3:
        return None
    expected_ratio = []
    for damping in candidates:
        rows = [row['measured'] for row in course if row['damping'] == damping]
        pairs = sorted((row['impact_speed_mps'], row['optical_restitution']) for row in rows)
        speeds, ratios = np.asarray(pairs).T
        expected_ratio.append(np.interp([row['measured']['impact_speed_mps'] for row in source], speeds, ratios))
    grid = np.linspace(candidates[0], candidates[-1], 117)
    ratios = np.stack([np.interp(grid, candidates, row) for row in np.asarray(expected_ratio).T], axis=1)
    observed = np.array([row['measured']['optical_restitution'] for row in source])
    residual = np.abs(ratios - observed)
    losses = np.mean(np.minimum(residual, .09), axis=1)
    prior = .002 * ((grid - 25.) / 15.)**2
    winner = int(np.argmin(losses + prior))
    return dict(effective_damping=float(grid[winner]), ratio_mae=float(losses[winner]),
                observed_ratios=observed.tolist(), hypotheses=len(candidates))


def collect(output):
    settings = dict(width=SIZE[0], height=SIZE[1], view_mode='fixed')
    source = ProbeArena(**settings)
    renderer = CachedStereoRenderer(source)
    result = dict(dimensions=SIZE, hypotheses=HYPOTHESES, source=[], held={})
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        cameras = Rig.from_arena(source).fixed
        for surface, geom in SURFACES.items():
            contact = source.model.geom(geom)
            original = float(contact.solref[1])
            try:
                for damping in HYPOTHESES:
                    contact.solref[1] = -damping
                    for height in SOURCE_HEIGHTS:
                        row = probe(source, renderer, cameras, surface, height)
                        row['damping'] = damping
                        result['source'].append(row)
            finally:
                contact.solref[1] = original
        _write(output, result)
    finally:
        renderer.close()
        source.close()
    from .tt29_registration import register

    for name in ('stock', 'materials', 'compound'):
        arena = ProbeArena(**settings)
        nominal = Rig.from_arena(arena)
        apply(arena, profile(name))
        renderer = CachedStereoRenderer(arena)
        try:
            frames = renderer.capture()
            cameras = {eye: register(frames[eye], nominal.fixed[eye]).camera for eye in STEREO}
            rows = [probe(arena, renderer, cameras, surface, height)
                    for surface in SURFACES for height in HELD_HEIGHTS]
            fits = {surface: estimate(rows, result['source'], surface) for surface in SURFACES}
            result['held'][name] = dict(probes=rows, parameters=fits,
                                       applied={k: getattr(profile(name), f'{k}_damping') for k in SURFACES})
            print(name, fits, flush=True)
            _write(output, result)
        finally:
            renderer.close()
            arena.close()


def play(feeds, output, checkpoint, condition):
    """Apply the active optical survey to the nominal planner in paired live points."""
    from .tt29_action_memory import RecallPlanner
    from .tt29_endurance import audit
    from .tt29_material import memory_from
    from .tt29_partner import FOREHAND_LANES
    from .tt29_partner_fast import FastPartnerArena, FastPracticePartner, NativeBallFlight
    from .tt29_registration import register
    from .tt29_vision import FlightPractice, OpticalFlight
    from .tt29_vision_twin import OpticalPlanner

    report = _read(output)
    example_bank = _read(checkpoint)
    prior = {r['shot']['id'] for r in example_bank['rows']}
    if any(shot_id in prior for _, shot_id in LIVE):
        raise ValueError('A scored opening occurred in the motor checkpoint')
    launches = measured_launches(feeds)
    cap = 2
    settings = dict(strike_ready='lateral', width=SIZE[0], height=SIZE[1], view_mode='fixed',
                    fast_search=True, clearance_guard=True, motor_envelope=True, velocity_guard=True,
                    rate_scale=.8, robust_contact=True, max_steps=240)
    report['live'] = dict(dataset_sha256=_hash(feeds), checkpoint_sha256=_hash(checkpoint),
                          cap=cap, rows=[])
    for name, shot_id in LIVE:
        if condition is not None and name != condition:
            continue
        fits = report['held'][name]['parameters']
        strong_separation = any(fits[surface] is not None and report['held']['stock']['parameters'][surface] is not None
                                and abs(fits[surface]['effective_damping']
                                        - report['held']['stock']['parameters'][surface]['effective_damping']) > 3.
                                for surface in SURFACES)
        if not strong_separation:
            print(name, 'active probes did not separate at least one response from stock', flush=True)
            continue
        active = FastPartnerArena(**settings, targets=FOREHAND_LANES, guard_table=True)
        nominal = Rig.from_arena(active)
        apply(active, profile(name))
        active.partner = FastPracticePartner(active, targets=FOREHAND_LANES, guard_table=True)
        render = CachedStereoRenderer(active)
        try:
            frames = render.capture()
            cameras = Rig({eye: register(frames[eye], nominal.fixed[eye]).camera for eye in STEREO},
                          nominal.wrist)
            for mode in ('nominal', 'measured'):
                twin = RallyArena(**settings)
                try:
                    applied = {}
                    if mode == 'measured':
                        for surface, geom in SURFACES.items():
                            stock = report['held']['stock']['parameters'][surface]
                            if fits[surface] is not None and stock is not None:
                                shift = fits[surface]['effective_damping'] - stock['effective_damping']
                                contact = twin.model.geom(geom)
                                contact.solref[1] -= shift
                                applied[surface] = -float(contact.solref[1])
                    active.launch(launches[shot_id])
                    twin.reset(seed=shot_id, server=-1, render=False)
                    cameras.begin()
                    tracker = OpticalFlight(cameras, FlightPractice())
                    policy = OpticalPlanner(twin, seed=290110)
                    policy.nominal = RecallPlanner(twin, memory_from(example_bank), seed=290110)
                    policy.nominal.flight = NativeBallFlight(twin)
                    prev = active.joints.copy()
                    target = prev.copy()
                    lo, hi = twin._joint_low[policy.actuators], twin._joint_high[policy.actuators]
                    commands = []
                    far = np.zeros(5)
                    for tick in range(active.max_steps):
                        q = active.joints.copy()
                        tracker.observe(tick * active.control_dt, render.capture(), q)
                        action = policy.command(tick, tracker.latest, q, (q - prev) / active.control_dt, target)
                        target = np.clip(target + active.control_dt * policy.nominal.rates * action, lo, hi)
                        prev = q
                        _, _, done, truncated, _ = active.exchange(action, far)
                        commands.append(action.tolist())
                        if done or truncated or active.legal_returns + active.rival_returns >= cap + 1:
                            break
                    score = audit(active.events)
                    print(name, shot_id, mode, score['returns'], score['learner'], flush=True)
                    report['live']['rows'].append(dict(condition=name, id=shot_id, mode=mode,
                                                      applied_damping=applied, score=score,
                                                      actions=commands, events=active.events.copy()))
                    _write(output, report)
                finally:
                    twin.close()
        finally:
            render.close()
            active.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--feeds', type=Path)
    parser.add_argument('--checkpoint', type=Path, default=MOTOR)
    parser.add_argument('--play-only', action='store_true')
    parser.add_argument('--condition', choices=tuple(name for name, _ in LIVE))
    args = parser.parse_args()
    if not args.play_only:
        collect(args.output)
    if args.feeds is not None:
        play(args.feeds, args.output, args.checkpoint, args.condition)
