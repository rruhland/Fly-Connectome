"""Measure the real wrist field of view and test an encoder-only opening pan."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _write
from .tt29_arm_clearance import ArmClearance
from .tt29_rally_physics import RallyArena, measured_launches
from .tt29_vision import OpticalFlight, Rig, ball_pixels
from .tt29_vision_twin import OpticalPlanner


POSES = {'edge-ready': False, 'lateral-ready': 'lateral', 'lateral-pan': 'lateral', 'side-on': 'side-on'}
OUT = Path('artifacts/tt29/tt29_vision_gaze.json.gz')
UPCOURT_REFERENCE = np.array([.10, .65, 1.24])
PAN_GOAL = 1.10
PAN_TICKS = 11


def observe_round(arena, shadow_model, rig, twin, name, launch):
    arena.launch(launch)
    twin.reset(seed=launch['id'], server=-1, render=False)
    arena.sample_clearance(True)
    arena._cad_reader = (ArmClearance(shadow_model, tolerance=0.), mujoco.MjData(shadow_model))
    arena.streams = ('wrist',)
    eye = OpticalFlight(rig)
    pilot = OpticalPlanner(twin, seed=290110)
    q = arena.joints.copy()
    previous = target = q.copy()
    limits = twin._joint_low[pilot.actuators], twin._joint_high[pilot.actuators]
    initial = rig.wrist.at(q)
    fovy = rig.wrist.lens.fovy
    fovx = np.rad2deg(2 * np.arctan(arena.width / (2 * initial.focal)))
    frame_log = []
    actions = []
    for step in range(arena.max_steps):
        frame = arena.render_cameras()['wrist']
        timestamp = step * arena.control_dt
        q = arena.joints.copy()
        flight = eye.observe(timestamp, {'wrist': frame}, q)
        if name == 'lateral-pan' and step < PAN_TICKS:
            action = np.zeros(arena.action_size)
            action[0] = np.clip((PAN_GOAL - target[0]) / (arena.control_dt * pilot.nominal.rates[0]), -.85, .85)
        else:
            action = pilot.command(step, flight, q, (q - previous) / arena.control_dt, target)
        target = np.clip(target + action * arena.control_dt * pilot.nominal.rates, *limits)
        previous = q
        # Coverage and tracking error are evaluation-side checks only.
        actual = arena.ball.copy()
        projected = rig.wrist.at(q).project(actual)
        seen = ball_pixels(frame)
        viewport = (projected[2] > .025 and 0. <= projected[0] < arena.width and
                    0. <= projected[1] < arena.height)
        correct = bool(seen is not None and viewport and np.linalg.norm(seen.center - projected[:2]) < 3.)
        estimate = eye.estimate(timestamp)
        error = None if estimate is None else float(np.linalg.norm(estimate[0] - actual))
        frame_log.append(dict(tick=step, in_viewport=bool(viewport), rgb_ball=correct, position_error_m=error,
                              q0=float(q[0]), receiving=bool(arena.velocity[1] < -.7 and -.4 > actual[1] > -1.3)))
        actions.append(action.tolist())
        _, _, done, cap, _ = arena.exchange(action, np.zeros(5))
        if done or cap or arena.legal_returns:
            break
    row = dict(id=launch['id'], mode=name, lens=dict(horizontal_fov_deg=float(fovx), vertical_fov_deg=fovy,
                                                    initial_translation=initial.eye.tolist(),
                                                    initial_forward=initial.rotation[2].tolist(),
                                                    upcourt_reference_pixel=initial.project(UPCOURT_REFERENCE).tolist()),
               image_reads=sum(r['rgb_ball'] for r in frame_log),
               first_12_image_reads=sum(r['rgb_ball'] for r in frame_log[:12]),
               below_10cm=sum(r['position_error_m'] is not None and r['position_error_m'] < .10 for r in frame_log),
               track_near=[r['position_error_m'] for r in frame_log if r['receiving'] and r['position_error_m'] is not None],
               clearance=arena.clearance_result(), result=arena.point_result(), plans=pilot.nominal.decisions,
               trace=frame_log, actions=actions, events=arena.events.copy())
    arena.sample_clearance(False)
    return row


def run(feeds, output, ids, names):
    shots = measured_launches(feeds)
    result = dict(schema='tt29-moving-wrist-field-1', measured_source_sha256=_hash(feeds),
                  width=480, height=360, gaze=dict(axis='shoulder pan', target_rad=PAN_GOAL, ticks=PAN_TICKS),
                  upcourt_reference=UPCOURT_REFERENCE.tolist(), rows=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    for name in names:
        settings = dict(strike_ready=POSES[name], width=480, height=360, max_steps=110,
                        clearance_guard=True, fast_search=True)
        arena, cad, twin = (RallyArena(**settings), RallyArena(**settings, detail='cad'), RallyArena(**settings))
        try:
            rig = Rig.from_arena(arena)
            for sid in ids:
                row = observe_round(arena, cad.model, rig, twin, name, shots[sid])
                result['rows'].append(row)
                _write(output, result)
                print(name, sid, 'reads', row['first_12_image_reads'], row['image_reads'],
                      'error med', round(float(np.median(row['track_near'])), 3) if row['track_near'] else None,
                      'legal', row['result']['legal_returns'], 'clearance', row['clearance'], flush=True)
        finally:
            arena.close()
            cad.close()
            twin.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('feeds', type=Path)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--ids', type=int, nargs='+', default=(7345, 7271))
    parser.add_argument('--modes', nargs='+', choices=tuple(POSES), default=tuple(POSES))
    flags = parser.parse_args()
    run(flags.feeds, flags.output, flags.ids, flags.modes)
