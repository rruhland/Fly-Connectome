"""Play regulation points from one native wrist camera on a compact alternate bracket."""

import argparse
from pathlib import Path
from time import perf_counter
import xml.etree.ElementTree as ET

import numpy as np

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_so101 import CAMERA_BODY, GRIP_BODY, WRIST_CAMERA
from .tt29_action_memory import Experience, RecallPlanner, refine_after_bounce
from .tt29_arm_clearance import ArmClearance
from .tt29_endurance import audit
from .tt29_partner import FOREHAND_LANES, LANES, PartnerArena
from .tt29_rally_physics import RallyArena, measured_launches
from .tt29_vision import FlightPractice, OpticalFlight, Rig
from .tt29_vision_replay import clearance_final
from .tt29_vision_twin import OpticalPlanner


OUT = Path('artifacts/tt29/tt29_wrist_only.json.gz')
# Surveyed gripper rotation at the lateral ready configuration.
LATERAL_BASIS = np.array([
    [-.11125296, .12456999, -.98595390],
    [-.51435693, -.85611000, -.05012596],
    [-.85032919, .50155556, .15931821],
])
APERTURE = (-.021, .050, -.058)
BRACKET = (38., 102.)
OPPONENT = np.array(FOREHAND_LANES)
SCAN_POS = 1.03


def vector(x):
    return ' '.join(f'{value:.9g}' for value in x)


def mount(document, offset, aim, fovy):
    """Change the compact camera body with its housing and a two-standoff bracket."""
    gripper = document.find(f".//body[@name='{GRIP_BODY}']")
    dock = gripper.find("body[@name='learner_wrist_camera_mount']")
    camera = dock.find(f"body[@name='{CAMERA_BODY}']")
    turn = np.deg2rad(aim)
    optical = np.array([np.sin(turn), np.cos(turn), -.058])
    optical /= np.linalg.norm(optical)
    across = np.cross(optical, (0., 0., 1.))
    across /= np.linalg.norm(across)
    basis = LATERAL_BASIS.T @ np.column_stack((across, np.cross(across, optical), -optical))
    camera.set('pos', vector(offset))
    camera.set('xyaxes', vector(np.r_[basis[:, 0], basis[:, 1]]))
    camera.find(f"camera[@name='{WRIST_CAMERA}']").set('fovy', str(fovy))
    old = dock.find("geom[@name='tt29_camera_bracket']")
    dock.remove(old)
    for side in (-1, 1):
        foot = np.array([-0.010 + side * .007, .024, -.039])
        back = np.asarray(offset) + basis @ [.011 * side, -.010, .015]
        ET.SubElement(dock, 'geom', name=f'tt29_camera_standoff_{side}', type='capsule',
                      fromto=vector(np.r_[foot, back]), size='.0036', rgba='.84 .86 .84 1',
                      contype='0', conaffinity='1', mass='0')
    return document


class CompactBoard:
    def __init__(self, *, board=None, **settings):
        self.board = board
        if board is not None:
            settings['wrist_fovy'] = board[1]
        super().__init__(**settings)

    def _model_document(self):
        document = super()._model_document()
        return document if self.board is None else mount(document, APERTURE, *self.board)


class WristReceiver(CompactBoard, RallyArena):
    pass


class WristPartner(CompactBoard, PartnerArena):
    pass


class SingleView(OpticalFlight):
    """Require a supported monocular fit for a fresh stroke, including range uncertainty."""

    def __init__(self, rig, practice, *, confidence=.112, window=.155, min_frames=4):
        super().__init__(rig, practice, learn=True, window=window)
        if not 0. < confidence <= .40 or min_frames < 3:
            raise ValueError('expected a positive range uncertainty limit and at least three captures')
        self.confidence = confidence
        self.min_frames = min_frames
        self.usable = None
        self.last_read = -np.inf

    def capture(self, time, image, joints):
        current = self.observe(time, {'wrist': image}, joints)
        if self.last_frame_views:
            self.last_read = time
        if current is not None and current.time == time and current.frames >= self.min_frames:
            p, v = current.position, current.velocity
            if (current.depth_sigma < self.confidence and current.rms < 1.5 and
                    .60 < p[2] < 1.88 and abs(v[0]) < 2.3 and abs(v[1]) < 9.):
                self.usable = current
        if self.usable is None or time - self.usable.time > .095:
            return None
        return self.usable


def scan_toward(pupil, goal, target):
    out = np.zeros(6)
    out[0] = np.clip((goal - target[0]) / (pupil.nominal.rates[0] * pupil.twin.control_dt), -.55, .55)
    return out


def capture_player(arena, twin, model, shot, *, course, memory, confidence, scan, seed, max_rally,
                   min_frames, refine, camera=None):
    arena.launch(shot)
    arena.streams = ('wrist',)
    arena.partner.reset(seed=seed)
    twin.reset(seed=seed, server=-1, render=False)
    arena.sample_clearance(True)
    guard = ArmClearance(model, tolerance=0.)
    shadow = mujoco.MjData(model)
    arena._cad_reader = guard, shadow
    rig = Rig({}, Rig.from_arena(twin).wrist if camera is None else camera)
    eye = SingleView(rig, course, confidence=confidence, min_frames=min_frames)
    pupil = OpticalPlanner(twin, seed=290110)
    planner = RecallPlanner(twin, memory, seed=290110)
    pupil.nominal = planner
    target = pupil.home.copy()
    last_angle = arena.joints.copy()
    lower, upper = twin._joint_low[pupil.actuators], twin._joint_high[pupil.actuators]
    moves = []
    optical_track = []
    frames = misses = scans = estimates = 0
    start = perf_counter()
    for step in range(arena.max_steps):
        now = step * arena.control_dt
        angles = arena.joints.copy()
        rgb = arena.render_wrist()
        flight = eye.capture(now, rgb, angles)
        rates = (angles - last_angle) / arena.control_dt
        if refine:
            refine_after_bounce(pupil, step, flight, angles, rates, target)
        command = pupil.command(step, flight, angles, rates, target)
        if scan and pupil.tape is None and flight is None and now - eye.last_read > .07 and step % 25 < 13:
            command = scan_toward(pupil, SCAN_POS, target)
            scans += 1
        target = np.clip(target + command * planner.rates * arena.control_dt, lower, upper)
        last_angle = angles
        frames += bool(eye.last_frame_views)
        misses += int(not eye.last_frame_views)
        estimates += int(flight is not None)
        if flight is not None and flight.time == now:
            p, v = flight.advance(now)
            optical_track.append([step, *p.tolist(), *v.tolist(), flight.depth_sigma])
        _, _, done, cap, info = arena.exchange(command, np.zeros(5))
        moves.append(command.tolist())
        if any(e['kind'] == 'legal_return' and e['side'] == 1 for e in info['events']):
            planner.remember()
        if arena.legal_returns and arena.legal_returns % 5 == 0 and any(
            e['kind'] == 'legal_return' and e['side'] == 1 for e in info['events']
        ):
            print('live', shot['id'], arena.legal_returns, arena.rival_returns - 1, flush=True)
        if done or cap or arena.legal_returns + arena.rival_returns >= max_rally + 1:
            break
    outcome = audit(arena.events)
    clearance = clearance_final(arena, guard, shadow)
    arena.sample_clearance(False)
    return dict(shot=shot, opponent_seed=seed, summary={k: v for k, v in outcome.items() if k != 'shots'},
                shots=outcome['shots'], result=arena.point_result(), native_wrist_frames=len(moves),
                detections=frames, missing_detections=misses, metric_estimates=estimates, gaze_commands=scans,
                optical_track=optical_track, events=arena.events.copy(), actions=moves,
                decisions=planner.decisions, memory_retrievals=planner.retrieved, accepted_plans=planner.fallback,
                CAD=clearance, motor=arena.motor_audit(), seconds=perf_counter() - start)


def load_memory(path):
    memory = Experience()
    if path is not None:
        for e in _read(path)['checkpoint']:
            memory.add(np.array(e['flight']), np.array(e['targets']), e['time_to_contact'])
    return memory


def run(feeds, output, ids, modes, *, width, height, warm, confidence, max_rally, seed_offset,
        min_frames, refine, margin, robust):
    shots = measured_launches(feeds)
    result = dict(schema='tt29-wrist-only-rally-1', source_sha256=_hash(feeds), camera_size=[width, height],
                  memory_source_sha256=_hash(warm) if warm else None, opens=list(ids),
                  streams=['wrist'], inputs=['native RGB', 'capture timestamp', 'measured joint angles',
                                               'issued near joint targets', 'legal near return reward'],
                  gaze_target_rad=SCAN_POS, configuration=dict(min_frames=min_frames, confidence=confidence,
                                                             refine=refine, clearance_margin=margin,
                                                             robust_contact=robust), outputs=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    for mode in modes:
        ready = False if mode == 'edge' else 'lateral'
        board = BRACKET if mode in ('offset', 'offset-scan') else None
        optics = dict(board=board, strike_ready=ready, width=width, height=height, max_steps=1200,
                      clearance_guard=True, fast_search=True, clearance_margin=margin, robust_contact=robust)
        partner = dict(targets=OPPONENT if ready else LANES, guard_table=True)
        arena = WristPartner(**optics, **partner)
        cad = WristPartner(**optics, **partner, detail='cad')
        twin = WristReceiver(**optics)
        memory = load_memory(warm) if ready else Experience()
        course = FlightPractice()
        try:
            for sid in ids:
                row = capture_player(arena, twin, cad.model, shots[sid], course=course, memory=memory,
                                     confidence=confidence, scan=mode.endswith('scan'),
                                     seed=sid + seed_offset, max_rally=max_rally, min_frames=min_frames, refine=refine)
                row.update(mode=mode, mount_translation=APERTURE if board else None,
                           mount_aim_deg=board[0] if board else None, mount_fov_deg=board[1] if board else arena.wrist_fovy)
                result['outputs'].append(row)
                _write(output, result)
                print(mode, sid, row['summary']['returns'], row['result']['reason'], row['detections'],
                      row['metric_estimates'], row['accepted_plans'], row['memory_retrievals'], row['CAD'], flush=True)
        finally:
            twin.close()
            cad.close()
            arena.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('feeds', type=Path)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--ids', nargs='+', type=int, default=(7345, 7271))
    parser.add_argument('--modes', nargs='+', choices=('offset', 'offset-scan', 'stock-scan', 'edge', 'stock'),
                        default=('offset',))
    parser.add_argument('--warm', type=Path)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--confidence', type=float, default=.112)
    parser.add_argument('--min-frames', type=int, default=4)
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--clearance-margin', type=float, default=0.)
    parser.add_argument('--robust-contact', action='store_true')
    parser.add_argument('--seed-offset', type=int, default=0)
    parser.add_argument('--target', type=int, default=20)
    a = parser.parse_args()
    run(a.feeds, a.output, a.ids, a.modes, width=a.width, height=a.height, warm=a.warm,
        confidence=a.confidence, seed_offset=a.seed_offset, max_rally=a.target,
        min_frames=a.min_frames, refine=a.refine, margin=a.clearance_margin, robust=a.robust_contact)
