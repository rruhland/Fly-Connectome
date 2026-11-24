"""Play camera-driven regulation points and separately audit the flight estimates."""

import argparse
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _write
from .tt27_world_camera import CameraMounts
from .tt29_rally_physics import measured_launches
from .tt29_vision import Flight, FlightPractice, OpticalFlight, Rig
from .tt29_vision_control import EXPLORATION, OpticalStroke, OutcomeMemory
from .tt25_reference_receiver import ReferenceReceiver


OUT = Path('artifacts/tt29')
DEVELOPMENT_SHOTS = (4197, 3728, 4230, 7637, 15489, 8230, 14562, 10318)
HELD_SHOTS = (15400, 7471, 2950, 12670, 8388, 9447, 6866, 7165, 10268, 3248, 8741, 6341)


@dataclass(frozen=True)
class Condition:
    eyes: tuple[str, ...]
    lag: int = 0
    mask: bool = False
    turn: bool = False
    register: bool = False
    brake: float = 1.
    height: float = 1.


CONDITIONS = {
    'stereo': Condition(('left', 'right')),
    'outside': Condition(('left',)),
    'side': Condition(('side',)),
    'wrist': Condition(('wrist',)),
    'combined': Condition(('wrist', 'left', 'right')),
    'delay': Condition(('left', 'right'), lag=3),
    'drop': Condition(('left', 'right'), mask=True),
    'rotated': Condition(('left', 'right'), turn=True),
    'registered': Condition(('left', 'right'), turn=True, register=True),
    'soft': Condition(('left', 'right'), brake=1.37),
    'strong_gravity': Condition(('left', 'right'), height=1.08),
    'blind': Condition(()),
    'state_diagnostic': Condition(()),
}


def play(arena, rig, practice, memory, shot, condition, *, fixed_action=None, learn=False, exact=False):
    arena.launch(shot)
    rig.begin()
    tracker = OpticalFlight(rig, practice, learn=learn)
    player = OpticalStroke(arena, memory=memory, explore=learn, fixed_action=fixed_action)
    target = player.home.copy()
    captures = deque()
    measurements = []
    actions, reads, estimate_ticks = [], Counter(), []
    render_seconds = control_seconds = 0.
    for tick in range(arena.max_steps):
        now = tick * arena.control_dt
        q = arena.joints.copy()
        start = perf_counter()
        eyes = condition.eyes
        if condition.mask:
            if 12 <= tick < 24:
                eyes = ('left',)
            elif 25 <= tick < 29:
                eyes = ()
        arena.streams = eyes
        frames = arena.render_cameras() if eyes else {}
        render_seconds += perf_counter() - start
        captures.append((now, frames, q))
        start = perf_counter()
        if len(captures) > condition.lag:
            t, images, enc = captures.popleft()
            tracker.observe(t, images, enc)
            reads.update(tracker.last_frame_views)
        flight = tracker.latest
        # Scoring and the named diagnostic alone may inspect hidden state.
        if exact:
            p, v = arena.ball.copy(), arena.velocity.copy()
            flight = Flight(p, v, now, .133 * np.linalg.norm(v), 0., 0., 0, arena._received_bounce)
        command = player.command(now, flight, q, target, practice.rebound)
        target = np.clip(target + arena.control_dt * player.rates * command,
                         np.r_[player.ik.low, -1.75], np.r_[player.ik.high, 1.75])
        control_seconds += perf_counter() - start
        estimate = tracker.estimate(now)
        if estimate is not None:
            p, v = estimate
            actual = arena.ball.copy()
            if arena.velocity[1] < -.7 and -.45 > actual[1] > -1.26:
                measurements.append([tick, float(np.linalg.norm(p - actual)),
                                     float(np.linalg.norm(v - arena.velocity)), flight.bounced])
            estimate_ticks.append(tick)
        actions.append(command.tolist())
        _, _, done, cap, _ = arena.step(command, render=False)
        if done or cap or arena.legal_returns:
            break
    if learn:
        player.outcome(arena.legal_returns > 0)
    return dict(id=shot['id'], result=arena.point_result(), events=arena.events.copy(), actions=actions,
                plans=player.history, measurements=measurements, reads=dict(reads), estimate_ticks=estimate_ticks,
                wall_render_s=render_seconds, wall_optics_control_s=control_seconds,
                registrations=rig.registrations.copy())


def start(mode, width, height):
    arena = ReferenceReceiver(fixture='table-edge', view_mode='wrist+fixed', width=width, height=height,
                              frameskip=4, max_steps=110)
    # Supply an original public image to the camera's registration, before any held shot.
    arena.streams = ('left', 'right')
    reference = arena.render_cameras() if mode.register else None
    rig = Rig.from_arena(arena, reference=reference)
    if mode.turn:
        CameraMounts(arena).move(((2.8, -2.2, 2.3), (-2.7, 2.1, -2.9)))
    for name in arena.table_panels:
        arena.model.geom(name).solref[1] *= mode.brake
    arena.model.opt.gravity[2] *= mode.height
    return arena, rig


def summarize(rows):
    measurements = np.asarray([m for row in rows for m in row['measurements']])
    return dict(points=len(rows), contacts=sum(r['result']['contacts'] for r in rows),
                legal=sum(r['result']['legal_returns'] for r in rows),
                faults=dict(Counter(r['result']['reason'] for r in rows)),
                hit_speed_m_s=[float(np.linalg.norm(event['incoming'][:2])) for row in rows for event in row['events']
                               if event['kind'] == 'contact' and event['side'] == 1],
                plans=sum(len(r['plans']) for r in rows),
                tracking_samples=len(measurements),
                median_position_cm=float(np.median(measurements[:, 1]) * 100.) if len(measurements) else None,
                median_velocity_m_s=float(np.median(measurements[:, 2])) if len(measurements) else None,
                camera_s_per_tick=sum(r['wall_render_s'] for r in rows) / sum(len(r['actions']) for r in rows),
                compute_s_per_tick=sum(r['wall_optics_control_s'] for r in rows) / sum(len(r['actions']) for r in rows))


def run(feeds, output, conditions, ids, *, width=480, height=360, passes=0, fixed_action=None):
    shots = measured_launches(feeds)
    output.parent.mkdir(parents=True, exist_ok=True)
    results = dict(schema='tt29-vision-regulation-1', source_sha256=_hash(feeds), dimensions=[width, height],
                   timing_hz=60, ids=list(ids), practice=[], assessment={})
    memory = OutcomeMemory(241170)
    practice = FlightPractice()
    if passes:
        arena, rig = start(CONDITIONS['combined'], width, height)
        try:
            for epoch in range(passes):
                rows = []
                for sid in DEVELOPMENT_SHOTS:
                    row = play(arena, rig, practice, memory, shots[sid], CONDITIONS['combined'], learn=True)
                    rows.append(row)
                results['practice'].append(dict(pass_num=epoch + 1, summary=summarize(rows), rows=rows))
                print('practice', epoch + 1, results['practice'][-1]['summary'], flush=True)
                _write(output, results)
        finally:
            arena.close()
    results['optical_calibration'] = dict(radius={k: practice.radius_factor(k) for k in practice.radius_samples},
                                          rebound=practice.rebound.tolist())
    for name in conditions:
        mode = CONDITIONS[name]
        arena, rig = start(mode, width, height)
        rows = []
        try:
            for sid in ids:
                row = play(arena, rig, practice, memory, shots[sid], mode,
                           fixed_action=fixed_action, exact=name == 'state_diagnostic')
                rows.append(row)
                e = row['result']
                print(name, sid, 'contact', e['contacts'], 'legal', e['legal_returns'], e['reason'],
                      'plans', len(row['plans']), 'reads', row['reads'], flush=True)
            results['assessment'][name] = dict(summary=summarize(rows), rows=rows)
            print(name, 'summary', results['assessment'][name]['summary'], flush=True)
            _write(output, results)
        finally:
            arena.close()
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('feeds', type=Path)
    p.add_argument('--output', type=Path, default=OUT / 'tt29_vision_assay.json.gz')
    p.add_argument('--modes', nargs='+', choices=tuple(CONDITIONS), default=('stereo', 'outside', 'wrist'))
    p.add_argument('--ids', nargs='+', type=int, default=HELD_SHOTS)
    p.add_argument('--width', type=int, default=480)
    p.add_argument('--height', type=int, default=360)
    p.add_argument('--passes', type=int, default=0)
    p.add_argument('--action', type=int, choices=range(len(EXPLORATION)))
    a = p.parse_args()
    fixed = EXPLORATION[a.action] if a.action is not None else None
    run(a.feeds, a.output, a.modes, a.ids, width=a.width, height=a.height, passes=a.passes, fixed_action=fixed)


if __name__ == '__main__':
    main()
