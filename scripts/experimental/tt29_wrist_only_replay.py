"""Verify an archived wrist-only point against the native referee and the full arm CAD."""

import argparse
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_action_memory import context
from .tt29_arm_clearance import ArmClearance
from .tt29_endurance import audit
from .tt29_partner import LANES
from .tt29_partner_fast import NativeBallFlight
from .tt29_rally_physics import measured_launches
from .tt29_vision import FlightPractice, Rig
from .tt29_vision_replay import clearance_final, frame, start_encoder, start_renderer
from .tt29_wrist_only import BRACKET, OPPONENT, OUT, FastWristPartner, SingleView, WristPartner, WristReceiver


def carry_examples(arena, trace, score, base, source, output):
    """Use camera reconstructions and issued targets from physically rewarded strokes."""
    labels = [shot for shot in score['shots'] if shot['side'] == 1]
    original = _read(base)['checkpoint'] if base is not None else []
    examples = []
    recorded = []
    learner_model = NativeBallFlight(arena)
    dt = arena.control_dt
    targets = [row[2].tolist() for row in trace]
    for label in labels:
        hit = label['hit']
        choices = [row for row in trace if row[1] is not None and .075 < hit - row[0] * dt < .29 and
                   row[1].velocity[1] < -1.8 and row[1].depth_sigma < .13]
        choices.sort(key=lambda row: abs(hit - row[0] * dt - .165) + .7 * row[1].depth_sigma)
        for step, flight, _ in choices[:8]:
            position, velocity = flight.advance(step * dt)
            path = learner_model.path(position, velocity, np.zeros(3), 1, flight.bounced)
            packet = context(path, -1.29)
            if packet is None:
                continue
            key, eta = packet
            if not .038 < eta < .34:
                continue
            future = targets[step:step + round((eta + .65) / dt)]
            examples.append(dict(flight=key.tolist(), targets=future, time_to_contact=float(eta)))
            recorded.append(dict(tick=step, legal_hit=hit, optical_time=flight.time, depth_sigma=flight.depth_sigma))
            break
    if not examples:
        raise ValueError('No rewarded stroke had a supported wrist approach for motor memory')
    _write(output, dict(schema='tt29-wrist-rewarded-stroke-memory-1', source_sha256=_hash(source),
                        seed_memory_sha256=_hash(base) if base else None, base_examples=len(original),
                        admitted=recorded, added_examples=len(examples), checkpoint=original + examples))
    print('recalled', len(examples), 'wrist-only examples from', len(labels), 'rewarded strokes', flush=True)


def run(feeds, source, sid, receipt, movie, *, checkpoint=None, base=None):
    archive = _read(source)
    row = max((r for r in archive['outputs'] if r['shot']['id'] == sid), key=lambda r: r['summary']['returns'])
    if measured_launches(feeds)[sid] != row['shot'] or _hash(feeds) != archive['source_sha256']:
        raise ValueError('Recorded wrist point has an unexpected launch source')
    w, h = archive['camera_size']
    config = archive.get('configuration', {})
    mode = row['mode']
    ready = False if mode == 'edge' else 'lateral'
    settings = dict(board=BRACKET if mode in ('offset', 'offset-scan') else None,
                    strike_ready=ready, width=w, height=h, max_steps=config.get('max_steps', 1200),
                    clearance_guard=True, fast_search=True, targets=OPPONENT if ready else LANES,
                    guard_table=True, clearance_margin=config.get('clearance_margin', 0.),
                    robust_contact=config.get('robust_contact', False))
    world = FastWristPartner if config.get('fast_partner', False) else WristPartner
    arena = world(**settings)
    cad = world(**settings, detail='cad')
    encoder = renderer = nominal = observer = None
    rhythm = []
    if checkpoint is not None:
        nominal = WristReceiver(board=settings['board'], strike_ready=ready, width=w, height=h)
        observer = SingleView(Rig({}, Rig.from_arena(nominal).wrist), FlightPractice(),
                              confidence=config.get('confidence', .112), min_frames=config.get('min_frames', 4))
    measured = arena._arm_ids[1][2]
    target = np.asarray(arena._arm_home(1)[0])
    stride = arena.control_dt * arena._joint_increments[measured] / arena.model.opt.timestep
    bounds = arena._joint_low[measured], arena._joint_high[measured]
    digest = sha256()
    count = n_video = 0
    try:
        arena.launch(row['shot'])
        arena.partner.reset(seed=row['opponent_seed'])
        arena.streams = ('wrist',)
        arena.sample_clearance(True)
        guard, shadow = ArmClearance(cad.model, tolerance=0.), mujoco.MjData(cad.model)
        arena._cad_reader = guard, shadow
        if movie is not None:
            renderer = start_renderer(arena)
            encoder = start_encoder(movie)
        for step, command in enumerate(row['actions']):
            if encoder is not None and step % 2 == 0:
                encoder.stdin.write(frame(arena, renderer, step, count, ('wrist',), sid).tobytes())
                n_video += 1
            if observer is not None:
                flight = observer.capture(step * arena.control_dt, arena.render_wrist(), arena.joints.copy())
                target = np.clip(target + command * stride, *bounds)
                rhythm.append((step, flight, target.copy()))
            _, _, done, cap, _ = arena.exchange(command, np.zeros(5))
            digest.update(arena.data.qpos.astype('<f8').tobytes())
            digest.update(arena.data.qvel.astype('<f8').tobytes())
            count = max(0, arena.legal_returns + arena.rival_returns - 1)
            if done or cap:
                break
        score = audit(arena.events)
        if (digest_json(arena.events) != digest_json(row['events']) or arena.steps != len(row['actions']) or
                {k: v for k, v in score.items() if k != 'shots'} != row['summary']):
            raise ValueError('Wrist-only action replay changed the archived physical result')
        clearance = clearance_final(arena, guard, shadow)
        if clearance != row['CAD'] or clearance['samples'] != round(arena.data.time / arena.model.opt.timestep) + 1:
            raise ValueError('The wrist-only full-CAD audit changed or skipped a physics state')
        if encoder is not None:
            encoder.stdin.close()
            if encoder.wait() != 0:
                raise RuntimeError('FFmpeg did not encode the wrist-only replay')
        details = dict(schema='tt29-wrist-only-replay-1', source_sha256=_hash(source), shot_id=sid,
                       source_view='wrist', mode=mode, timings=dict(control_dt=arena.control_dt,
                       physics_dt=arena.model.opt.timestep, physics_duration=arena.data.time),
                       state_sha256=digest.hexdigest(), events_sha256=digest_json(arena.events),
                       event_count=len(arena.events), exact_replay=True, clearance=clearance,
                       score=score, motor=arena.motor_audit(), terminal=arena.point_result(),
                       video=None if movie is None else dict(sha256=_hash(movie), frames=n_video, fps=30))
        _write(receipt, details)
        if checkpoint is not None:
            carry_examples(arena, rhythm, score, base, source, checkpoint)
        print(sid, score['returns'], 'exact events', len(arena.events), clearance, flush=True)
    finally:
        if encoder is not None and encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
        if renderer is not None:
            renderer.close()
        if nominal is not None:
            nominal.close()
        arena.close()
        cad.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('feeds', type=Path)
    parser.add_argument('--id', type=int, default=7271)
    parser.add_argument('--source', type=Path, default=OUT)
    parser.add_argument('--receipt', type=Path, default=OUT.with_name('tt29_wrist_only_replay.json'))
    parser.add_argument('--movie', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--base-memory', type=Path)
    a = parser.parse_args()
    a.receipt.parent.mkdir(parents=True, exist_ok=True)
    if a.movie is None:
        run(a.feeds, a.source, a.id, a.receipt, None, checkpoint=a.checkpoint, base=a.base_memory)
    else:
        a.movie.parent.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(prefix='tt29-wrist-only-', dir=a.movie.parent) as temporary:
            path = Path(temporary) / 'wrist.mp4'
            run(a.feeds, a.source, a.id, a.receipt, path, checkpoint=a.checkpoint, base=a.base_memory)
            path.replace(a.movie)
