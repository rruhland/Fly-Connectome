"""Verify an archived wrist-only point against the native referee and the full arm CAD."""

import argparse
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_arm_clearance import ArmClearance
from .tt29_endurance import audit
from .tt29_partner import LANES
from .tt29_rally_physics import measured_launches
from .tt29_vision_replay import clearance_final, frame, start_encoder, start_renderer
from .tt29_wrist_only import BRACKET, OPPONENT, OUT, WristPartner


def run(feeds, source, sid, receipt, movie):
    archive = _read(source)
    row = max((r for r in archive['outputs'] if r['shot']['id'] == sid), key=lambda r: r['summary']['returns'])
    if measured_launches(feeds)[sid] != row['shot'] or _hash(feeds) != archive['source_sha256']:
        raise ValueError('Recorded wrist point has an unexpected launch source')
    w, h = archive['camera_size']
    config = archive.get('configuration', {})
    mode = row['mode']
    ready = False if mode == 'edge' else 'lateral'
    settings = dict(board=BRACKET if mode in ('offset', 'offset-scan') else None,
                    strike_ready=ready, width=w, height=h, max_steps=1200,
                    clearance_guard=True, fast_search=True, targets=OPPONENT if ready else LANES,
                    guard_table=True, clearance_margin=config.get('clearance_margin', 0.),
                    robust_contact=config.get('robust_contact', False))
    arena = WristPartner(**settings)
    cad = WristPartner(**settings, detail='cad')
    encoder = renderer = None
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
        print(sid, score['returns'], 'exact events', len(arena.events), clearance, flush=True)
    finally:
        if encoder is not None and encoder.poll() is None:
            encoder.terminate()
            encoder.wait()
        if renderer is not None:
            renderer.close()
        arena.close()
        cad.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('feeds', type=Path)
    parser.add_argument('--id', type=int, default=7271)
    parser.add_argument('--source', type=Path, default=OUT)
    parser.add_argument('--receipt', type=Path, default=OUT.with_name('tt29_wrist_only_replay.json'))
    parser.add_argument('--movie', type=Path)
    a = parser.parse_args()
    if a.movie is None:
        run(a.feeds, a.source, a.id, a.receipt, None)
    else:
        with TemporaryDirectory(prefix='tt29-wrist-only-', dir=a.movie.parent) as temporary:
            path = Path(temporary) / 'wrist.mp4'
            run(a.feeds, a.source, a.id, a.receipt, path)
            path.replace(a.movie)
