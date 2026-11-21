"""State-versus-timing ablation at six native, matched, legal rival replies."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_arena import mujoco
from .tt4_transfer_trial import ROOT, _hash, _read, _write
from .tt19_flight import optical_start
from .tt20_curriculum import OUTPUT as TT20, apparatus, configuration, optics
from .tt20_natural import NaturalViewTransport
from .tt21_body_state import ReplyRanker


SOURCE = TT20 / 'curriculum-frozen-2082473549.json.gz'
CASES = {890890078: 41, 32637606: 44, 531698253: 41, 416923263: 37, 1532332251: 48, 1634221833: 42}
WINDOWS = {'early': .15, 'late': -.24}
ZERO = np.zeros(6)


def _body(model, data, axes):
    mujoco.mj_kinematics(model, data)
    location = data.geom('learner_racket').xpos.copy()
    normal = data.geom('learner_racket').xmat.reshape(3, 3)[:, 2].copy()
    grip = model.joint('learner_paddle_free')
    dof = grip.dofadr[0]
    body = model.geom('learner_racket').bodyid[0]
    body_pos = data.xpos[body]
    angular = data.xmat[body].reshape(3, 3) @ data.qvel[dof + 3:dof + 6]
    velocity = data.qvel[dof:dof + 3] + np.cross(angular, location - body_pos)
    joints = model.actuator_trnid[axes, 0]
    posture = data.qpos[model.jnt_qposadr[joints]].copy()
    speeds = data.qvel[model.jnt_dofadr[joints]].copy()
    return dict(center=location, normal=normal, speed=velocity, joint=posture, joint_speed=speeds,
                targets=data.ctrl[axes].copy())


def _metrics(estimate, truth):
    return dict(blade_mm=float(1000 * np.linalg.norm(estimate['center'] - truth['center'])),
                blade_speed_mm_s=float(1000 * np.linalg.norm(estimate['speed'] - truth['speed'])),
                blade_normal_deg=float(np.rad2deg(np.arccos(np.clip(estimate['normal'] @ truth['normal'], -1., 1.)))),
                joint_rms_deg=float(np.rad2deg(np.mean((estimate['joint'] - truth['joint']) ** 2) ** .5)),
                joint_speed_rms_deg_s=float(np.rad2deg(np.mean((estimate['joint_speed'] - truth['joint_speed']) ** 2) ** .5)),
                joint_target_rms_deg=float(np.rad2deg(np.mean((estimate['targets'] - truth['targets']) ** 2) ** .5)))


def _neutral_body(sim):
    scratch = mujoco.MjData(sim.model)
    mujoco.mj_setState(sim.model, scratch, sim.base, sim.state_flags)
    scratch.ctrl[:] = sim.home
    return _body(sim.model, scratch, sim.twin.axes)


def readiness(arena, ranker, neutral, accumulated):
    """Read target encoders for evaluation only, outside the selection call."""
    truth = _body(arena.model, arena.data, arena._arm_ids[1][2])
    twin = _body(ranker.sim.model, ranker.sim.twin.data, ranker.sim.twin.axes)
    return dict(net_requested_rates=np.asarray(accumulated).tolist(),
                target_blade_speed_mm_s=float(1.e3 * np.linalg.norm(truth['speed'])),
                home=_metrics(neutral, truth), twin=_metrics(twin, truth))


def common_approach(arena, rig, ranker, neutral, row, cutoff):
    arena.reset(seed=row['seed'], server=row['server'], render=False)
    ranker.reset_body()
    tracker = rig.tracker()
    history, evidence, gate, decisions = [], [], {}, {}
    prefix = row['actions'][:cutoff]
    accumulated = np.sum(np.asarray(prefix), axis=0)
    for tick in range(arena.max_steps):
        if tick >= cutoff - 5:
            tracker.update(arena.render_cameras(), tick=tick)
            ball = tracker.ball
            position = None if ball is None or not ball.observed else list(ball.position)
            history.append((tick, position))
            evidence.append(dict(tick=tick, position=position, sources=list(tracker.sources.get('ball', ()))))
        if tick == cutoff:
            replayed = [e for e in arena.events if e['kind'] == 'contact' and e['side'] == -1 and e['step'] == cutoff]
            originals = [e for e in row['events'] if e['kind'] == 'contact' and e['side'] == -1 and e['step'] == cutoff]
            if not replayed or json.dumps(replayed) != json.dumps(originals):
                raise ValueError(f"shared rival contact for seed {row['seed']} changed")
            contact_state = readiness(arena, ranker, neutral, accumulated)
        if tick >= cutoff:
            estimate = optical_start(history, ranker.law.gravity)
            if estimate is not None:
                point, velocity = estimate
                for name, limit in WINDOWS.items():
                    if name in gate or not (-.81 < point[1] < limit and velocity[1] < -.031):
                        continue
                    home, home_work = ranker.rank_at(history.copy(), estimate)
                    twin, twin_work = ranker.rank_at(history.copy(), estimate, ranker.sim.twin.state())
                    gate[name] = dict(tick=tick, ball=point.tolist(), velocity_m_per_frame=velocity.tolist(),
                                      independent_rollouts=dict(home=home_work, twin=twin_work),
                                      predecision=readiness(arena, ranker, neutral, accumulated))
                    decisions[name] = {'home': home, 'twin': twin}
        action = prefix[tick] if tick < cutoff else ZERO
        _, _, done, capped, _ = arena.step(action, render=False)
        ranker.advance(action)
        if done or capped:
            break
    if not any(event['kind'] == 'legal_return' and event['side'] == -1 and event['step'] >= cutoff
               for event in arena.events):
        raise ValueError('the neutral continuation did not reproduce the actual legal rival reply')
    return dict(seed=row['seed'], server=row['server'], prefix=prefix, cutoff=cutoff, cutoff_state=contact_state,
                gate=gate, decisions=decisions, optical=evidence, observation_sha256=tracker.digest.hexdigest())


def live_suffix(arena, ranker, approach, timing, initialization):
    seed, server = approach['seed'], approach['server']
    arena.reset(seed=seed, server=server, render=False)
    gate = approach['gate'].get(timing)
    start = None if gate is None else gate['tick']
    picked = approach['decisions'].get(timing, {}).get(initialization)
    if picked is not None and not picked['legal'] and min(picked['robustness']) > 2.8 and gate['ball'][1] > -.51:
        picked = None
    tape, progress = [], np.zeros(5)
    previous = approach['prefix']
    for tick in range(arena.max_steps):
        if tick < len(previous):
            action = previous[tick]
        elif picked is not None and tick >= start:
            elapsed = tick - start
            rate = (ranker.tapes[picked['index'], elapsed].astype(float) if elapsed < picked['duration']
                    else -np.clip(progress, -1., 1.))
            progress += rate
            action = np.r_[rate, 0.].tolist()
        else:
            action = ZERO.tolist()
        tape.append(action)
        _, reward, done, capped, _ = arena.step(action, render=False)
        if done or capped:
            break
    own = [(e['kind'], e['step']) for e in arena.events if e.get('side') == 1 and e['step'] > approach['cutoff']]
    return dict(actions=tape, result=arena.point_result(), events=arena.events.copy(), reward=reward,
                planned_tick=start if picked is not None else None, applied_decision=picked,
                reply_contacts=sum(kind == 'contact' for kind, _ in own),
                legal_followups=sum(kind == 'legal_return' for kind, _ in own))


def replay(arena, cases):
    count = 0
    for row in cases:
        for treatment in row['trials'].values():
            for result in treatment.values():
                arena.reset(seed=row['seed'], server=row['server'], render=False)
                for action in result['actions']:
                    arena.step(action, render=False)
                if arena.point_result() != result['result'] or json.dumps(arena.events) != json.dumps(result['events']):
                    raise ValueError(f"physical tape changed for {row['seed']}")
                count += 1
    return count


def _stats(values):
    return None if not values else dict(n=len(values), median=float(np.median(values)), max=float(np.max(values)))


def summarize(cases):
    stats = {}
    for timing in ('cutoff', *WINDOWS):
        rows = ([row['cutoff_state'] for row in cases] if timing == 'cutoff'
                else [row['gate'][timing]['predecision'] for row in cases if timing in row['gate']])
        result = {'actual_blade_speed_mm_s': _stats([row['target_blade_speed_mm_s'] for row in rows])}
        for initial in ('home', 'twin'):
            result[initial] = {} if not rows else {key: _stats([row[initial][key] for row in rows]) for key in rows[0][initial]}
            if timing != 'cutoff':
                trials = [row['trials'][timing][initial] for row in cases]
                result[initial].update(planned=sum(row['planned_tick'] is not None for row in trials),
                                       contacts=sum(row['reply_contacts'] for row in trials),
                                       legal_returns=sum(row['legal_followups'] for row in trials),
                                       won=sum(row['reward'] > 0 for row in trials))
        stats[timing] = result
    return stats


def run(output):
    log = _read(SOURCE)
    rows = {row['seed']: row for row in log['points'] if row['seed'] in CASES}
    if set(rows) != set(CASES) or log['opponent'] != 'standard':
        raise ValueError('common native reply schedule changed')
    _, catalog, law, response, _ = configuration(TT20)
    ranker = ReplyRanker(catalog, law, response)
    nominal = _neutral_body(ranker.sim)
    arena = apparatus(False)
    arena.streams = ('left', 'right')
    rig = NaturalViewTransport('fixed', lens_fov=optics(arena), frame_seconds=arena.control_dt)
    cases = []
    clock = perf_counter()
    try:
        for seed, cutoff in CASES.items():
            trial = common_approach(arena, rig, ranker, nominal, rows[seed], cutoff)
            trial['trials'] = {timing: {state: live_suffix(arena, ranker, trial, timing, state)
                                      for state in ('home', 'twin')} for timing in WINDOWS}
            cases.append(trial)
            print(seed, {name: (entry['tick'], entry['independent_rollouts']) for name, entry in trial['gate'].items()},
                  {timing: [entry['legal_followups'] for entry in arms.values()] for timing, arms in trial['trials'].items()},
                  flush=True)
        result = dict(schema='tt21-body-state-1', source_sha256=_hash(SOURCE),
                      dry_catalog_sha256=_hash(TT20 / 'curriculum-dry-catalog.npz'),
                      frame_seconds=arena.control_dt, gates=WINDOWS, cameras=['left', 'right'],
                      source=SOURCE.name, cases=cases, summary=summarize(cases), replayed_points=replay(arena, cases),
                      wall_seconds=perf_counter() - clock)
        output.parent.mkdir(exist_ok=True, parents=True)
        _write(output, result)
        print(result['summary'], flush=True)
    finally:
        arena.close()
        ranker.close()
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path, nargs='?', default=ROOT / 'artifacts' / 'tt21' / 'body-state.json.gz')
    run(parser.parse_args().output)
