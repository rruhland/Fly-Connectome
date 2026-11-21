"""Grounded, evaluator-only substitutions at a failed optical rally continuation."""

import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from .tt4_arena import TABLE_HEIGHT, mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_flight import optical_start
from .tt21_coach_oracle import PHASE
from .tt22_arc_observer import arc
from .tt22_cycle import World
from .tt22_cycle_optical import OpticalReceiver
from .tt22_cycle_trial import OUTPUT


SOURCE = OUTPUT/'cycle-optical-held.json.gz'


def _vision(receiver, arena, tick):
    receiver.tracker.update(arena.render_cameras(), tick=tick)
    ball = receiver.tracker.ball
    point = np.asarray(ball.position) if ball is not None and ball.observed else None
    receiver.history.append((tick, point))
    return receiver.calibration.estimate(tick, optical_start(receiver.history, receiver.law.gravity))


def _actual(camera_world, trial, tape):
    if tape is None:
        return None
    live, arena = camera_world.arena, trial.arena
    mujoco.mj_copyData(arena.data, arena.model, live.data)
    for key in PHASE:
        if hasattr(live, key):
            setattr(arena, key, deepcopy(getattr(live, key)))
    arena.events = list(live.events)
    used, before = [], len(live.events)
    for rate in tape:
        action = np.r_[rate, 0.]
        used.append(action.tolist())
        _, _, end, cap, _ = trial.step(action, np.zeros(4))
        if arena.legal_returns > live.legal_returns or end or cap:
            break
    return dict(legal=bool(arena.legal_returns > live.legal_returns), actions=used, events=arena.events[before:],
                physical_contact=bool(arena.contacts > live.contacts), result=arena.point_result())


def _terminal_decision(row):
    opponent = [e['step'] for e in row['events'] if e['kind']=='contact' and e.get('side')==-1]
    after = opponent[-1] if opponent else -1
    terminal = [n for n in row['decisions'] if n['tick'] >= after]
    applied = [n for n in terminal if 'chosen' in n]
    if not terminal:
        raise ValueError('no terminal optical plan to diagnose')
    return (applied or terminal)[-1]


def audit(source_path, destination, branch, seed, *, components=False, prebounce_arc=False):
    data = _read(source_path)
    row = next(r for r in data['rows'][branch] if r['seed']==seed)
    stage = _terminal_decision(row)
    tick = stage['tick']
    styles = [tuple(style) for style in data['styles']]
    exact, replay = World(), World()
    near = OpticalReceiver(exact.arena.lens_intrinsics(), styles=styles,
                           samples=row['online_samples'] if branch=='teacher-plus-own-optics' else [])
    try:
        exact.reset(seed)
        near.reset()
        for t, (a,b) in enumerate(row['actions'][:tick]):
            _vision(near, exact.arena, t)
            near.twin.advance(a)
            exact.step(a,b)
        estimate = _vision(near, exact.arena, tick)
        if (estimate is None or not np.array_equal(estimate[0], stage['optical'])
                or not np.array_equal(30*estimate[1], stage['optical_velocity_m_s'])):
            raise ValueError('the evaluator failed to reconstruct the archived optical decision inputs')
        near._shadow(tick, estimate)
        truth, model = exact.arena, near.world.arena
        source = mujoco.MjData(model.model)
        mujoco.mj_copyData(source, model.model, model.data)
        ball, velocity = model._ball_joint, model._ball_dof
        optics_bounced = model._received_bounce
        near_body = model._arm_ids[1][2]
        fitted = arc(near.history, np.zeros(3)) if prebounce_arc else None
        if prebounce_arc and (optics_bounced or truth._received_bounce):
            raise ValueError('smooth arc comparison requires a prebounce frame')

        def assess(exact_ball, exact_mechanics, component=None):
            mujoco.mj_copyData(model.data, model.model, truth.data if exact_mechanics else source)
            initialize = truth.data if exact_ball else source
            model.data.qpos[ball:ball+7] = initialize.qpos[ball:ball+7]
            model.data.qvel[velocity:velocity+6] = initialize.qvel[velocity:velocity+6]
            model._received_bounce = truth._received_bounce if exact_ball else optics_bounced
            if component == 'position':
                model.data.qpos[ball:ball+3] = truth.ball
            elif component == 'velocity':
                model.data.qvel[velocity:velocity+3] = truth.velocity
            elif component == 'spin':
                model.data.qvel[velocity+3:velocity+6] = truth.spin
            elif component in ('arc', 'arc_velocity'):
                if fitted is None:
                    return dict(selection=dict(solver_legal=0, arc_available=False), enacted_in_exact_world=None)
                position, speed = fitted
                if component == 'arc':
                    model.data.qpos[ball:ball+3] = position + [0., 0., TABLE_HEIGHT]
                model.data.qvel[velocity:velocity+3] = speed*30.
            mujoco.mj_forward(model.model, model.data)
            candidate, choice = near.planner.choose(1, wide=not stage['donor_subset'])
            return dict(selection=choice, enacted_in_exact_world=_actual(exact, replay, candidate))
        errors = dict(stereo_position_mm=(1000*(estimate[0]- (truth.ball-[0.,0.,TABLE_HEIGHT]))).tolist(),
                      velocity_error_m_s=(30*estimate[1]-truth.velocity).tolist(),
                      spin_model_rad_s=source.qvel[velocity+3:velocity+6].tolist(),
                      spin_actual_rad_s=truth.spin.tolist(),
                      requested_joint_target_max_abs_rad=float(np.max(abs(source.ctrl[near_body]-truth.data.ctrl[near_body]))),
                      blade_center_mm=(1000*(source.geom('learner_racket').xpos-truth.paddle)).tolist(),
                      optical_bounce=bool(optics_bounced), actual_bounce=bool(truth._received_bounce))
        controls = {}
        if prebounce_arc:
            variants = (('stereo_and_twin', False, False, None), ('smooth_rgb_arc', False, False, 'arc'),
                        ('smooth_rgb_arc_velocity_only', False, False, 'arc_velocity'))
            if fitted is not None:
                errors.update(rgb_arc_position_mm=(1000*(fitted[0]-(truth.ball-[0.,0.,TABLE_HEIGHT]))).tolist(),
                              rgb_arc_velocity_error_m_s=(30*fitted[1]-truth.velocity).tolist())
        elif components:
            variants = tuple((f'exact_{part}_only', False, False, part) for part in ('position','velocity','spin'))
        else:
            variants = (('stereo_and_twin',False,False,None), ('exact_ball_only',False,True,None),
                        ('exact_mechanics_only',True,False,None), ('exact_both',True,True,None))
        for name, body, ball_state, part in variants:
            controls[name] = assess(ball_state, body, part)
            action = controls[name]['enacted_in_exact_world']
            print(name, controls[name]['selection']['solver_legal'], 'chosen' in controls[name]['selection'],
                  None if action is None else (action['physical_contact'],action['legal']), flush=True)
        _write(destination, dict(schema='tt22-cycle-terminal-audit-1', source=source_path.name, source_sha256=_hash(source_path),
                                 branch=branch, seed=seed, terminal_decision=stage, model_errors=errors,
                                 variations=controls, evaluator_only=True, arc_has_no_privileged_inputs=prebounce_arc))
        return controls
    finally:
        near.close()
        exact.close()
        replay.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--branch', default='teacher-plus-own-optics')
    mode = p.add_mutually_exclusive_group()
    mode.add_argument('--components', action='store_true')
    mode.add_argument('--prebounce-arc', action='store_true')
    p.add_argument('--seed', type=int, default=1134144980)
    p.add_argument('--source', type=Path, default=SOURCE)
    p.add_argument('--output', type=Path, default=OUTPUT/'cycle-audit.json.gz')
    args = p.parse_args()
    audit(args.source, args.output, args.branch, args.seed, components=args.components,
          prebounce_arc=args.prebounce_arc)
