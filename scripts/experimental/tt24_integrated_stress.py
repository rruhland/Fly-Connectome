"""Qualify one carried wrist bundle across paired physical compound stresses."""

import argparse
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from .tt4_transfer_trial import _hash, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE, CarriedWristPolicy, commands
from .tt24_donor_trial import ROOT
from .tt24_posture import WidePosture
from .tt24_public_body import FUSED_OUTPUT, load_public
from .tt24_ready import prepare_step


OUTPUT = ROOT / 'integrated-stress.json.gz'
ROSTER_SEED = 1422789647
PER_BODY = 8
CAP = 900
MODES = ('bundle', 'reset_origin', 'passive')
BODIES = ('stock', 'alternate')


def schedule(seed=ROSTER_SEED, per_body=PER_BODY):
    rng = np.random.default_rng(seed)
    rows = []
    for body in BODIES:
        for high_first in rng.permutation([False, True] * (per_body//2)):
            low, high = rng.uniform(.78, .88), rng.uniform(1.13, 1.26)
            first, second = (high, low) if high_first else (low, high)
            rows.append(dict(body=body, learner_seed=int(rng.integers(1, 2**31-1)),
                             launcher_seed=int(rng.integers(1, 2**31-1)),
                             prep_ticks=int(rng.integers(14, 39)),
                             mass_ratio=float(rng.uniform(.82, 1.18)), drag_ratio=float(rng.uniform(.82, 1.22)),
                             changes=[(int(rng.integers(180, 223)), float(first)),
                                      (int(rng.integers(360, 415)), float(second))]))
    return [rows[i] for i in rng.permutation(len(rows))]


class InstalledArm:
    """Target physics and evaluator bookkeeping kept outside the wrist interface."""

    def __init__(self, body):
        self.arena = WidePosture(view='wrist', width=192, height=144, max_steps=CAP, arm=body)
        model = self.arena._model
        self.ball = self.arena._ball_body
        self.mass = float(model.body_mass[self.ball])
        self.inertia = model.body_inertia[self.ball].copy()
        self.gravity = model.opt.gravity.copy()
        self.regime = 'standard' if body == 'stock' else 'changed'

    def reset(self, plan, location, render):
        arena = self.arena
        model = arena._model
        mass_ratio = plan['mass_ratio']
        model.opt.gravity[:] = self.gravity
        model.body_mass[self.ball] = mass_ratio*self.mass
        model.body_inertia[self.ball] = mass_ratio*self.inertia
        mujoco.mj_setConst(model, arena._data)
        arena.reset(seed=plan['learner_seed'], split='held', regime=self.regime, render=False)
        # The chosen drag multiplier applies to drag acceleration independently
        # of mass, while physical impact mass and inertia still change.
        arena._drag_force_scale *= mass_ratio*plan['drag_ratio']
        arena._ball_position[:] = location
        arena._ball_velocity[:] = 0.
        arena._ball_spin[:] = 0.
        mujoco.mj_forward(model, arena._data)
        return arena._observation(render)

    def switch_gravity(self, tick, ratio):
        arena = self.arena
        active_contacts = [c for c in arena._data.contact if c.efc_address >= 0 and
                           arena._ball_geom in (c.geom1, c.geom2)]
        state = dict(tick=tick, absolute_time_s=float(arena._data.time), ratio=ratio,
                     airborne=not active_contacts, ball_velocity=arena._ball_velocity.tolist(),
                     ball_xyz=arena._ball_position.tolist())
        arena._model.opt.gravity[:] = self.gravity*ratio
        return state


def episode(installation, bundle, plan, launch, mode, *, replay=None):
    arena = installation.arena
    location, velocity, spin = launch
    lead = plan['prep_ticks']
    changes = dict(plan['changes'])
    render = mode != 'passive' and replay is None
    obs = installation.reset(plan, location, render)
    model_index = None
    if render and mode == 'bundle':
        bundle.begin_release(obs)
        model_index = bundle.calibration_index
        act = bundle.act
    elif render:
        stock = load_public(FUSED_OUTPUT)
        issued = np.zeros(5)
        act = lambda observation: commands(stock.observe(observation), bundle.weights, issued)
        model_index = 0
    tape, events = [], []
    for tick in range(lead+CAP):
        physical_tick = tick-lead
        if physical_tick in changes:
            events.append(installation.switch_gravity(physical_tick, changes[physical_tick]))
        if replay is not None:
            action = np.asarray(replay[tick])
        elif render:
            action = act(obs)
        else:
            action = np.zeros(5)
        tape.append(action.tolist())
        if tick < lead:
            prepare_step(arena, action, location)
            if tick+1 == lead:
                arena._ball_velocity[:] = velocity
                arena._ball_spin[:] = spin
                mujoco.mj_forward(arena._model, arena._data)
            obs = arena._observation(render)
        else:
            obs, _, done = arena.step(action, render=render)
            if replay is not None and bool(done) != (tick+1 == len(replay)):
                raise ValueError(f'physics replay changed the terminal tick on {plan["learner_seed"]}')
            if done:
                break
    metrics = arena.judge_metrics()
    physical_seconds = metrics['elapsed_s']-lead*arena.dt
    return dict(actions=tape, metrics=metrics, model_index=model_index, changes_reached=events,
                right_censored=metrics['termination']=='timeout', observed_seconds=physical_seconds,
                loss_seconds=None if metrics['termination']=='timeout' else physical_seconds)


def summary(rows, modes=MODES):
    result = {}
    for mode in modes:
        trials = [row['modes'][mode] for row in rows]
        streaks = np.array([trial['metrics']['longest_consecutive_rebounds'] for trial in trials])
        time = np.array([trial['observed_seconds'] for trial in trials])
        both = [trial for trial in trials if len(trial['changes_reached'])==2]
        post_second = [sum(e['contact_time'] > trial['changes_reached'][1]['absolute_time_s']
                           for e in trial['metrics']['rebound_events']) for trial in both]
        interval_returns = [sum(trial['changes_reached'][0]['absolute_time_s'] < e['contact_time'] <=
                                trial['changes_reached'][1]['absolute_time_s']
                                for e in trial['metrics']['rebound_events']) for trial in both]
        predicted = [trial['model_index']==BODIES.index(row['plan']['body']) for row, trial in zip(rows,trials,strict=True)
                     if trial['model_index'] is not None]
        result[mode] = dict(count=len(trials), mean_streak=float(streaks.mean()), longest=int(streaks.max()),
                            threshold_counts={f'at_least_{k}':int((streaks>=k).sum()) for k in (10,20,30)},
                            reached_first=sum(bool(trial['changes_reached']) for trial in trials), reached_both=len(both),
                            airborne_switches=sum(event['airborne'] for trial in trials for event in trial['changes_reached']),
                            reached_switches=sum(len(trial['changes_reached']) for trial in trials),
                            legal_rebound_after_both=sum(a>0 and b>0 for a,b in zip(interval_returns,post_second,strict=True)),
                            post_second_rebounds=post_second, max_post_second=max(post_second,default=0),
                            mean_time_alive_with_cap_s=float(time.mean()), median_time_alive_with_cap_s=float(np.median(time)),
                            completed_without_loss=sum(trial['right_censored'] for trial in trials),
                            correct_calibration=sum(predicted) if predicted else None,
                            streaks=streaks.tolist())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=BUNDLE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    start = perf_counter()
    bundle = CarriedWristPolicy(args.bundle)
    plans = schedule()
    report = dict(schema='tt24-integrated-stress-1', bundle_sha256=_hash(args.bundle),
                  bundle_bytes=args.bundle.stat().st_size, bundle_reuse=bundle.reuse, fused_sha256=_hash(FUSED_OUTPUT),
                  raster=[192,144], roster_seed=ROSTER_SEED, cap_ticks=CAP, schedule=plans, cases=[], verified_tapes=0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output, report)
    arms = {body:InstalledArm(body) for body in BODIES}
    launcher = BounceArena(view='none')
    try:
        for plan in plans:
            launcher.reset(seed=plan['launcher_seed'], split='held', render=False)
            launch = tuple(v.copy() for v in (launcher._ball_position, launcher._ball_velocity, launcher._ball_spin))
            arm = arms[plan['body']]
            modes = {}
            for mode in MODES:
                row = episode(arm, bundle, plan, launch, mode)
                checked = episode(arm, bundle, plan, launch, mode, replay=row['actions'])
                for key in ('metrics','changes_reached'):
                    if digest_json(row[key]) != digest_json(checked[key]):
                        raise ValueError(f'physical replay changed {key} on {plan["learner_seed"]} ({mode})')
                modes[mode] = row
                report['verified_tapes'] += 1
            report['cases'].append(dict(plan=plan, launch=dict(position=launch[0].tolist(), velocity=launch[1].tolist(),
                                                               spin=launch[2].tolist()), modes=modes))
            _write(args.output, report)
            print(plan['learner_seed'], plan['body'], modes['bundle']['model_index'],
                  [modes[mode]['metrics']['longest_consecutive_rebounds'] for mode in MODES], flush=True)
    finally:
        launcher.close()
        for arm in arms.values():
            arm.arena.close()
    if _hash(args.bundle) != report['bundle_sha256']:
        raise ValueError('the carried bundle changed during the test')
    report['summary'] = {'all': summary(report['cases'])}
    report['summary'].update({body:summary([row for row in report['cases'] if row['plan']['body']==body]) for body in BODIES})
    report['wall_seconds'] = perf_counter()-start
    _write(args.output, report)
    print(report['summary']['all'], flush=True)


if __name__ == '__main__':
    main()
