"""Update a native Pong motor map in two poorly observed directions on another arm."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_current_pause import Fixture, replay
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import InstalledArm, schedule
from .tt24_motor_bank import I, STEP, decoder
from .tt24_pong_arm_local import samples, windows
from .tt24_tool_equilibrium import ToolEquilibrium
from .tt4_transfer_trial import _hash, _read, _write


TAG = 'tip-two-probes'
PLAN = ROOT / f'{TAG}-plan.json'
CAP, PER_WIRE, SEED = 600, 6, 1793663587
AMPLITUDE, PULSE, REST, INITIAL_REST = .075, 4, 10, 12
MODES = ('source', 'source_plus_two', 'stale_plus_two', 'known')


def native_path(index):
    return ROOT / f'learned-tip-transfer-{index}.json.gz'


def coverage(clips, fit):
    """Give the public command windows that supported each fit row equal total weight."""
    gram = np.zeros((5, 5))
    for joint, row in enumerate(fit):
        x, _, accepted = samples(windows(clips, row['alpha']))
        seen = x[accepted[:, joint]]
        g = seen.T @ seen
        gram += g / np.trace(g)
    values, vectors = np.linalg.eigh(gram / 5.)
    axes = vectors[:, :2].T.copy()
    for axis in axes:
        pivot = np.argmax(abs(axis))
        axis *= np.sign(axis[pivot])
    return axes, np.sqrt(np.maximum(values, 0.) / values[-1])


def register():
    sources = []
    for index in (0, 1):
        path = native_path(index)
        native = _read(path)
        if not native['usable']:
            raise ValueError('the native player did not qualify')
        clips, fit = native['clips'][:12], native['fits']['source']
        axes, singular = coverage(clips, fit)
        launches = [p for p in schedule(SEED+index, PER_WIRE) if p['body']=='alternate']
        sources.append(dict(index=index, native=path.name, native_sha256=_hash(path), plans=launches,
                            axes=axes.tolist(), relative_singular_values=singular.tolist(),
                            source_points=len(clips), source_seconds=sum(len(c['commands']) for c in clips)/60.))
    plan = dict(schema='tt24-tip-two-probes-1', sources=sources, bundle_sha256=_hash(BUNDLE),
                cap=CAP, amplitude_rad=AMPLITUDE, pulse_ticks=PULSE, rest_ticks=REST,
                initial_rest_ticks=INITIAL_REST, eye_resolution=[192, 144], conditions=list(MODES),
                matched_setup='balanced encoder-only probes and original source optical prep for every condition',
                evaluation='fresh paired longest consecutive rebounds, ten+ and twenty+; report true and both controls')
    if PLAN.exists():
        if digest_json(_read(PLAN)) != digest_json(plan):
            raise ValueError('the registered source or test data changed')
    else:
        _write(PLAN, plan)
    return plan


def measure(read, initial, axes):
    """Use opposite open-loop encoder displacements and return to the command origin."""
    q = np.array(initial, dtype=float)
    for _ in range(INITIAL_REST):
        q = read(np.zeros(5))
    start = q.copy()
    traces = [initial.tolist(), q.tolist()]
    columns = []
    for axis in axes:
        endpoints = []
        requested = 0.
        for target in (AMPLITUDE, -AMPLITUDE, 0.):
            command = ((target-requested)/(PULSE*STEP))*axis
            for _ in range(PULSE):
                q = read(command)
            for _ in range(REST):
                q = read(np.zeros(5))
            endpoints.append(q.copy())
            traces.append(q.tolist())
            requested = target
        columns.append((endpoints[0]-endpoints[1])/(2*AMPLITUDE))
    measured = np.stack(columns, axis=1)
    return measured, dict(encoders_by_endpoint=traces, closed_return_rad=(q-start).tolist(),
                          measured_columns=measured.tolist())


def update(prior, axes, columns):
    return prior + (columns - prior @ axes.T) @ axes


def run_drop(arm, wrist, plan, launch, wire, estimate, common, axes):
    fixture = Fixture(arm, plan, launch, wire)
    obs = fixture.start(False)

    def read(command):
        received, _ = fixture.issue(command, 'encoder_probe', False)
        return received['joints'].astype(float)

    measured, inspection = measure(read, obs['joints'].astype(float), axes)
    inspection['ticks'] = len(fixture.actions)
    inspection['command_return_rad'] = (STEP*np.sum(fixture.requests, axis=0)).tolist()
    if np.linalg.norm(inspection['command_return_rad']) > 1e-9:
        raise ValueError('balanced actuator request did not return to origin')
    installed = estimate(measured) if callable(estimate) else estimate
    obs = fixture.env._observation(True)
    wrist.begin_release(obs)
    for _ in range(plan['prep_ticks']):
        request = np.clip(common @ wrist.act(obs), -1., 1.)
        obs, _ = fixture.issue(request, 'optical_setup', True)
    obs = fixture.release(True)
    live = decoder(installed)
    while True:
        request = np.clip(live @ wrist.act(obs), -1., 1.)
        obs, done = fixture.issue(request, 'playing', True)
        if done:
            break
    result = fixture.score() | dict(probe=inspection, estimate=installed.tolist(),
                                    eye=wrist.calibration_index, live_seconds=fixture.live/60.)
    for command, request in zip(result['actions'], result['requests'], strict=True):
        if not np.array_equal(np.clip(wire @ request, -1., 1.), command):
            raise ValueError('recorded physical command does not match the fixture')
    replay(fixture, result)
    return result


def check_setup(conditions, plan):
    baseline = conditions['source']
    tick = baseline['release_tick']
    if tick != INITIAL_REST + 6*(PULSE+REST) + plan['prep_ticks']:
        raise ValueError('release clock omitted paid setup commands')
    for result in conditions.values():
        if (result['release_tick'] != tick or result['release_state'] != baseline['release_state']
                or result['actions'][:tick] != baseline['actions'][:tick]
                or result['requests'][:tick] != baseline['requests'][:tick]
                or result['probe'] != baseline['probe']):
            raise ValueError('two-probe controls did not start from the same physical history')


def run(source):
    index = source['index']
    path = ROOT / f'{TAG}-{index}.json.gz'
    prior = _read(native_path(index))
    native = np.asarray([r['matrix_row'] for r in prior['fits']['source']])
    wire = np.asarray(prior['wiring'])
    axes = np.asarray(source['axes'])
    estimates = dict(source=native, source_plus_two=lambda cols: update(native, axes, cols),
                     stale_plus_two=lambda cols: update(I, axes, cols), known=wire)
    arm = InstalledArm('alternate')
    arm.arena.max_steps = CAP
    launcher = BounceArena(view='none')
    wrist = ToolEquilibrium(BUNDLE, .0557)
    wrist.mode = 'workspace_local'
    common = decoder(native)
    report = dict(index=index, plan_sha256=_hash(PLAN), rows=[])
    try:
        for plan in source['plans']:
            launcher.reset(seed=plan['launcher_seed'], split='held', render=False)
            launch = tuple(v.copy() for v in (launcher._ball_position, launcher._ball_velocity, launcher._ball_spin))
            conditions = {name:run_drop(arm, wrist, plan, launch, wire, model, common, axes)
                          for name, model in estimates.items()}
            check_setup(conditions, plan)
            error = {name:float(np.linalg.norm(value['estimate']-wire) / np.linalg.norm(wire))
                     for name, value in conditions.items()}
            report['rows'].append(dict(plan=plan, conditions=conditions, evaluator_relative_map_error=error))
            _write(path, report)
            print(index, plan['learner_seed'], 'longest',
                  [c['metrics']['longest_consecutive_rebounds'] for c in conditions.values()],
                  'map error', np.round(list(error.values()), 4).tolist(), flush=True)
    finally:
        launcher.close()
        arm.arena.close()
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    plan = register()
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        list(pool.map(run, plan['sources']))


if __name__ == '__main__':
    main()
