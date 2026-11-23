"""Check if a single bounded effort parameter has reliable physical headroom."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import InstalledArm
from .tt24_long_wrist import play
from .tt24_tool_equilibrium import ToolEquilibrium


TAG = 'live-effort'
PLAN = ROOT / f'{TAG}-plan.json'
HEADROOM = ROOT / f'{TAG}-headroom.json.gz'
ANCHORS = (0, 7)
MODES = {'lower': .82, 'base': 1., 'higher': 1.18}
CAP = 720
STEP = 5.2 / 60.
WORKSPACE_LIMIT = .0557


class EffortPolicy(ToolEquilibrium):
    """Apply one global effort scale and carry the actual issued increment."""

    def __init__(self, gain):
        super().__init__(BUNDLE, WORKSPACE_LIMIT)
        self.mode = 'workspace_local'
        self.gain = gain

    def act(self, obs):
        nominal = super().act(obs)
        if self.gain == 1.:
            return nominal
        applied = np.clip(self.gain * nominal, -1., 1.)
        self.issued += STEP * (applied - nominal)
        return applied


def source_file(index):
    return ROOT / f'long-wrist-{index}.json.gz'


def register():
    schedules = []
    for case in ANCHORS:
        path = source_file(case)
        archive = _read(path)
        if archive['plan']['body'] != 'alternate' or archive['results']['live']['metrics']['steps'] < CAP:
            raise ValueError('predeclared headroom anchor lacks an already feasible alternate prefix')
        schedules.append(dict(index=case, source_sha256=_hash(path), plan=archive['plan']))
    result = dict(schema='tt24-live-effort-1', tested=schedules, effort=MODES, cap=CAP,
                  portable_sha256=_hash(BUNDLE), reference_workspace_limit_m=WORKSPACE_LIMIT,
                  primary='longest consecutive physically qualified rebound run',
                  gate='one common nonunit effort adds at least two consecutive rebounds in both feasible anchors and at least 20 percent pooled',
                  selection='previously surviving altered starts used only to assess headroom',
                  gameplay_learning_max_drops_per_cohort=4,
                  eventual_update='bounded scalar rate gain using causal random perturbation eligibility modulated by received scalar rewards',
                  fixed_phase_or_contact_labels_in_controller=False)
    if PLAN.exists():
        if _read(PLAN) != result:
            raise ValueError('the preregistered effort headroom changed')
    else:
        _write(PLAN, result)
    return result


def headroom_case(spec):
    case, plan = spec['index'], spec['plan']
    source = source_file(case)
    if _hash(source) != spec['source_sha256']:
        raise ValueError('archived headroom reference changed')
    reference = _read(source)['results']['live']
    arm = InstalledArm(plan['body'])
    arm.arena.max_steps = CAP
    server = BounceArena(view='none')
    try:
        server.reset(seed=plan['launcher_seed'], split='held', render=False)
        location = tuple(x.copy() for x in (server._ball_position, server._ball_velocity, server._ball_spin))
        results = {}
        for name, gain in MODES.items():
            controller = EffortPolicy(gain)
            observed = play(arm, controller, plan, location, 'live')
            replay = play(arm, controller, plan, location, 'live', observed['actions'])
            if digest_json(observed) != digest_json(replay):
                raise ValueError(f'effort case {case} did not replay')
            if name == 'base':
                count = len(observed['actions'])
                if observed['actions'] != reference['actions'][:count] or observed['metrics']['steps'] != CAP:
                    raise ValueError(f'unit effort differs from the archived carried controller, case {case}')
                horizon = observed['metrics']['elapsed_s']
                expected = [e for e in reference['events'] if e['absolute_time_s'] < horizon]
                if expected != observed['events']:
                    raise ValueError('the unchanged prefix reached different physical parameters')
            results[name] = observed
        print('effort bench', case, {mode: (row['metrics']['longest_consecutive_rebounds'], row['metrics']['steps'])
                                    for mode, row in results.items()}, flush=True)
        return dict(index=case, results=results)
    finally:
        arm.arena.close()
        server.close()


def adjudicate(cases):
    keys = ('longest_consecutive_rebounds', 'rebounds', 'steps')
    summaries = {mode: {key: [case['results'][mode]['metrics'][key] for case in cases] for key in keys} for mode in MODES}
    nominal = np.array(summaries['base']['longest_consecutive_rebounds'])
    differences = {mode: np.array(summaries[mode]['longest_consecutive_rebounds']) - nominal for mode in ('lower', 'higher')}
    passed = [mode for mode, gain in differences.items() if (gain >= 2).all() and gain.sum() >= .20 * nominal.sum()]
    return dict(summary=summaries, paired_longest={name: q.tolist() for name, q in differences.items()},
                qualifying_directions=passed, commence_adaptation=bool(passed))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    plan = register()
    report = _read(HEADROOM) if HEADROOM.exists() else dict(plan_sha256=_hash(PLAN), cases=[])
    if report['plan_sha256'] != _hash(PLAN):
        raise ValueError('the held feasibility receipt does not match its registered plan')
    tasks = plan['tested'][len(report['cases']):]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context('spawn')) as pool:
        for outcome in pool.map(headroom_case, tasks):
            report['cases'].append(outcome)
            _write(HEADROOM, report)
    report.update(adjudicate(report['cases']))
    _write(HEADROOM, report)
    print('effort headroom result', report['summary'], 'paired', report['paired_longest'],
          'proceed to adaptation', report['commence_adaptation'], flush=True)


if __name__ == '__main__':
    main()
