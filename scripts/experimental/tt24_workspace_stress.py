"""Compare scene-based and fixed workspace adjustments on a fresh compound course."""

import argparse
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import BODIES, CAP, InstalledArm, episode, schedule, summary
from .tt24_tool_equilibrium import (LOCAL_OUTPUT, OUTPUT as GEOMETRY, ToolEquilibrium,
                                   bounded_xy)


OUTPUT = ROOT / 'workspace-stress.json.gz'
SEED = 392648105
MODES = ('bundle','local_xy','fixed_mean_xy')


class StressPolicy(ToolEquilibrium):
    def translation(self, center):
        if self.fixed_mean:
            return self.prior_shifts[self.calibration_index]
        return super().translation(center)


def development_shifts(policy, archives):
    means, examples = [], []
    for (body, mode), archive in zip((('stock','workspace'),('alternate','workspace_local')),archives,strict=True):
        rows = []
        for case in archive['cases']:
            if case['plan']['body'] != body:
                continue
            base, adjusted = [case['modes'][m]['reference'] for m in ('bundle',mode)]
            if adjusted['visible'] is None:
                continue
            shift = bounded_xy(np.array(adjusted['visible']), np.array(base['pose']),
                               policy.stock_decoder.default, policy.workspace_limit)
            rows.append(dict(seed=case['plan']['learner_seed'], shift=shift.tolist()))
        means.append(np.mean([row['shift'] for row in rows],axis=0))
        examples.append(rows)
    return np.asarray(means), examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--resume',action='store_true',help='continue after the last fully verified start')
    args = parser.parse_args()
    start = perf_counter()
    sources = [_read(p) for p in (GEOMETRY,LOCAL_OUTPUT)]
    bundle_sha = _hash(BUNDLE)
    if any(receipt['bundle_sha256']!=bundle_sha for receipt in sources):
        raise ValueError('workspace development used a different carried bundle')
    policy = StressPolicy(BUNDLE,sources[0]['geometry']['workspace_cap_m'])
    policy.prior_shifts, examples = development_shifts(policy,sources)
    plans = schedule(seed=SEED,per_body=12)
    report = dict(schema='tt24-workspace-stress-1', seed=SEED,cap_ticks=CAP,bundle_sha256=bundle_sha,
                  development_sources={p.name:_hash(p) for p in (GEOMETRY,LOCAL_OUTPUT)},
                  fixed_mean_shifts_m=policy.prior_shifts.tolist(),development_shifts=examples,
                  modes=MODES,schedule=plans,cases=[],verified=0)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.resume:
        saved = _read(args.output)
        headers = (key for key in report if key not in ('cases','verified'))
        if any(digest_json(saved[key])!=digest_json(report[key]) for key in headers):
            raise ValueError('the retained schedule, controller or sources changed')
        recorded = saved['cases']
        matching = digest_json([row['plan'] for row in recorded])==digest_json(plans[:len(recorded)])
        if not matching or saved['verified']!=len(recorded)*len(MODES) or any(set(c['modes'])!=set(MODES) for c in recorded):
            raise ValueError('the retained report is missing part of its verified prefix')
        report = saved
        report.setdefault('resumed_after_cases',[]).append(len(recorded))
    _write(args.output,report)
    print('fixed prior shifts, mm',1000*policy.prior_shifts,flush=True)
    arms = {body:InstalledArm(body) for body in BODIES}
    dispenser = BounceArena(view='none')
    try:
        for plan in plans[len(report['cases']):]:
            dispenser.reset(seed=plan['launcher_seed'],split='held',render=False)
            launch = tuple(v.copy() for v in (dispenser._ball_position,dispenser._ball_velocity,dispenser._ball_spin))
            modes = {}
            for mode in MODES:
                policy.mode = 'bundle' if mode=='bundle' else 'workspace_local'
                policy.fixed_mean = mode=='fixed_mean_xy'
                row = episode(arms[plan['body']],policy,plan,launch,'bundle')
                row['reference'] = policy.receipt()
                checked = episode(arms[plan['body']],policy,plan,launch,'bundle',replay=row['actions'])
                if any(digest_json(row[k])!=digest_json(checked[k]) for k in ('metrics','changes_reached')):
                    raise ValueError('command replay changed the score or a gravity cut')
                modes[mode] = row
                report['verified'] += 1
            active,fixed = (modes[m] for m in ('local_xy','fixed_mean_xy'))
            tick = active['reference']['switch']
            if (active['reference']['visible']!=fixed['reference']['visible'] or
                    tick!=fixed['reference']['switch'] or active['actions'][:tick]!=fixed['actions'][:tick]):
                raise ValueError('the active and fixed comparisons have different viewing prefixes')
            report['cases'].append(dict(plan=plan,launch=[v.tolist() for v in launch],modes=modes))
            _write(args.output,report)
            print(plan['body'],plan['learner_seed'],[modes[m]['metrics']['longest_consecutive_rebounds'] for m in MODES],
                  flush=True)
    finally:
        dispenser.close()
        for arm in arms.values():
            arm.arena.close()
    if _hash(BUNDLE)!=bundle_sha:
        raise ValueError('the carried bundle changed during held evaluation')
    report['summary'] = {'all':summary(report['cases'],MODES)}
    for body in BODIES:
        report['summary'][body] = summary([case for case in report['cases'] if case['plan']['body']==body],MODES)
    report['last_process_wall_seconds'] = perf_counter()-start
    _write(args.output,report)
    print(report['summary'],flush=True)


if __name__=='__main__':
    main()
