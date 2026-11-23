"""Compare two encoder calibration policies through one uninterrupted receiving course."""

from time import perf_counter

import numpy as np

from .tt19_view_full_game import digest_json
from .tt24_autonomous_drive import MotorMemory
from .tt24_bounce import BounceArena
from .tt24_carried_wrist import BUNDLE
from .tt24_donor_trial import ROOT
from .tt24_integrated_stress import InstalledArm
from .tt24_motor_bank import I, MotorBank, decoder
from .tt24_single_arm_course import CAP, STAGES, course_plan, trial
from .tt24_tool_equilibrium import ToolEquilibrium
from .tt24_validated_motor import RELOCATION, relocated
from .tt4_transfer_trial import _read, _write


OUTPUT = ROOT / 'validated-motor-course.json.gz'
INDEX = 720119
MODES = ('baseline','validated')


class ValidatedBank(MotorBank):
    def connect(self, joints, step, seed, *, measured_settling=False):
        estimate, report = relocated(joints, step, seed, self.maps)
        self.selected = report['known']
        if self.selected is None:
            self.selected = len(self.maps)
            self.maps.append(estimate)
        self.inverse = decoder(estimate)
        return report


def aggregate(rows, stages):
    result = {}
    for name in MODES:
        entries = [row['modes'][name] for row in rows]
        streak = np.asarray([row['metrics']['longest_consecutive_rebounds'] for row in entries])
        changed = np.asarray(stages[:len(entries)]) != 0
        fits = [row for row in entries if row['report'] is not None]
        result[name] = dict(rebounds=sum(row['metrics']['rebounds'] for row in entries),
                           changed_rebounds=sum(row['metrics']['rebounds']
                                                for row, different in zip(entries, changed, strict=True) if different),
                           mean_streak=float(streak.mean()), max_streak=int(streak.max()),
                           changed_mean_streak=float(streak[changed].mean()),
                           ten=int((streak>=10).sum()), changed_ten=int((streak[changed]>=10).sum()),
                           exact_replays=len(entries), inspections=len(fits),
                           total_inspection_s=sum(len(r['inspection']) for r in entries)/60.,
                           total_occupied_s=sum(r['metrics']['elapsed_s'] for r in entries),
                           capped_play_s=sum(r['capped_release_s'] for r in entries),
                           inspection_error_mean=float(np.mean([r['active_error'] for r in fits])),
                           inspections_over_20pct=sum(r['active_error']>.2 for r in fits),
                           changed_error_mean=float(np.mean([r['active_error'] for r,s in
                                                             zip(entries,changed,strict=True) if s])),
                           accepted_relocations=sum(bool((r['report'].get('relocation') or {}).get('accepted',False))
                                                    for r in fits))
    return result


def restore(saved):
    banks = {}
    monitors = {}
    previous = {}
    for name, kind in zip(MODES,(MotorBank,ValidatedBank),strict=True):
        status = saved.get('memory',{}).get(name,{})
        bank = kind(np.asarray(status['maps'])) if status else kind()
        bank.selected = status.get('selected')
        if bank.selected is not None:
            bank.inverse = decoder(bank.maps[bank.selected])
        memory = MotorMemory()
        memory.pending = status.get('pending',False)
        banks[name], monitors[name], previous[name] = bank, memory, status.get('previous')
    return banks,monitors,previous


def run():
    mechanical = _read(RELOCATION)
    original, measured = (mechanical['summary'][kind] for kind in MODES)
    if measured['inverse_mean'] >= original['inverse_mean'] or measured['errors_over_20pct'] >= original['errors_over_20pct']:
        raise ValueError('fresh unscored mechanics must improve before an integrated course')
    plans, wires = course_plan(INDEX)
    report = _read(OUTPUT) if OUTPUT.exists() else dict(index=INDEX, rows=[], stages=list(STAGES))
    banks, monitors, previous = restore(report)
    arms = {body:InstalledArm(body) for body in ('stock','alternate')}
    for installed in arms.values():
        installed.arena.max_steps = CAP
    server = BounceArena(view='none')
    policy = ToolEquilibrium(BUNDLE,.0557)
    policy.mode = 'workspace_local'
    start = perf_counter()
    try:
        for stage in range(len(report['rows']),len(plans)):
            plan, physical_wiring = plans[stage], wires[STAGES[stage]]
            server.reset(seed=plan['launcher_seed'],split='held',render=False)
            release = [v.copy() for v in (server._ball_position,server._ball_velocity,server._ball_spin)]
            modes = {}
            for name in MODES:
                bank, monitor = banks[name], monitors[name]
                args = (arms[plan['body']],release,plan,physical_wiring,policy,bank,monitor,previous[name],'adaptive')
                played = trial(*args)
                replayed = trial(*args,probe_tape=played['inspection'],replay=played['actions'])
                for key in ('initial_q','after_setup_q','inspection','actions','metrics','events'):
                    if digest_json(played[key]) != digest_json(replayed[key]):
                        raise ValueError(f'{name} physical tape mismatch in {key} on drop {stage}')
                selected = bank.maps[bank.selected] if bank.selected is not None else I
                played['active_error'] = float(np.linalg.norm(selected-physical_wiring)/np.linalg.norm(physical_wiring))
                played['capped_release_s'] = played['metrics']['elapsed_s'] - (
                    len(played['inspection'])+plan['prep_ticks'])/60.
                previous[name] = played['index']
                modes[name] = played
            report['rows'].append(dict(plan=plan, interface=STAGES[stage], modes=modes))
            report['memory'] = {name:dict(maps=[m.tolist() for m in banks[name].maps], selected=banks[name].selected,
                                          pending=monitors[name].pending, previous=previous[name]) for name in MODES}
            _write(OUTPUT,report)
            print('integral course',stage+1,plan['body'],
                  [(name,row['metrics']['longest_consecutive_rebounds'],len(row['inspection']),
                    round(row['active_error'],3)) for name,row in modes.items()],flush=True)
        report['summary'] = aggregate(report['rows'], STAGES)
        report['wall_seconds_last_run'] = perf_counter()-start
        _write(OUTPUT,report)
        print(report['summary'],flush=True)
    finally:
        server.close()
        for a in arms.values():
            a.arena.close()


if __name__=='__main__':
    run()
