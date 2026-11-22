"""Play a frozen native hold/direction selector on new simulated arm openings."""

import argparse
from time import perf_counter

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt22_duration_transfer import MACRO
from .tt22_motion_prefix import prefix
from .tt22_motor_transfer import MotorPrimitive
from .tt23_abstain import SOURCE, choose, preferences
from .tt23_anonymous_target import target_context
from .tt23_crossflight import CrossFlightArena, options, schedule, SOURCE_ARCHIVES
from .tt23_direction import CHOICES, SOURCE as ORDINAL, paired_returns, pick
from .tt23_optical_contact import OUT


SEED = 1543662170
PLAN = OUT/'abstain-plan.json.gz'
BANK = OUT/'abstain-arm.json.gz'
RESULT = OUT/'abstain-results.json.gz'
MODES = ('learned','shuffled','center','ordinal')


def freeze():
    starts = schedule(SEED)
    seeds = {row['seed'] for row in starts}
    used = {row['seed'] for name in (*SOURCE_ARCHIVES,'crossflight-arm.json.gz') for row in _read(OUT/name)['rows']}
    if len(seeds)!=len(starts) or seeds&used:
        raise ValueError('the new physical course overlaps earlier openings')
    plan = dict(schema='tt23-abstain-plan-1',seed=SEED,roster=starts,source_sha256=_hash(SOURCE),
                ordinal_sha256=_hash(ORDINAL),macro_sha256=_hash(MACRO),local_steps=list(CHOICES))
    if PLAN.exists() and plan!=_read(PLAN):
        raise ValueError('the frozen data, model, or target schedule changed')
    _write(PLAN,plan)
    return plan


def policies(context,models,ordinal):
    seen = np.array([context is not None])
    x = np.zeros((1,4)) if context is None else np.atleast_2d(context)
    decisions = {mode:int(choose(model,x,seen)[0]) for mode,model in models.items()}
    scores = {mode:preferences(model,x)[0].tolist() if seen[0] else None for mode,model in models.items()}
    decisions.update(center=3,ordinal=pick(context,ordinal,'learned'))
    return decisions,scores


def collect():
    plan = freeze()
    models = _read(SOURCE)['models']
    ordinal = _read(ORDINAL)['pooled']
    macro = _read(MACRO)['parameters']
    ready = MotorPrimitive(np.asarray(macro),None).ready_world
    metadata = dict(schema='tt23-abstain-arm-1',plan_sha256=_hash(PLAN))
    receipt = _read(BANK) if BANK.exists() else dict(metadata=metadata,rows=[])
    if receipt['metadata']!=metadata:
        raise ValueError('the saved physical options came from a different frozen plan')
    rows = receipt['rows']
    for row,expected in zip(rows,plan['roster'][:len(rows)],strict=True):
        if any(row[key]!=value for key,value in expected.items()):
            raise ValueError('the replayed physical schedule changed')
    arena = CrossFlightArena()
    began = perf_counter()
    try:
        for start in plan['roster'][len(rows):]:
            arena.release = start['start_x_draw'],start['throw_offset_draw']
            images = prefix(arena,start['seed'],0.)
            observed = target_context(images,ready)
            choices,advantages = policies(observed,models,ordinal)
            initial,snapshot = arena.initial,arena.snapshot
            outcomes = options(arena,start,macro,ready,observed is not None,snapshot,initial)
            rows.append(dict(start,sight=images,initial=initial,prefix=snapshot,observed=observed is not None,
                             options=outcomes,choices=choices,advantages=advantages))
            if len(rows)%8==0:
                _write(BANK,receipt)
                print('physical abstain',len(rows),round(perf_counter()-began,1),flush=True)
    finally:
        arena.close()
    receipt['verified'] = len(rows)*len(CHOICES)
    receipt['wall_seconds'] = perf_counter()-began
    _write(BANK,receipt)
    return receipt


def group(rows):
    stats = {}
    legal = {}
    for mode in MODES:
        trials = [row['options'][row['choices'][mode]] for row in rows]
        legal[mode] = [trial['legal']>0 for trial in trials]
        stats[mode] = dict(contacts=sum(trial['contacts']>0 for trial in trials),legal=sum(legal[mode]),
                           holds=sum(row['choices'][mode]==3 for row in rows))
    return dict(points=len(rows),detected=sum(row['observed'] for row in rows),scores=stats,
                paired={mode:paired_returns(legal['learned'],legal[mode]) for mode in MODES[1:]})


def assess():
    bank = _read(BANK)
    plan = freeze()
    rows = bank['rows']
    if bank['metadata']['plan_sha256']!=_hash(PLAN) or bank['verified']!=len(plan['roster'])*len(CHOICES):
        raise ValueError('score only the entire verified course')
    results = {key:group(rows if key=='all' else [r for r in rows if r['family']==key])
               for key in ('all','same','opposite')}
    gate = all(scores['scores']['learned']['legal']>=2+max(scores['scores'][mode]['legal'] for mode in MODES[1:])
               for name,scores in results.items() if name!='all')
    report = dict(schema='tt23-abstain-results-1',bank_sha256=_hash(BANK),plan_sha256=_hash(PLAN),
                  source_sha256=_hash(SOURCE),groups=results,gate=gate)
    _write(RESULT,report)
    print('abstain arm', {key:value['scores'] for key,value in results.items()},'full-point gate',gate,flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assess-only',action='store_true')
    args = parser.parse_args()
    if not args.assess_only:
        collect()
    assess()
