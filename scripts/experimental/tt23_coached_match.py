"""Compare three optical calibration conditions in full games with the learned coach."""

import argparse
from hashlib import sha256
import json
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np

from .tt4_arena import mujoco
from .tt4_transfer_trial import _hash, _read, _write
from .tt8_service_arena import ServiceScore
from .tt22_far_cache import FarCache
from .tt22_match import CHECKPOINT, FACTORY, STANDARD_PICK
from .tt22_opponent import RallyArena
from .tt22_ordinary_game import DIR, FrozenMemory, StandardMatch
from .tt22_rally import ReturnMemory
from .tt22_rally_trial import CONFIG, alternating
from .tt23_coach import OUTPUT, PRACTICE
from .tt23_coach_trial import RECEIPT as COACH_HELD, ReplyRanking, summarize_points


SEED = 1839619077
ATTEMPTS = 90
FIRST_SERVER = -1
TARGET = 11
MODES = ('unpracticed','frozen','online')
PLAN = OUTPUT/'coached-match-plan.json.gz'
RECEIPT = OUTPUT/'coached-match.json.gz'


def game():
    return ServiceScore(target=TARGET,first_server=FIRST_SERVER)


def freeze(path):
    source,coach = _read(CHECKPOINT),_read(PRACTICE)
    if len(source['evidence'])!=7 or len(source['practice'])!=4 or not coach['live_gate']:
        raise ValueError('expected seven optical practice observations and the frozen accepted coach')
    starts = np.random.default_rng(SEED).integers(2**31,size=ATTEMPTS).tolist()
    prior = {row['seed'] for row in source['practice']+coach['rows']}
    prior.update(_read(COACH_HELD)['metadata']['seeds'])
    for name,key in (('ordinary-game.json.gz','branches'),('after-service.json.gz','rows')):
        prior.update(row['seed'] for branch in _read(DIR/name)[key].values() for row in branch)
    prior.update(case['seed'] for case in _read(OUTPUT/'tt23_sequence_coach.json.gz')['cases'])
    if len(set(starts))!=len(starts) or prior.intersection(starts):
        raise ValueError('the match draw repeats an earlier launch')
    metadata = dict(schema='tt23-coached-match-plan-1',seed=SEED,starts=starts,first_server=FIRST_SERVER,
                    target=TARGET,attempt_allowance=ATTEMPTS,arena=CONFIG,profile=coach['profile'],
                    coach_sha256=_hash(PRACTICE),checkpoint_sha256=_hash(CHECKPOINT),
                    service_sha256=_hash(FACTORY),service_program=STANDARD_PICK,modes=list(MODES))
    if path.exists():
        if _read(path)!=metadata:
            raise ValueError('the new match setup differs from the frozen plan')
    else:
        _write(path,metadata)
    return metadata


def memory_for(mode,evidence):
    learner = ReturnMemory if mode=='online' else FrozenMemory
    return learner(() if mode=='unpracticed' else evidence,guard_empty=True)


def restore(mode,rows,starts,evidence):
    score = game()
    memory = memory_for(mode,evidence)
    for i,row in enumerate(rows):
        if (score.finished or (row['seed'],row['role'],row['score_before'])
                !=(starts[i],score.server,list(score.score))):
            raise ValueError('game continued after completion or changed its launch/service schedule')
        context = row['learning']
        if (context['before'],context['bias_before'])!=(len(memory.evidence),memory.bias().tolist()):
            raise ValueError('the point began with the wrong optical calibration')
        for shot in row['shots']:
            if 'optical_outcome' in shot:
                memory.observe(shot,shot['optical_outcome'])
        if (context['after'],context['bias_after'])!=(len(memory.evidence),memory.bias().tolist()):
            raise ValueError('the issued strokes did not regenerate the optical memory')
        if list(score.record(row['outcome']))!=row['score_after']:
            raise ValueError('the strict outcome changed the scoreboard')
    return score,memory


def verify(branches,starts,evidence):
    """Replay official game timing and both complete motor tapes on the five-axis body."""
    arena = RallyArena(**CONFIG)
    flag = mujoco.mjtState.mjSTATE_INTEGRATION
    initial = np.empty(mujoco.mj_stateSize(arena.model,flag))
    result = {}
    try:
        for mode,rows in branches.items():
            score,_ = restore(mode,rows,starts,evidence)
            if not score.finished:
                raise ValueError(f'incomplete regulation game in {mode}')
            records = []
            for row in rows:
                arena.reset(seed=row['seed'],server=row['role'],render=False)
                mujoco.mj_getState(arena.model,arena.data,initial,flag)
                records.append(dict(seed=row['seed'],role=row['role'],sha256=sha256(initial.tobytes()).hexdigest()))
                saved_aim,saved_control = arena._aim,arena._command_to
                try:
                    for tick,(near,far) in enumerate(zip(row['actions'],row['rival_actions'],strict=True),1):
                        arena._aim = arena._command_to = lambda *args: np.asarray(far)
                        _,reward,done,cap,_ = arena.step(near,render=False)
                        if bool(done or cap)!=(tick==len(row['actions'])):
                            raise ValueError('physical replay ended on a different frame')
                finally:
                    arena._aim,arena._command_to = saved_aim,saved_control
                if (row['outcome']!=arena.point_result() or row['reward']!=reward
                        or json.dumps(row['events'])!=json.dumps(arena.events)
                        or row['longest']!=alternating(arena.events)):
                    raise ValueError(f"physical replay changed match point {row['seed']}")
            result[mode] = records
    finally:
        arena.close()
    return result


def comparison(branches,verified):
    modes = {mode:{(r['seed'],r['role']):(i,r,proof['sha256']) for i,(r,proof) in
                  enumerate(zip(rows,verified[mode],strict=True),1)} for mode,rows in branches.items()}
    results = {}
    for left,right in combinations(MODES,2):
        a,b = modes[left],modes[right]
        shared = [key for key in a if key in b and a[key][2]==b[key][2]]
        unmatched = {mode:[dict(seed=seed,server=role,attempt=details[(seed,role)][0]) for seed,role in details
                           if (seed,role) not in shared] for mode,details in ((left,a),(right,b))}
        rows = [dict(seed=key[0],server=key[1],attempt=a[key][0],lengths=[a[key][1]['longest'],b[key][1]['longest']])
                for key in shared]
        summaries = {mode:summarize_points([details[key][1] for key in shared])
                     for mode,details in ((left,a),(right,b))} if shared else {}
        results[f'{left}-{right}'] = dict(common=len(shared),pairs=rows,summaries=summaries,unmatched=unmatched)
    return results


def complete(data,plan,evidence,started):
    rows = data['branches']
    verified = verify(rows,plan['starts'],evidence)
    data['verified'] = {mode:len(records) for mode,records in verified.items()}
    data['physical_openings'] = verified
    if set(rows)==set(MODES):
        data['common_openings'] = comparison(rows,verified)
    data['summary'] = {}
    for mode,branch in rows.items():
        score,memory = restore(mode,branch,plan['starts'],evidence)
        extra = len(memory.evidence)-(0 if mode=='unpracticed' else len(evidence))
        data['summary'][mode] = dict(summarize_points(branch),final_score=list(score.score),new_optical_examples=extra)
    data['wall_seconds'] = sum(row['wall_seconds'] for branch in rows.values() for row in branch)
    data['finishing_session_seconds'] = perf_counter()-started
    return data


def run(plan_path,output,*,branch=None,resume_from=None):
    plan = freeze(plan_path)
    check,coach = _read(CHECKPOINT),_read(PRACTICE)
    configuration = dict(schema='tt23-coached-match-1',plan_sha256=_hash(plan_path),initial_evidence=check['evidence'])
    modes = MODES if branch is None else (branch,)
    if output.exists():
        data = _read(output)
    else:
        previous = _read(resume_from) if resume_from else dict(**configuration,branches={mode:[] for mode in modes})
        if any(previous.get(key)!=value for key,value in configuration.items()):
            raise ValueError('the resume source uses a different game schedule or fit')
        data = dict(**configuration,branches={mode:previous['branches'][mode] for mode in modes})
    if any(data.get(key)!=value for key,value in configuration.items()) or set(data['branches'])!=set(modes):
        raise ValueError('cannot resume an incompatible match configuration')
    rows = data['branches']
    resumed = {mode:restore(mode,branch,plan['starts'],check['evidence']) for mode,branch in rows.items()}
    started = perf_counter()
    if any(not score.finished for score,_ in resumed.values()):
        station = StandardMatch(plan['profile'])
        station.rival.confirm = ReplyRanking(station.rival,coach).confirm
        cache = FarCache(station.rival)
        try:
            for i,seed in enumerate(plan['starts']):
                for mode in modes[i%len(modes):]+modes[:i%len(modes)]:
                    score,memory = resumed[mode]
                    if score.finished or len(rows[mode])>i:
                        continue
                    before = list(score.score)
                    point_start = perf_counter()
                    row = station.play(seed,score.server,memory)
                    after = list(score.record(row['outcome']))
                    rows[mode].append(dict(row,score_before=before,score_after=after,
                                           wall_seconds=perf_counter()-point_start))
                    _write(output,data)
                    print('coached game',mode,i+1,row['role'],after,row['longest'],
                          row['outcome']['legal_returns'],row['outcome']['rival_returns'],flush=True)
                if all(score.finished for score,_ in resumed.values()):
                    break
        finally:
            station.close()
        data['opponent_cache'] = dict(hits=cache.hits,misses=cache.misses,audited=cache.checked)
    complete(data,plan,check['evidence'],started)
    _write(output,data)
    print(data['summary'],flush=True)


def merge(plan_path,output,parts):
    started = perf_counter()
    plan = freeze(plan_path)
    evidence = _read(CHECKPOINT)['evidence']
    data = dict(schema='tt23-coached-match-1',plan_sha256=_hash(plan_path),initial_evidence=evidence,
                branches={},worker_receipts={part.name:_hash(part) for part in parts})
    for part in parts:
        receiver = _read(part)
        if receiver['plan_sha256']!=data['plan_sha256'] or receiver['initial_evidence']!=evidence:
            raise ValueError('a worker was given a different frozen plan or observation checkpoint')
        for mode,branch in receiver['branches'].items():
            if mode in data['branches'] or receiver['verified'][mode]!=len(branch):
                raise ValueError('a condition is duplicated or its game did not verify')
            data['branches'][mode] = branch
    if set(data['branches'])!=set(MODES):
        raise ValueError('the complete result needs exactly the three registered conditions')
    complete(data,plan,evidence,started)
    _write(output,data)
    print(data['summary'],flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan',type=Path,default=PLAN)
    parser.add_argument('--output',type=Path,default=RECEIPT)
    work = parser.add_mutually_exclusive_group()
    work.add_argument('--freeze-only',action='store_true')
    work.add_argument('--branch',choices=MODES)
    work.add_argument('--merge',type=Path,nargs=3)
    parser.add_argument('--resume-from',type=Path)
    args = parser.parse_args()
    if args.freeze_only:
        plan = freeze(args.plan)
        print('frozen match',plan['seed'],plan['starts'][:12],flush=True)
    elif args.merge:
        merge(args.plan,args.output,args.merge)
    else:
        run(args.plan,args.output,branch=args.branch,resume_from=args.resume_from)
