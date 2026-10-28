"""Reuse own reward-proven trajectories after checking exact action conflicts."""
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from run_rally_incremental_cycle import dataset
import run_practical_pong_v2 as base


def row_key(X,i):
    lo,hi=X.indptr[i:i+2]
    return X.indices[lo:hi].astype(np.int64).tobytes(),X.data[lo:hi].astype(np.float64).tobytes()


def conflicts(positive,guards):
    requested={};count=0
    for case in positive:
        X=case['X'].tocsr(copy=True);X.sum_duplicates();X.sort_indices()
        for i,action in enumerate(case['actions']):requested.setdefault(row_key(X,i),set()).add(int(action));count+=1
    inconsistent={k for k,actions in requested.items() if len(actions)>1};guard_conflicts=set()
    for case in guards:
        X=case['X'].tocsr(copy=True);X.sum_duplicates();X.sort_indices()
        for i,action in enumerate(case['actions']):
            k=row_key(X,i)
            if k in requested and requested[k]!={int(action)}:guard_conflicts.add(k)
    return dict(positive_rows=count,exact_positive_conflict_groups=len(inconsistent),exact_guard_conflict_groups=len(guard_conflicts),compatible=not inconsistent and not guard_conflicts)


def select_saved(sources,W,guards,excluded,used,source_signature):
    choices=[];screen=[]
    for source in map(Path,sources):
        manifest=base.read(source/'manifest.json');assert manifest['source']==source_signature
        summary=base.read(source/'summary.json');assert summary['source_frozen']
        for witness in summary['witnesses']:
            key=(str(source),witness['seed'],witness['path'])
            if key in used or witness['seed'] in excluded:continue
            p,_,m=dataset(source,W,[],[witness]);report=conflicts(p,guards+m);screen.append(dict(source=str(source),seed=witness['seed'],**report))
            if report['compatible'] and report['positive_rows']:
                choices.append((report['positive_rows'],witness['remaining_forced'],witness['seed'],str(source),witness))
            else:used.add(key)
    if not choices:return None,screen
    _,_,_,source,witness=min(choices,key=lambda x:x[:4]);used.add((source,witness['seed'],witness['path']))
    return (Path(source),witness),screen


def save_replay(directory,source,witness,current_head,bank_path):
    """No camera replay: retain the original rewarded trajectory's provenance."""
    directory=Path(directory);directory.mkdir();(directory/'witnesses').symlink_to(source/'witnesses',target_is_directory=True)
    original=base.read(source/'manifest.json')
    base.atomic_json(directory/'manifest.json',dict(source=base.source_signature(),current_head_sha256=base.sha(current_head),bank_sha256=base.sha(bank_path),reward_witness_source=str(source),reward_witness_source_head_sha256=original['current_head_sha256'],off_policy_own_reward_replay=True))
    base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=time.monotonic(),replayed_observations=0))
    result=dict(results=[],witnesses=[witness],selected_witness=witness,transition_cameras=0,initial_cameras=0,wall_seconds=0.,source_frozen=True,reward_witness_source=str(source),off_policy_own_reward_replay=True,new_observations=0)
    base.atomic_json(directory/'summary.json',result);return result
