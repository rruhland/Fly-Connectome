"""One fixed covariance-preserving target update and actual TRAIN validation."""
import argparse,json,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from pong_covariance_trace_merge import fit
from run_rally_earned_merge import load_case
from run_rally_fixed_bank_check import run as native_check
import run_practical_pong_v2 as base


def run(directory,index_path,covariance_root,bank_path,sentinel_schedule,reference_root,hard_seeds=(),guard_source=None,protect_first_returns=False):
    directory=Path(directory);index_path=Path(index_path);covariance_root=Path(covariance_root);bank_path=Path(bank_path);reference_root=Path(reference_root);sentinel_schedule=Path(sentinel_schedule);assert not directory.exists();directory.mkdir()
    index=base.read(index_path);bank=base.read(bank_path);covariance=base.read(covariance_root/'summary.json');assert Path(covariance['source_index'])==index_path;assert index['targets']==[104562292];assert index['reference_sha256']==bank['current_head_sha256']==base.sha(index['reference_head'])
    targets=[e for e in index['cases'] if e['seed'] in index['targets']];assert len(targets)==1;hard_entries=targets+[e for e in index['cases'] if e['seed'] in hard_seeds];assert len(hard_entries)==1+len(set(hard_seeds));first_cases=0
    if protect_first_returns:
        assert not hard_seeds and guard_source is None
        for entry in bank['bank']:
            receipt=base.read(Path(entry['receipt']));hits=[i for i,r in enumerate(receipt['records']) if r['hits']]
            if not hits:continue
            hard_entries.append(dict(seed=entry['seed'],receipt=entry['receipt'],raw_feature=entry['raw_feature'],end=hits[0]+1,source_head=index['reference_head'],source_head_sha256=index['reference_sha256']));first_cases+=1
        assert first_cases==bank['current_native_aggregate']['first_incoming_returns']
    cases=[load_case(e) for e in hard_entries];base.atomic_json(directory/'hard-cases.json',dict(cases=hard_entries,first_return_prefixes=first_cases,raw_rows=sum(e['end'] for e in hard_entries),known_TRAIN_only=True));others=[e for e in index['cases'] if e['seed'] not in index['targets']];assert len(others)==covariance['episodes'] and sum(e['end'] for e in others)==covariance['rows']
    if hard_seeds:
        assert guard_source is not None;guard_source=Path(guard_source);outcome=base.read(guard_source/'summary.json');lost=[c for c in outcome['per_seed_hit_changes'] if c['after_hits']<c['before_hits']];selected=sorted(lost,key=lambda c:(c['after_hits']-c['before_hits'],c['seed']))[:3];assert set(hard_seeds)=={c['seed'] for c in selected}
    scale=np.load(covariance_root/'scale.npy');assert np.array_equal(scale,np.load(index['feature_metric']));L=np.load(covariance_root/'cholesky.npy');reference=np.load(index['reference_head']);normal=base.read(base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json')
    schedule=base.read(sentinel_schedule);seeds=schedule['seeds'];assert len(seeds)==len(set(seeds))==32 and all(s in seeds for s in index['targets']);base.atomic_json(directory/'sentinel-schedule.json',schedule)
    files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_covariance_merge.py','pong_covariance_trace_merge.py','run_rally_earned_merge.py','run_rally_fixed_bank_check.py')};manifest=dict(source=base.source_signature(),files=files,index=str(index_path),index_sha256=base.sha(index_path),bank=str(bank_path),bank_sha256=base.sha(bank_path),reference_root=str(reference_root),reference_sha256=index['reference_sha256'],covariance_root=str(covariance_root),covariance=covariance,covariance_cholesky_sha256=base.sha(covariance_root/'cholesky.npy'),covariance_scale_sha256=base.sha(covariance_root/'scale.npy'),seconds=60,hard_target_only=not bool(hard_seeds or protect_first_returns),adaptive_TRAIN_guard_seeds=list(hard_seeds),adaptive_TRAIN_guard_source=str(guard_source) if guard_source else None,stage_specific_first_return_prefixes=first_cases,all_old_actions_hard_preserved=False,no_fresh_admission=True);base.atomic_json(directory/'manifest.json',manifest)
    began=time.monotonic();base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=began+60));candidate,report=fit(reference,cases,scale,L,normal['kappa'],began+60);base.atomic_json(directory/'merge-report.json',report)
    if candidate is None:base.atomic_json(directory/'summary.json',dict(status='no_verified_target_merge',merge=report,new_observations=0));print(json.dumps(report),flush=True);return
    base.save_array(directory/'proposal.npy',candidate);delta=candidate-reference;empirical=[];old_energy=[];action_changes=0;rows=0
    for entry in others:
        X=sp.load_npz(entry['raw_feature'])[:entry['end']];change=X@delta.T;old=X@reference.T;empirical.append(float(np.mean(np.sum(change*change,axis=1))));old_energy.append(float(np.mean(np.sum(old*old,axis=1))));action_changes+=int(np.count_nonzero(np.argmax(old+change,axis=1)!=np.argmax(old,axis=1)));rows+=X.shape[0]
    metric=dict(equal_episode_mean_squared_score_change=float(np.mean(empirical)),equal_episode_reference_score_energy=float(np.mean(old_energy)),changed_cached_greedy_rows=action_changes,cached_rows=rows,episodes=len(others),changed_choices_are_not_actual_return_losses=True,covariance_objective=report['whitened_parameter_cost'],covariance_empirical_component=report['whitened_parameter_cost']-covariance['ridge_fraction']*report['scaled_coefficient_norm2']);base.atomic_json(directory/'empirical-score-preservation.json',metric)
    entries=[e for s in seeds for e in bank['bank'] if e['seed']==s];assert len(entries)==32;sentinel_bank=dict(bank,bank=entries);base.atomic_json(directory/'sentinel-bank.json',sentinel_bank);native_check(directory/'sentinel',directory/'sentinel-bank.json',directory/'proposal.npy',native_seconds=120,alternate_native_root=reference_root);s=base.read(directory/'sentinel/summary.json');advance=s['status']=='complete' and s['candidate']['own_returns']>s['baseline']['own_returns'];summary=dict(status='sentinel_complete' if s['status']=='complete' else 'sentinel_partial',merge=report,empirical_score_preservation=metric,sentinel=s,advance_to_full=advance,rule='strict actual net return gain on fixed TRAIN sentinel; first/later and individual tradeoffs reported',wall_seconds=time.monotonic()-began);base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)
    if advance:native_check(directory/'full',bank_path,directory/'proposal.npy',cached_native_root=directory/'sentinel',fresh_root=str(directory/'fresh'),practical_later_screen=True,alternate_native_root=reference_root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--index',required=True);p.add_argument('--covariance-root',required=True);p.add_argument('--bank',required=True);p.add_argument('--sentinel-schedule',required=True);p.add_argument('--reference-root',required=True);p.add_argument('--hard-seed',type=int,action='append',default=[]);p.add_argument('--guard-source');p.add_argument('--protect-first-returns',action='store_true');a=p.parse_args();run(a.directory,a.index,a.covariance_root,a.bank,a.sentinel_schedule,a.reference_root,a.hard_seed,a.guard_source,a.protect_first_returns)
