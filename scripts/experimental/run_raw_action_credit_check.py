"""One native reward test of the already-fitted, unscaled causal-credit head.

A new explicitly authorized allocation, after the capped trial exited. No new
labels, fit, dose grid, or mutation of its immutable result.
"""
import argparse,json,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import run_practical_pong_v2 as base
from run_action_credit_pong import dispatch


def run(directory,previous):
    directory=Path(directory);previous=Path(previous);old=base.read(previous/'manifest.json');sr=base.read(previous/'summary.json')
    assert base.read(previous/'completion-verification.json')['passed'] and not sr['selected_proposal'] and sr['fit']['raw']['repair_states_greedy_success']==5 and sr['fit']['applied']['repair_states_greedy_success']==0
    directory.mkdir(parents=True,exist_ok=True);assert not (directory/'work-clock.json').exists();(directory/'episodes').mkdir()
    for src,dst in [('carry.npy','carry.npy'),('carry_theta.npy','carry_theta.npy'),('derived_raw_proposal.npy','proposal.npy'),('derived_raw_theta.npy','proposal_theta.npy'),('radial-cap.json','radial-cap.json')]:
        (directory/dst).write_bytes((previous/src).read_bytes())
    for src,dst in [('carry.npy','selected.npy'),('carry_theta.npy','selected_theta.npy')]:(directory/dst).write_bytes((directory/src).read_bytes())
    s=base.read(previous/'schedules.json');base.atomic_json(directory/'schedules.json',s)
    files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_raw_action_credit_check.py','run_action_credit_pong.py','pong_action_credit.py','run_practical_pong_v2.py','fast_pong_feature_rollout.py')}
    manifest=dict(source=base.source_signature(),files=files,source_block=str(previous),bank=s['bank'],carry_hits=old['carry_hits'],work_seconds=600,
        raw_legal_contrast_rms=sr['fit']['raw_legal_contrast_rms'],no_refit=True,raw_fit_verified=True,inherited_credit_cost=sr['cost'],
        reason='explicit parent-authorized native reward test of same raw ridge optimum after the .05 radial cap removed every observed repair',
        reserved_probe='same previously fixed and wholly unused action-credit probe64; now assigned to this explicit follow-on check',
        data_hashes={n:base.sha(directory/n) for n in ('carry.npy','carry_theta.npy','proposal.npy','proposal_theta.npy','radial-cap.json')})
    base.atomic_json(directory/'manifest.json',manifest)
    began=time.monotonic();base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began));deadline=began+600
    cost=dict(train=0,phase_probe=0,full_probe=0,initials=0);accepted=False;bank_result=None;probe={};fresh_complete=True
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        jobs=base.bank_jobs(directory,'proposal',s['bank'],'proposal.npy');rr,bank_complete=dispatch(directory,jobs,pool,deadline-60,cost)
        if bank_complete:
            hits={str(r['seed']):int(r['report']['hits']>0) for r in rr.values()};before=base.phases(s['bank'],manifest['carry_hits']);score=base.phases(s['bank'],hits)
            accepted=score['key'][0]>before['key'][0]+1e-15 and all(score['rates'][p]>=before['rates'][p]-.03-1e-15 for p in ('left','right'))
            bank_result=dict(before=before,proposal=score,selected=accepted,hits=hits,gains_losses={p:dict(gains=sum(hits[str(v)]>manifest['carry_hits'][str(v)] for v in s['bank'][p]),losses=sum(hits[str(v)]<manifest['carry_hits'][str(v)] for v in s['bank'][p])) for p in s['bank']})
            base.atomic_json(directory/'bank-result.json',bank_result);print(json.dumps(dict(bank=bank_result['proposal'],selected=accepted,gains_losses=bank_result['gains_losses'],cost=cost)),flush=True)
            if accepted:
                for src,dst in [('proposal.npy','selected.npy'),('proposal_theta.npy','selected_theta.npy')]:(directory/dst).write_bytes((directory/src).read_bytes())
                jobs=[]
                for p,values in s['probe'].items():
                    for label in ('carry','selected'):
                        jobs.extend(base.job(directory,f'probe_{label}_{p}_{i:02d}',seed,label+'.npy',p,'phase_probe') for i,seed in enumerate(values))
                rr,fresh_complete=dispatch(directory,jobs,pool,deadline,cost)
                if fresh_complete:
                    for p,values in s['probe'].items():
                        a=[int(rr[f'probe_carry_{p}_{i:02d}']['report']['hits']>0) for i in range(len(values))];b=[int(rr[f'probe_selected_{p}_{i:02d}']['report']['hits']>0) for i in range(len(values))]
                        probe[p]=dict(carry=a,selected=b,carry_hits=sum(a),selected_hits=sum(b),gains=sum(y>x for x,y in zip(a,b)),losses=sum(y<x for x,y in zip(a,b)))
    source_frozen=manifest['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files}
    summary=dict(status='raw_action_credit_checked' if bank_complete and fresh_complete else 'bounded_partial',selected_proposal=accepted,bank_complete=bank_complete,
        fresh_requested=accepted,fresh_complete=fresh_complete if accepted else None,bank=bank_result,probe=probe,cost=cost,total_wall_seconds=time.monotonic()-began,
        source_frozen=source_frozen,raw_ridge_fit_only=True,selected_hash=base.sha(directory/'selected.npy'),previous_credit_root=str(previous))
    base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--previous',default='/workspace/m1b-experiments/pong-action-credit-20261025-r1');a=p.parse_args();run(a.directory,a.previous)
