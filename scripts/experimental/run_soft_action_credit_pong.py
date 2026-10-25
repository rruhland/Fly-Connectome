"""Same causal credit plus unit soft preservation of broad actual own-success states."""
import argparse,json,multiprocessing as mp,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import run_practical_pong_v2 as base
from run_action_credit_pong import dispatch
from pong_action_credit import fit_preferences


def run(directory,previous,success_source):
    directory=Path(directory);previous=Path(previous);success_source=Path(success_source);old=base.read(previous/'manifest.json');sm=base.read(success_source/'manifest.json')
    assert base.read(previous/'completion-verification.json')['passed'];directory.mkdir(parents=True,exist_ok=True)
    assert not (directory/'work-clock.json').exists();(directory/'episodes').mkdir()
    for src,dst in [('carry.npy','carry.npy'),('carry_theta.npy','carry_theta.npy'),('radial-cap.json','radial-cap.json')]:
        (directory/dst).write_bytes((previous/src).read_bytes())
    assert (directory/'carry.npy').read_bytes()==(success_source/'selected.npy').read_bytes()
    s=base.read(previous/'schedules.json');base.atomic_json(directory/'schedules.json',s)
    cases=[dict(c,source=str(success_source/c['feature_file'])) for c in sm['reused'] if c['kind']=='guard']
    for p in (success_source/'episodes').glob('*.json'):
        v=base.read(p)
        if v.get('capture_kind')=='guard':cases.append(dict(seed=v['seed'],phase=v['phase'],source=str(success_source/v['feature_file'])))
    assert len(cases)==33 and len({c['seed'] for c in cases})==33
    theta=np.load(directory/'carry_theta.npy');broad=[]
    for case in cases:
        with np.load(case['source']) as values:
            x=values['X'];actions=values['actions'];assert np.array_equal(np.argmax(x@theta.T,1),actions)
            broad.append(dict(seed=case['seed'],phase=case['phase'],X=x,actions=actions))
        case['sha256']=base.sha(case['source'])
    states=[]
    for p in (previous/'credit').glob('*.json'):states.extend(base.read(p)['states'])
    states.sort(key=lambda v:(v['phase'],v['seed'],v['transition']))
    files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_soft_action_credit_pong.py','run_action_credit_pong.py','pong_action_credit.py','run_practical_pong_v2.py','fast_pong_feature_rollout.py')}
    m=dict(source=base.source_signature(),files=files,source_block=str(previous),success_source=str(success_source),bank=s['bank'],carry_hits=old['carry_hits'],broad_success_cases=cases,
        work_seconds=600,ridge=.001,sampled_soft_anchor=.25,broad_soft_weight=1.,no_radial_shrink=True,no_hard_action_or_return_veto=True,
        same_native_credit_only=True,previous_raw_check='/workspace/m1b-experiments/pong-action-credit-rawcheck-20261025-r1',reserved_probe='same fixed wholly unused causal-trial probe64',
        data_hashes={n:base.sha(directory/n) for n in ('carry.npy','carry_theta.npy','radial-cap.json')})
    base.atomic_json(directory/'manifest.json',m);began=time.monotonic();base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began));deadline=began+600
    with np.load(previous/'feature-buffer.npz') as b:delta,fit=fit_preferences(theta,states,b['X'],b['weights'],maximum_rms=float('inf'),broad_preservation=broad,broad_weight=1.)
    # Infinite mathematical cap is encoded explicitly as absent, never nonfinite JSON.
    fit['maximum_rms']=None;base.atomic_json(directory/'fit-report.json',fit);print(json.dumps(dict(fit=fit)),flush=True)
    cost=dict(train=0,phase_probe=0,full_probe=0,initials=0);accepted=False;bank_result=None;probe={};bank_complete=None;fresh_complete=None
    for src,dst in [('carry.npy','selected.npy'),('carry_theta.npy','selected_theta.npy')]:(directory/dst).write_bytes((directory/src).read_bytes())
    if fit['applied']['repair_states_greedy_success']>0:
        proposal=theta+delta;oldT=np.load(previous/'old-transform.npy');extra=np.load(previous/'extra-transform.npy')
        base.save_array(directory/'proposal_theta.npy',proposal);base.save_array(directory/'proposal.npy',proposal[:,:214]@oldT+proposal[:,214:]@extra);base.save_array(directory/'applied_delta.npy',delta)
        with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
            rr,bank_complete=dispatch(directory,base.bank_jobs(directory,'proposal',s['bank'],'proposal.npy'),pool,deadline-60,cost)
            if bank_complete:
                hits={str(r['seed']):int(r['report']['hits']>0) for r in rr.values()};before=base.phases(s['bank'],m['carry_hits']);score=base.phases(s['bank'],hits)
                accepted=score['key'][0]>before['key'][0]+1e-15 and all(score['rates'][p]>=before['rates'][p]-.03-1e-15 for p in ('left','right'))
                rescue=[r for r in (base.read(p) for p in (previous/'credit').glob('*.json')) if not r['baseline_hit'] and any(any(v['outcomes']) for v in r['states'])]
                bank_result=dict(before=before,proposal=score,selected=accepted,hits=hits,causal_repairs_retained=[dict(seed=r['seed'],phase=r['phase'],hit=hits[str(r['seed'])]) for r in rescue],gains_losses={p:dict(gains=sum(hits[str(v)]>m['carry_hits'][str(v)] for v in s['bank'][p]),losses=sum(hits[str(v)]<m['carry_hits'][str(v)] for v in s['bank'][p])) for p in s['bank']})
                base.atomic_json(directory/'bank-result.json',bank_result);print(json.dumps(dict(bank=score,selected=accepted,repairs=bank_result['causal_repairs_retained'],cost=cost)),flush=True)
                if accepted:
                    for src,dst in [('proposal.npy','selected.npy'),('proposal_theta.npy','selected_theta.npy')]:(directory/dst).write_bytes((directory/src).read_bytes())
                    jobs=[]
                    for p,values in s['probe'].items():
                        for label in ('carry','selected'):jobs.extend(base.job(directory,f'probe_{label}_{p}_{i:02d}',seed,label+'.npy',p,'phase_probe') for i,seed in enumerate(values))
                    rr,fresh_complete=dispatch(directory,jobs,pool,deadline,cost)
                    if fresh_complete:
                        for p,values in s['probe'].items():
                            a=[int(rr[f'probe_carry_{p}_{i:02d}']['report']['hits']>0) for i in range(32)];b=[int(rr[f'probe_selected_{p}_{i:02d}']['report']['hits']>0) for i in range(32)]
                            probe[p]=dict(carry=a,selected=b,carry_hits=sum(a),selected_hits=sum(b),gains=sum(y>x for x,y in zip(a,b)),losses=sum(y<x for x,y in zip(a,b)))
    summary=dict(status='bounded_partial' if bank_complete is False or fresh_complete is False else 'soft_action_credit_checked',selected_proposal=accepted,fit=fit,bank_complete=bank_complete,bank=bank_result,probe=probe,cost=cost,
        bank_skipped_reason='broader preservation removed all sampled repair choices' if bank_complete is None else None,source_frozen=m['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files},
        total_wall_seconds=time.monotonic()-began,selected_hash=base.sha(directory/'selected.npy'))
    base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--previous',default='/workspace/m1b-experiments/pong-action-credit-20261025-r1');p.add_argument('--success-source',default='/workspace/m1b-experiments/pong-local-repair-20261025-r4');a=p.parse_args();run(a.directory,a.previous,a.success_source)
