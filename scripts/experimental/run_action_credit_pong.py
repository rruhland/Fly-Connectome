"""Bounded K1 action-credit learning with exact shared-prefix simulator resets."""
import argparse,json,multiprocessing as mp,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
import numpy as np
from fly_connectome.control.pong_evaluation import FreshPongEvaluator,PongEvaluationConfig
from fly_connectome.control.codec import fingerprint
from pong_action_credit import clone_session,continue_session,legal_input,fit_preferences
import run_practical_pong_v2 as base
import run_preserved_rewarded_residual_pong as seeds

CONFIG=dict(work_seconds=600,workers=4,misses_per_phase=24,successes_per_phase=8,preterminal_leads=[5,10,15,20],
    intervention_cameras=1,ridge=.001,soft_anchor_weight=.25,maximum_legal_contrast_rms=.05,maximum_phase_regression=.03)


def pins():return {n:base.sha(Path(__file__).with_name(n)) for n in ('run_action_credit_pong.py','pong_action_credit.py','fast_pong_feature_rollout.py','run_practical_pong_v2.py')}


def stage(directory,source):
    directory=Path(directory);source=Path(source);m=base.read(source/'manifest.json')
    assert base.read(source/'completion-verification.json')['passed'];directory.mkdir(parents=True,exist_ok=True)
    assert not (directory/'manifest.json').exists()
    for n in ('episodes','credit'):(directory/n).mkdir()
    for src,dst in [('selected.npy','carry.npy'),('selected_theta.npy','carry_theta.npy'),('radial-cap.json','radial-cap.json'),
        ('old-transform.npy','old-transform.npy'),('extra-transform.npy','extra-transform.npy'),('search-transform.npy','search-transform.npy'),('feature-buffer.npz','feature-buffer.npz')]:
        (directory/dst).write_bytes((source/src).read_bytes())
    theta=np.load(directory/'carry_theta.npy');old=np.load(directory/'old-transform.npy');extra=np.load(directory/'extra-transform.npy')
    assert np.array_equal(theta[:,:214]@old+theta[:,214:]@extra,np.load(directory/'carry.npy'))
    hits={};receipts={}
    for path in (source/'episodes').glob('proposal_*.json'):
        r=base.read(path);assert r['complete'] and r['observations_exact'] and r['weights_sha256']==base.sha(directory/'carry.npy')
        assert base.config_key(r['config'])==base.config_key(base.job(directory,'c',r['seed'],'carry.npy',r['phase'],'train')['config'])
        hits[str(r['seed'])]=int(r['report']['hits']>0);receipts[str(r['seed'])]=str(path)
    bank=m['bank'];assert len(hits)==sum(map(len,bank.values()))==782
    rng=np.random.default_rng(3287932017);contexts=[]
    for p in ('left','right'):
        for hit,n in [(0,24),(1,8)]:
            candidates=[v for v in bank[p] if hits[str(v)]==hit];selected=rng.permutation(candidates)[:n];assert len(selected)==n
            for seed in selected:
                r=base.read(receipts[str(int(seed))]);contexts.append(dict(seed=int(seed),phase=p,hit=hit,transitions=r['transitions'],receipt=receipts[str(int(seed))]))
    schedule=seeds.fresh_seeds();schedule.update(bank=bank,contexts=contexts);base.atomic_json(directory/'schedules.json',schedule)
    manifest=dict(config=CONFIG,source=base.source_signature(),files=pins(),source_block=str(source),bank=bank,carry_hits=hits,
        data_hashes={n:base.sha(directory/n) for n in ('carry.npy','carry_theta.npy','old-transform.npy','extra-transform.npy','search-transform.npy','radial-cap.json','feature-buffer.npz')},
        assistance='exact full simulator/vision prefix reset access for TRAIN counterfactuals;policy sees unchanged visual features only;native returns supply pairwise preferences;no BP/teacher/action guards',
        accounting='baseline prefix/episode replay once per context;every actually executed alternative and null suffix counted, inherited prefix counters subtracted;clones charge wall only',
        boundary='K1 camera20ms then frozen greedy continuation;no burst, no ranking tied outcomes; no qualification2 TRAIN import')
    base.atomic_json(directory/'manifest.json',manifest);print(json.dumps(dict(staged=True,contexts=len(contexts),alternatives=512,new_cameras=0,pins=manifest['files'])),flush=True)


def credit_task(directory,c,deadline,parity):
    directory=Path(directory);began=time.monotonic();weights=np.load(directory/'carry.npy');matrix=np.load(directory/'search-transform.npy')
    j=base.job(directory,'credit',c['seed'],'carry.npy',c['phase'],'train');session=FreshPongEvaluator(weights,PongEvaluationConfig(**j['config']),first_hit_cutoff=True)
    indices=sorted(set(max(0,c['transitions']-v) for v in CONFIG['preterminal_leads']));snapshots={};records=[];states=[];branches=[]
    while not session.complete:
        if time.monotonic()>=deadline:raise TimeoutError('baseline deadline')
        if session.initial_cameras==session.episodes:session._begin_episode()
        index=session.transitions
        if index in indices:
            x=legal_input(session,matrix);action,q=session._choose_policy();assert np.max(abs(x@np.load(directory/'carry_theta.npy').T-q))<1e-10
            snapshots[index]=clone_session(session)
            states.append(dict(seed=c['seed'],phase=c['phase'],transition=index,x=x.tolist(),base_action=int(action),outcomes=[None,None,None],q=q.tolist()))
        records.append(session.step(check_frozen=False))
    session.verify_frozen();assert session.transitions==c['transitions'] and int(session.summary()['hits']>0)==c['hit']
    null=None
    if parity:
        i=indices[0];snapshot=snapshots[i];a=next(s['base_action'] for s in states if s['transition']==i)
        null=continue_session(snapshot,a,deadline,trace=True)
        expected=[dict(v) for v in records[i:]];actual=[dict(v) for v in null.pop('records')]
        actual[0].pop('intervention_action');actual[0].pop('unmodified_greedy_action')
        assert actual==expected and null['terminal_hash']==fingerprint(session),'null intervention must match every suffix record and final graph'
        null['parity_passed']=True
    partial_path=directory/'credit'/f'{c["phase"]}_{c["seed"]}.partial.json'
    def save_partial():
        base.atomic_json(partial_path,dict(complete=False,seed=c['seed'],phase=c['phase'],baseline_transitions=session.transitions,initial_cameras=session.initial_cameras,suffix_transition_cameras=sum(v['executed_suffix_cameras'] for v in branches)+(null['executed_suffix_cameras'] if null else 0),completed_branches=len(branches)))
    save_partial()
    for s in states:
        outcomes=s['outcomes'];outcomes[s['base_action']]=c['hit']
        for a in range(3):
            if a==s['base_action']:continue
            result=continue_session(snapshots[s['transition']],a,deadline)
            assert result['complete'];outcomes[a]=int(result['hit']);result.update(transition=s['transition']);branches.append(result);save_partial()
        assert all(v in (0,1) for v in outcomes)
    result=dict(complete=True,seed=c['seed'],phase=c['phase'],baseline_hit=c['hit'],baseline_transitions=session.transitions,initial_cameras=session.initial_cameras,
        suffix_transition_cameras=sum(v['executed_suffix_cameras'] for v in branches)+(null['executed_suffix_cameras'] if null else 0),
        total_transition_cameras=session.transitions+sum(v['executed_suffix_cameras'] for v in branches)+(null['executed_suffix_cameras'] if null else 0),
        branches=branches,states=states,null_parity=null,wall_seconds=time.monotonic()-began,observations_exact=True)
    base.atomic_json(directory/'credit'/f'{c["phase"]}_{c["seed"]}.json',result)
    partial_path.unlink()
    return result


def dispatch(directory,jobs,pool,deadline,cost):
    pending=list(jobs);active={};reports={}
    while pending or active:
        while pending and len(active)<4 and time.monotonic()<deadline-2:
            j=pending.pop(0);active[pool.submit(base.task,str(directory),j,deadline-2)]=j
        if not active:break
        done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
        for future in done:
            j=active.pop(future);r=future.result();reports[j['id']]=r
            cost[r['purpose']]+=r['transitions'];cost['initials']+=r['initial_cameras']
    return reports,len(reports)==len(jobs) and all(v['complete'] and v['observations_exact'] for v in reports.values())


def run(directory):
    directory=Path(directory);m=base.read(directory/'manifest.json');s=base.read(directory/'schedules.json')
    assert m['config']==CONFIG and m['files']==pins() and m['source']==base.source_signature()
    assert all(base.sha(directory/n)==v for n,v in m['data_hashes'].items());assert not (directory/'work-clock.json').exists()
    began=time.monotonic();base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began));deadline=began+CONFIG['work_seconds'];cost=dict(train=0,phase_probe=0,full_probe=0,initials=0)
    for src,dst in [('carry.npy','selected.npy'),('carry_theta.npy','selected_theta.npy')]:(directory/dst).write_bytes((directory/src).read_bytes())
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        pending=iter(enumerate(s['contexts'][1:],start=1));active={pool.submit(credit_task,str(directory),s['contexts'][0],deadline-150,True):s['contexts'][0]};results=[]
        def fill():
            while len(active)<4:
                try:i,c=next(pending)
                except StopIteration:break
                assert time.monotonic()<deadline-180,'not enough time for all causal contexts plus native bank'
                active[pool.submit(credit_task,str(directory),c,deadline-150,i==0)]=c
        while active:
            done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for future in done:
                c=active.pop(future);r=future.result();assert results or r['null_parity']['parity_passed'];results.append(r);cost['train']+=r['total_transition_cameras'];cost['initials']+=r['initial_cameras']
                base.atomic_json(directory/'progress.json',dict(completed_contexts=len(results),cost=cost));
            fill()
        states=[v for r in sorted(results,key=lambda v:(v['phase'],v['seed'])) for v in r['states']]
        outcomes={p:dict(contexts=sum(r['phase']==p for r in results),states=sum(v['phase']==p for v in states),
            outcome_mixed_states=sum(v['phase']==p and len(set(v['outcomes']))>1 for v in states),
            missed_contexts_repaired=len({v['seed'] for v in states if v['phase']==p and not next(r['baseline_hit'] for r in results if r['seed']==v['seed']) and any(v['outcomes'])}),
            success_contexts_spoiled=len({v['seed'] for v in states if v['phase']==p and next(r['baseline_hit'] for r in results if r['seed']==v['seed']) and not all(v['outcomes'])})) for p in ('left','right')}
        base.atomic_json(directory/'action-credit-report.json',dict(phases=outcomes,contexts=len(results),states=len(states),branches=sum(len(r['branches']) for r in results),null_parity=results[[r['null_parity'] is not None for r in results].index(True)]['null_parity'],cost=cost))
        theta=np.load(directory/'carry_theta.npy')
        with np.load(directory/'feature-buffer.npz') as b:delta,fit=fit_preferences(theta,states,b['X'],b['weights'])
        base.atomic_json(directory/'fit-report.json',fit);print(json.dumps(dict(action_credit=outcomes,fit=fit,cost=cost)),flush=True)
        selected=False;bank_result=None;probe={};proposal_hits={}
        if delta is not None and np.any(delta) and fit['applied']['repair_states_greedy_success']>0:
            proposal=theta+delta;old=np.load(directory/'old-transform.npy');extra=np.load(directory/'extra-transform.npy')
            weights=proposal[:,:214]@old+proposal[:,214:]@extra;base.save_array(directory/'proposal_theta.npy',proposal);base.save_array(directory/'proposal.npy',weights);base.save_array(directory/'applied_delta.npy',delta)
            jobs=base.bank_jobs(directory,'proposal',s['bank'],'proposal.npy');rr,ok=dispatch(directory,jobs,pool,deadline-45,cost)
            if ok:
                proposal_hits={str(v['seed']):int(v['report']['hits']>0) for v in rr.values()};score=base.phases(s['bank'],proposal_hits);before=base.phases(s['bank'],m['carry_hits'])
                selected=score['key'][0]>before['key'][0]+1e-15 and all(score['rates'][p]>=before['rates'][p]-.03-1e-15 for p in ('left','right'))
                bank_result=dict(before=before,proposal=score,selected=selected,gains_losses={p:dict(gains=sum(proposal_hits[str(v)]>m['carry_hits'][str(v)] for v in s['bank'][p]),losses=sum(proposal_hits[str(v)]<m['carry_hits'][str(v)] for v in s['bank'][p])) for p in s['bank']})
                if selected:
                    for src,dst in [('proposal.npy','selected.npy'),('proposal_theta.npy','selected_theta.npy')]:(directory/dst).write_bytes((directory/src).read_bytes())
        if selected:
            jobs=[]
            for p,values in s['probe'].items():
                for label in ('carry','selected'):
                    jobs.extend(base.job(directory,f'probe_{label}_{p}_{i:02d}',seed,label+'.npy',p,'phase_probe') for i,seed in enumerate(values))
            rr,ok=dispatch(directory,jobs,pool,deadline,cost)
            if ok:
                for p,values in s['probe'].items():
                    a=[int(rr[f'probe_carry_{p}_{i:02d}']['report']['hits']>0) for i in range(len(values))];b=[int(rr[f'probe_selected_{p}_{i:02d}']['report']['hits']>0) for i in range(len(values))]
                    probe[p]=dict(carry=a,selected=b,carry_hits=sum(a),selected_hits=sum(b),gains=sum(y>x for x,y in zip(a,b)),losses=sum(y<x for x,y in zip(a,b)))
        summary=dict(status='action_credit_complete',selected_proposal=selected,credit=outcomes,fit=fit,bank=bank_result,probe=probe,cost=cost,bank_skipped_reason=('no measured repair becomes greedy under applied fit' if delta is not None and bank_result is None else 'no strict outcome contrasts' if delta is None else None),
            total_wall_seconds=time.monotonic()-began,source_frozen=m['files']==pins() and m['source']==base.source_signature(),no_new_architecture=True,
            strict_native_preferences_only=True,reset_access_scaffold=True,selected_hash=base.sha(directory/'selected.npy'))
        base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['stage','run']);p.add_argument('directory');p.add_argument('--source',default='/workspace/m1b-experiments/pong-local-repair-20261025-r4');a=p.parse_args()
    if a.operation=='stage':stage(a.directory,a.source)
    else:run(a.directory)
