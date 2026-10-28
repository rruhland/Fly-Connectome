"""Eight identical-prefix action comparisons; no physics/diagnostic inputs."""
import argparse,json,time,multiprocessing as mp,random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from fly_connectome.control.policy import FrozenLinearPolicy
from fly_connectome.control.pong_evaluation import FreshPongEvaluator
from fly_connectome.control.scaling import policy_features
from pong_action_credit import clone_session,step_action
from rally_raw_capture import start,SparseRows
import run_practical_pong_v2 as base


def choose(bank,candidate,excluded=(),opening_repair=False,prioritize_openings=False):
    old={e['seed']:e for e in bank['bank']};cases=[]
    for p in sorted((candidate/'native_candidate/episodes').glob('*.json')):
        c=base.read(p);e=old[c['seed']];d=base.read(Path(e['receipt']));dh=d['report']['hits'];ch=c['report']['hits']
        if c['seed'] in excluded:continue
        if ch>=dh:continue
        t=next(i for i,(a,b) in enumerate(zip(d['records'],c['records'])) if a['action']!=b['action']);assert t<=max(i for i,r in enumerate(d['records']) if r['hits'])
        f=Path(c['feature_file']);f=f if f.is_absolute() else candidate/f
        cases.append(dict(seed=c['seed'],phase='left' if base.leftward(c['seed']) else 'right',old_receipt=e['receipt'],old_feature=e['raw_feature'],candidate_receipt=str(p),candidate_feature=str(f),old_hits=dh,candidate_hits=ch,first_divergence=t,old_action=d['records'][t]['action'],candidate_action=c['records'][t]['action'],later_loss=max(dh-1,0)-max(ch-1,0)))
    def key(c):return (not(c['candidate_hits']>0 and c['later_loss']>0),-c['later_loss'],-(c['old_hits']-c['candidate_hits']),c['seed'])
    if prioritize_openings:
        selected=sorted([c for c in cases if c['candidate_hits']==0],key=key)[:8]
        while len(selected)<8:
            remaining=[c for c in cases if c not in selected];assert remaining
            phase=min(('left','right'),key=lambda p:sum(c['phase']==p for c in selected));options=[c for c in remaining if c['phase']==phase] or remaining;selected.append(min(options,key=key))
        return selected
    selected=[]
    for phase in ('left','right'):
        phase_cases=[c for c in cases if c['phase']==phase]
        opening=sorted([c for c in phase_cases if c['candidate_hits']==0],key=key)[:1] if opening_repair else []
        picked=opening+sorted([c for c in phase_cases if c not in opening and c['candidate_hits']>0],key=key)[:4-len(opening)] if opening_repair else sorted(phase_cases,key=key)[:4]
        picked+=sorted([c for c in phase_cases if c not in picked],key=key)[:4-len(picked)];selected+=picked
    selected+=sorted([c for c in cases if c not in selected],key=key)[:8-len(selected)]
    assert len(selected)==8;return selected


def branch(snapshot,action,deadline,expected=None,features=None,trace=False):
    trial=clone_session(snapshot);random.setstate(snapshot.python_rng);np.random.set_state(snapshot.numpy_rng);torch.set_rng_state(snapshot.torch_rng)
    before=trial.transitions;past=trial.episode_hits;count=0;raw=SparseRows() if trace else None;records=[]
    while not trial.complete and time.monotonic()<deadline:
        i=trial.transitions
        if trace:raw.add(trial)
        if features is not None:
            ix,val=policy_features(trial.latest_encoded,trial.scaler);lo,hi=features.indptr[i:i+2];assert np.array_equal(ix,features.indices[lo:hi]) and np.array_equal(val,features.data[lo:hi])
        record=step_action(trial,int(action)) if count==0 else trial.step(check_frozen=False);count+=1
        if trace:records.append(record)
        if expected is not None:
            record={k:v for k,v in record.items() if k not in ('intervention_action','unmodified_greedy_action')};assert record==expected['records'][i]
    trial.verify_frozen();assert trial.transitions-before==count and trial.initial_cameras==snapshot.initial_cameras
    if expected is not None and trial.complete:assert trial.summary()==expected['report']
    result=dict(complete=trial.complete,subsequent_hits=int(trial.episode_hits-past),cameras=count,initials=0,exact_null_parity=expected is not None and trial.complete)
    if trace:result.update(_raw=raw.matrix(),_records=records)
    return result


def task(directory,case,index,deadline,save_positive=False):
    directory=Path(directory);began=time.monotonic();d=base.read(Path(case['old_receipt']));c=base.read(Path(case['candidate_receipt']));X=sp.load_npz(case['old_feature']);CX=sp.load_npz(case['candidate_feature']);t=case['first_divergence'];assert (X[:t+1]!=CX[:t+1]).nnz==0
    session=start(np.load(directory/'old.npy'),d['config']);prefix=0
    while prefix<t and time.monotonic()<deadline:
        record=session.step(check_frozen=False);assert record==d['records'][prefix];prefix+=1
    session.verify_frozen()
    if prefix<t:
        result=dict(seed=case['seed'],complete=False,prefix_cameras=prefix,initials=session.initial_cameras,total_cameras=prefix,reason='prefix deadline');base.atomic_json(directory/'cases'/f'case_{index:02d}.json',result);return result
    ix,val=policy_features(session.latest_encoded,session.scaler);lo,hi=X.indptr[t:t+2];assert np.array_equal(ix,X.indices[lo:hi]) and np.array_equal(val,X.data[lo:hi]);row=f'rows/case_{index:02d}.npz';sp.save_npz(directory/row,X[t:t+1])
    session.policy=FrozenLinearPolicy(np.load(directory/'candidate.npy'));session.initial_learned_hash=session.learned_hash();FreshPongEvaluator.validate_restored(session);assert session._choose_policy()[0]==case['candidate_action']
    past=session.episode_hits;baseline=int(c['report']['hits']-past);null=branch(session,case['candidate_action'],deadline,c,CX) if index==0 else None
    old=branch(session,case['old_action'],deadline,trace=save_positive);third=None;outcomes={str(case['candidate_action']):baseline};positive=old['complete'] and old['subsequent_hits']>baseline
    if old['complete']:outcomes[str(case['old_action'])]=old['subsequent_hits']
    if positive:
        a=next(a for a in range(3) if a not in (case['old_action'],case['candidate_action']));third=branch(session,a,deadline,trace=save_positive)
        if third['complete']:outcomes[str(a)]=third['subsequent_hits']
    complete=old['complete'] and (null is None or null['complete']) and (not positive or third['complete']);eligible=complete and positive;preferences=[]
    if eligible:
        good=case['old_action'];preferences=[dict(good=good,bad=int(a),good_subsequent_hits=outcomes[str(good)],bad_subsequent_hits=h) for a,h in outcomes.items() if h<outcomes[str(good)]]
    donors=[]
    if save_positive:
        for action,value in [(case['old_action'],old)]+([(a,third)] if third is not None else []):
            suffix=value.pop('_raw');records=value.pop('_records')
            if not value['complete'] or value['subsequent_hits']<=baseline:continue
            full=sp.vstack((CX[:t],suffix),format='csr');full_records=c['records'][:t]+records;actions=np.array([r['action'] for r in full_records],dtype=np.int64);hits=[i for i,r in enumerate(full_records) if r['hits']];name=f'donors/case_{index:02d}_a{action}'
            sp.save_npz(directory/(name+'.npz'),full);np.save(directory/(name+'_actions.npy'),actions,allow_pickle=False);base.atomic_json(directory/(name+'_records.json'),full_records)
            donors.append(dict(seed=case['seed'],phase=case['phase'],role='opening' if past==0 else 'later',path=name+'.npz',actions_path=name+'_actions.npy',records_path=name+'_records.json',hit_indices=hits,target_hits=len(hits),successful_prefix_rows=hits[-1]+1,forced_transition=t,forced_action=action,one_camera_override=True,subsequent_hit_advantage=value['subsequent_hits']-baseline,prefix_source_receipt=case['candidate_receipt'],prefix_reused_rows=t))
    result=dict(seed=case['seed'],phase=case['phase'],complete=complete,first_divergence=t,legal_row=row,past_hits=past,candidate_action=case['candidate_action'],old_action=case['old_action'],outcomes=outcomes,eligible=eligible,preferences=preferences,prefix_cameras=prefix,null_suffix_cameras=null['cameras'] if null else 0,old_suffix_cameras=old['cameras'],third_suffix_cameras=third['cameras'] if third else 0,total_cameras=prefix+(null['cameras'] if null else 0)+old['cameras']+(third['cameras'] if third else 0),initials=session.initial_cameras,default_candidate_suffix_reused_cameras=c['transitions']-t,exact_null_parity=null['exact_null_parity'] if null else None,wall_seconds=time.monotonic()-began)
    result['rewarded_donors']=donors;base.atomic_json(directory/'cases'/f'case_{index:02d}.json',result);return result


def run(directory,bank_path,candidate,exclude_sources=(),opening_repair=False,save_positive=False,prioritize_openings=False):
    directory=Path(directory);bank_path=Path(bank_path);candidate=Path(candidate);assert not directory.exists();directory.mkdir();(directory/'cases').mkdir();(directory/'rows').mkdir();bank=base.read(bank_path);assert bank['source']==base.source_signature()
    (directory/'old.npy').write_bytes(Path(bank['current_head']).read_bytes());(directory/'candidate.npy').write_bytes((candidate/'proposal.npy').read_bytes());excluded=sorted({c['seed'] for source in exclude_sources for c in base.read(Path(source)/'manifest.json')['cases']});cases=choose(bank,candidate,excluded,opening_repair,prioritize_openings);files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_divergence_credit.py','pong_action_credit.py','rally_raw_capture.py')}
    if save_positive:(directory/'donors').mkdir()
    # This fixed TRAIN sentinel is selected before any branch outcomes or fit.
    oldrows={e['seed']:base.read(Path(e['receipt'])) for e in bank['bank']};sentinel=[c['seed'] for c in cases]
    for phase in ('left','right'):
        available=sorted([s for s in oldrows if s not in sentinel and base.leftward(s)==(phase=='left')]);picked=[]
        for band in (0,1,2,3):
            group=[s for s in available if min(oldrows[s]['report']['hits'],3)==band];picked+=group[:3]
        picked+= [s for s in available if s not in picked][:12-len(picked)];sentinel+=picked
    assert len(sentinel)==32 and len(set(sentinel))==32;base.atomic_json(directory/'sentinel-schedule.json',dict(seeds=sentinel,selection='eight targets plus12other starts/phase spread over d7 hit depths0,1,2,3+; fixed before branch/fit',TRAIN_not_holdout=True))
    manifest=dict(source=base.source_signature(),files=files,bank_path=str(bank_path),candidate_root=str(candidate),old_sha256=base.sha(directory/'old.npy'),candidate_sha256=base.sha(directory/'candidate.npy'),cases=cases,excluded_previous_cases=excluded,opening_repair=opening_repair,prioritize_all_current_opening_regressions=prioritize_openings,save_positive_full_raw_actions=save_positive,selection='all remaining opening regressions first when requested, then phase-balanced costly later losses; otherwise four perphase with optional oneopening',seconds=150,credit='old single action versus candidate action, identical state and SAME candidate continuation to20s/native end; third queried only on positive cases; ties unordered',no_privileged_policy_inputs=True)
    base.atomic_json(directory/'manifest.json',manifest);began=time.monotonic();deadline=began+150;base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline))
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        futures=[pool.submit(task,str(directory),case,i,deadline-1,save_positive) for i,case in enumerate(cases)];results=[f.result() for f in futures]
    summary=dict(status='complete' if all(r['complete'] for r in results) else 'bounded_partial',results=results,positive_independent_cases=sum(r.get('eligible',False) for r in results),transition_cameras=sum(r['total_cameras'] for r in results),initials=sum(r['initials'] for r in results),exact_null_parity=results[0].get('exact_null_parity',False),wall_seconds=time.monotonic()-began,source_frozen=manifest['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files},no_counter_or_physics_features=True)
    base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--bank',required=True);p.add_argument('--candidate',required=True);p.add_argument('--exclude-source',action='append',default=[]);p.add_argument('--repair-openings',action='store_true');p.add_argument('--prioritize-openings',action='store_true');p.add_argument('--save-positive',action='store_true');a=p.parse_args();run(a.directory,a.bank,a.candidate,a.exclude_source,a.repair_openings,a.save_positive,a.prioritize_openings)
