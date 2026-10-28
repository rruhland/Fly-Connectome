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
from rally_raw_capture import start
import run_practical_pong_v2 as base


def choose(bank,candidate):
    old={e['seed']:e for e in bank['bank']};cases=[]
    for p in sorted((candidate/'native_candidate/episodes').glob('*.json')):
        c=base.read(p);e=old[c['seed']];d=base.read(Path(e['receipt']));dh=d['report']['hits'];ch=c['report']['hits']
        if ch>=dh:continue
        t=next(i for i,(a,b) in enumerate(zip(d['records'],c['records'])) if a['action']!=b['action']);assert t<=max(i for i,r in enumerate(d['records']) if r['hits'])
        f=Path(c['feature_file']);f=f if f.is_absolute() else candidate/f
        cases.append(dict(seed=c['seed'],phase='left' if base.leftward(c['seed']) else 'right',old_receipt=e['receipt'],old_feature=e['raw_feature'],candidate_receipt=str(p),candidate_feature=str(f),old_hits=dh,candidate_hits=ch,first_divergence=t,old_action=d['records'][t]['action'],candidate_action=c['records'][t]['action'],later_loss=max(dh-1,0)-max(ch-1,0)))
    def key(c):return (not(c['candidate_hits']>0 and c['later_loss']>0),-c['later_loss'],-(c['old_hits']-c['candidate_hits']),c['seed'])
    selected=[]
    for phase in ('left','right'):selected+=sorted([c for c in cases if c['phase']==phase],key=key)[:4]
    selected+=sorted([c for c in cases if c not in selected],key=key)[:8-len(selected)]
    assert len(selected)==8;return selected


def branch(snapshot,action,deadline,expected=None,features=None):
    trial=clone_session(snapshot);random.setstate(snapshot.python_rng);np.random.set_state(snapshot.numpy_rng);torch.set_rng_state(snapshot.torch_rng)
    before=trial.transitions;past=trial.episode_hits;count=0
    while not trial.complete and time.monotonic()<deadline:
        i=trial.transitions
        if features is not None:
            ix,val=policy_features(trial.latest_encoded,trial.scaler);lo,hi=features.indptr[i:i+2];assert np.array_equal(ix,features.indices[lo:hi]) and np.array_equal(val,features.data[lo:hi])
        record=step_action(trial,int(action)) if count==0 else trial.step(check_frozen=False);count+=1
        if expected is not None:
            record={k:v for k,v in record.items() if k not in ('intervention_action','unmodified_greedy_action')};assert record==expected['records'][i]
    trial.verify_frozen();assert trial.transitions-before==count and trial.initial_cameras==snapshot.initial_cameras
    if expected is not None and trial.complete:assert trial.summary()==expected['report']
    return dict(complete=trial.complete,subsequent_hits=int(trial.episode_hits-past),cameras=count,initials=0,exact_null_parity=expected is not None and trial.complete)


def task(directory,case,index,deadline):
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
    old=branch(session,case['old_action'],deadline);third=None;outcomes={str(case['candidate_action']):baseline};positive=old['complete'] and old['subsequent_hits']>baseline
    if old['complete']:outcomes[str(case['old_action'])]=old['subsequent_hits']
    if positive:
        a=next(a for a in range(3) if a not in (case['old_action'],case['candidate_action']));third=branch(session,a,deadline)
        if third['complete']:outcomes[str(a)]=third['subsequent_hits']
    complete=old['complete'] and (null is None or null['complete']) and (not positive or third['complete']);eligible=complete and positive;preferences=[]
    if eligible:
        good=case['old_action'];preferences=[dict(good=good,bad=int(a),good_subsequent_hits=outcomes[str(good)],bad_subsequent_hits=h) for a,h in outcomes.items() if h<outcomes[str(good)]]
    result=dict(seed=case['seed'],phase=case['phase'],complete=complete,first_divergence=t,legal_row=row,past_hits=past,candidate_action=case['candidate_action'],old_action=case['old_action'],outcomes=outcomes,eligible=eligible,preferences=preferences,prefix_cameras=prefix,null_suffix_cameras=null['cameras'] if null else 0,old_suffix_cameras=old['cameras'],third_suffix_cameras=third['cameras'] if third else 0,total_cameras=prefix+(null['cameras'] if null else 0)+old['cameras']+(third['cameras'] if third else 0),initials=session.initial_cameras,default_candidate_suffix_reused_cameras=c['transitions']-t,exact_null_parity=null['exact_null_parity'] if null else None,wall_seconds=time.monotonic()-began)
    base.atomic_json(directory/'cases'/f'case_{index:02d}.json',result);return result


def run(directory,bank_path,candidate):
    directory=Path(directory);bank_path=Path(bank_path);candidate=Path(candidate);assert not directory.exists();directory.mkdir();(directory/'cases').mkdir();(directory/'rows').mkdir();bank=base.read(bank_path);assert bank['source']==base.source_signature()
    (directory/'old.npy').write_bytes(Path(bank['current_head']).read_bytes());(directory/'candidate.npy').write_bytes((candidate/'proposal.npy').read_bytes());cases=choose(bank,candidate);files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_divergence_credit.py','pong_action_credit.py','rally_raw_capture.py')}
    manifest=dict(source=base.source_signature(),files=files,bank_path=str(bank_path),candidate_root=str(candidate),old_sha256=base.sha(directory/'old.npy'),candidate_sha256=base.sha(directory/'candidate.npy'),cases=cases,selection='four per initial direction; first retained and later-loss cases first, then descending later/total hit loss, seed ascending',seconds=150,credit='old single action versus candidate action, identical state and SAME candidate continuation to20s/native end; third queried only on positive cases; ties unordered',no_privileged_policy_inputs=True)
    base.atomic_json(directory/'manifest.json',manifest);began=time.monotonic();deadline=began+150;base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline))
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        futures=[pool.submit(task,str(directory),case,i,deadline-1) for i,case in enumerate(cases)];results=[f.result() for f in futures]
    summary=dict(status='complete' if all(r['complete'] for r in results) else 'bounded_partial',results=results,positive_independent_cases=sum(r.get('eligible',False) for r in results),transition_cameras=sum(r['total_cameras'] for r in results),initials=sum(r['initials'] for r in results),exact_null_parity=results[0].get('exact_null_parity',False),wall_seconds=time.monotonic()-began,source_frozen=manifest['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files},no_counter_or_physics_features=True)
    base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--bank',required=True);p.add_argument('--candidate',required=True);a=p.parse_args();run(a.directory,a.bank,a.candidate)
