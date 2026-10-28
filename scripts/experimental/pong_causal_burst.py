"""Own-reward verified donor-policy bursts; explicit overrides are never greedy labels."""
import random,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.scaling import policy_features
from pong_fixed_rbf import FreshRBFEvaluator,load_readout
from pong_action_credit import clone_session,step_action
from rally_raw_capture import SparseRows,BOOKKEEPING
import run_practical_pong_v2 as base


def improved(report,past,baseline_remaining):
    return report['complete'] and report['hits']-past>baseline_remaining


def trial(snapshot,donor,mask,deadline):
    session=clone_session(snapshot);random.setstate(snapshot.python_rng);np.random.set_state(snapshot.numpy_rng);torch.set_rng_state(snapshot.torch_rng);past=session.episode_hits;before=session.transitions;initial=session.initial_cameras;raw=SparseRows();records=[]
    while not session.complete and time.monotonic()<deadline:
        offset=session.transitions-before;raw.add(session)
        if offset<len(mask) and mask[offset]:
            action,q=donor.choose(*policy_features(session.latest_encoded,session.scaler));record=step_action(session,int(action));record['donor_policy_action']=int(action)
        else:record=session.step(check_frozen=False)
        records.append(record)
    session.verify_frozen();assert session.initial_cameras==initial and session.transitions-before==len(records)
    return dict(complete=session.complete,previous_hits=past,delta_hits=session.episode_hits-past,cameras=len(records),initial_cameras=0,report=session.summary(),records=records,X=raw.matrix())


def task(directory,case,index,deadline):
    directory=Path(directory);began=time.monotonic();current=load_readout(directory/'source-model.npz');donor=load_readout(directory/'donor-model.npz');r=base.read(case['current_receipt']);alt=base.read(case['alternative_receipt']);root=Path(case['current_root']);aroot=Path(case['alternative_root']);X=sp.load_npz(root/r['feature_file']).tocsr();AX=sp.load_npz(aroot/alt['feature_file']).tocsr();t=case['transition'];assert (X[:t+1]!=AX[:t+1]).nnz==0 and all(a['action']==b['action'] for a,b in zip(r['records'][:t],alt['records'][:t]));assert r['weights_sha256']==base.sha(directory/'source-model.npz') and alt['weights_sha256']==base.sha(directory/'donor-model.npz')
    random.seed(BOOKKEEPING);np.random.seed(BOOKKEEPING);torch.manual_seed(BOOKKEEPING);session=FreshRBFEvaluator(current,PongEvaluationConfig(**r['config']));session._begin_episode();prefix=0
    for expected in r['records'][:t]:
        if time.monotonic()>=deadline:break
        ix,val=policy_features(session.latest_encoded,session.scaler);lo,hi=X.indptr[prefix:prefix+2];assert np.array_equal(ix,X.indices[lo:hi]) and np.array_equal(val,X.data[lo:hi]);assert session.step(check_frozen=False)==expected;prefix+=1
    if prefix<t:
        result=dict(seed=case['seed'],complete=False,positive=False,prefix_cameras=prefix,transition_cameras=prefix,initial_cameras=session.initial_cameras,reason='prefix_budget');base.atomic_json(directory/f'case_{index:02d}.json',result);return result
    ix,val=policy_features(session.latest_encoded,session.scaler);lo,hi=X.indptr[t:t+2];assert np.array_equal(ix,X.indices[lo:hi]) and np.array_equal(val,X.data[lo:hi]);session.verify_frozen();snapshot=clone_session(session);past=session.episode_hits;baseline_remaining=r['report']['hits']-past;spent=prefix;trials=[];null=None
    if index==0:
        null=trial(snapshot,donor,np.zeros(10,dtype=bool),deadline);spent+=null['cameras'];assert null['complete'] and null['records']==r['records'][t:] and (null['X']!=X[t:]).nnz==0 and null['report']==r['report'];trials.append(dict(kind='null_parity',cameras=null['cameras'],hits=null['report']['hits'],complete=True))
    mask=np.ones(10,dtype=bool);best=trial(snapshot,donor,mask,deadline);spent+=best['cameras'];positive=improved(best['report'],past,baseline_remaining);trials.append(dict(kind='fixed10_policy_burst',cameras=best['cameras'],hits=best['report']['hits'],complete=best['complete'],positive=positive))
    if positive:
        boundaries=np.linspace(0,10,5,dtype=int)
        for chunk in reversed(range(4)):
            if time.monotonic()>=deadline:break
            candidate=mask.copy();candidate[boundaries[chunk]:boundaries[chunk+1]]=False
            if not candidate.any():trials.append(dict(kind='delete_chunk',chunk=chunk,known_null_reused=True,accepted=False,cameras=0));continue
            value=trial(snapshot,donor,candidate,deadline);spent+=value['cameras'];accept=improved(value['report'],past,baseline_remaining);trials.append(dict(kind='delete_chunk',chunk=chunk,cameras=value['cameras'],hits=value['report']['hits'],complete=value['complete'],accepted=accept))
            if accept:mask=candidate;best=value
    result=dict(seed=case['seed'],phase=case['phase'],complete=best['complete'],positive=positive,previous_hits=past,baseline_remaining_hits=baseline_remaining,baseline_total_hits=r['report']['hits'],branch_total_hits=best['report']['hits'],total_hit_advantage=best['report']['hits']-r['report']['hits'],transition=t,mask=mask.tolist(),trials=trials,transition_cameras=spent,initial_cameras=session.initial_cameras,wall_seconds=time.monotonic()-began,null_parity_verified=index==0 and null['complete'],source_policy_sha256=base.sha(directory/'source-model.npz'),donor_policy_sha256=base.sha(directory/'donor-model.npz'),same_original_episode_horizon=True,whole_joint_pattern_not_individual_action_causality=True,default_transport_episode_reuse_valid=False)
    if positive:
        fullX=sp.vstack((X[:t],best['X']),format='csr');records=r['records'][:t]+best['records'];base.atomic_json(directory/f'case_{index:02d}_records.json',records);sp.save_npz(directory/f'case_{index:02d}.npz',fullX);result.update(raw_feature=str(directory/f'case_{index:02d}.npz'),records=str(directory/f'case_{index:02d}_records.json'),original_receipt=case['current_receipt'],original_raw_feature=str(root/r['feature_file']),source_config=r['config'],report=best['report'],remaining_changed_choices=sum('intervention_action' in v and v['action']!=v['unmodified_greedy_action'] for v in records));assert result['remaining_changed_choices']>0 and sum(v['hits'] for v in records)==result['branch_total_hits']
    base.atomic_json(directory/f'case_{index:02d}.json',result);return result


def audited_cases(result,current,donor,current_sha,donor_sha):
    assert result['complete'] and result['positive'] and result['total_hit_advantage']>0 and result['same_original_episode_horizon'] and result['source_policy_sha256']==current_sha and result['donor_policy_sha256']==donor_sha
    X=sp.load_npz(result['raw_feature']).tocsr();records=base.read(result['records']);source=base.read(result['original_receipt']);original=sp.load_npz(result['original_raw_feature']).tocsr();assert source['weights_sha256']==current_sha and len(records)==X.shape[0] and result['report']['complete'] and sum(r['hits'] for r in records)==result['branch_total_hits'];assert result['branch_total_hits']-source['report']['hits']==result['total_hit_advantage'] and result['report']['config']==source['report']['config'];changed=[]
    for i,r in enumerate(records):
        lo,hi=X.indptr[i:i+2];ix=X.indices[lo:hi];values=X.data[lo:hi];a,q=current.choose(ix,values);assert np.array_equal(q,np.asarray(r['q']))
        if 'intervention_action' in r:
            assert result['transition']<=i<result['transition']+10 and result['mask'][i-result['transition']] and r['unmodified_greedy_action']==a and r['action']==r['intervention_action']==r['donor_policy_action']==donor.choose(ix,values)[0]
            if r['action']!=a:changed.append(i)
        else:assert r['action']==a
    assert changed and result['remaining_changed_choices']==len(changed);first=changed[0];assert (X[:first+1]!=original[:first+1]).nnz==0 and [r['action'] for r in records[:first]]==[r['action'] for r in source['records'][:first]];common=dict(seed=result['seed'],source_readout=current,source_policy_sha256=current_sha);cases=[]
    if first:cases.append(dict(common,X=X[:first].tocsr(),actions=np.array([r['action'] for r in records[:first]]),interventions=[]))
    actions=np.array([records[i]['action'] for i in changed]);interventions=[dict(row=j,action=int(actions[j]),source_action=records[i]['unmodified_greedy_action'],reward_verified=True,native_total_hit_advantage=result['total_hit_advantage'],source_transition=i) for j,i in enumerate(changed)];cases.append(dict(common,X=X[changed].tocsr(),actions=actions,interventions=interventions,causal_audit_passed=True));return cases,dict(seed=result['seed'],all_native_Q_and_greedy_or_declared_override_rows=len(records),rollin_rows=first,requested_changed_rows=changed,total_advantage=result['total_hit_advantage'],first_hit_contract_replaced_not_conflicting=True,postburst_ordinary_actions_unconstrained=True)
