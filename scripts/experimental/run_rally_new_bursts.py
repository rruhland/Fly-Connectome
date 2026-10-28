"""Short own-reward action bursts on four unresolved TRAIN later misses."""
import argparse,time,random,json,multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from pong_action_credit import clone_session,step_action
from pong_rally_credit import outcome_from_record
from rally_raw_capture import start,SparseRows
import run_practical_pong_v2 as base


def burst(snapshot,action,mask,deadline):
    trial=clone_session(snapshot);random.setstate(snapshot.python_rng);np.random.set_state(snapshot.numpy_rng);torch.set_rng_state(snapshot.torch_rng)
    raw=SparseRows();records=[];overrides=[];outcome=None
    while not trial.complete and time.monotonic()<deadline:
        i=len(records);raw.add(trial);forced=i<len(mask) and bool(mask[i])
        r=step_action(trial,action) if forced else trial.step(check_frozen=False)
        records.append(r);overrides.append(forced);outcome,ending=outcome_from_record(r)
        if ending is not None:break
    trial.verify_frozen()
    return dict(outcome=outcome,cameras=len(records),X=raw.matrix(),records=records,overrides=overrides)


def task(root,case,index,deadline):
    root=Path(root);r=base.read(Path(case['receipt']));X=sp.load_npz(case['feature']);s=start(np.load(root/'carry.npy'),r['config']);snap={}
    for t in range(max(case['positions'])+1):
        if time.monotonic()>=deadline:break
        if t in case['positions']:snap[t]=clone_session(s)
        if t==max(case['positions']):break
        assert s.step(check_frozen=False)==r['records'][t]
    s.verify_frozen();prefix=s.transitions;spent=0;trials=[];donor=None
    for hold in (10,20,40):
        if donor or time.monotonic()>=deadline:break
        for t in case['positions']:
            if donor or t not in snap or time.monotonic()>=deadline:break
            for action in range(3):
                if time.monotonic()>=deadline:break
                mask=np.ones(hold,dtype=bool);b=burst(snap[t],action,mask,deadline);spent+=b['cameras'];trials.append(dict(kind='burst',transition=t,hold=hold,action=action,outcome=b['outcome'],cameras=b['cameras']))
                if b['outcome']!=1:continue
                best=b;bestmask=mask
                for lo,hi in reversed(list(zip(np.linspace(0,hold,5,dtype=int)[:-1],np.linspace(0,hold,5,dtype=int)[1:]))):
                    if time.monotonic()>=deadline:break
                    proposal=bestmask.copy();proposal[lo:hi]=False
                    if not proposal.any():continue
                    v=burst(snap[t],action,proposal,deadline);spent+=v['cameras'];trials.append(dict(kind='prune',start=int(lo),end=int(hi),outcome=v['outcome'],cameras=v['cameras']))
                    if v['outcome']==1:best=v;bestmask=proposal
                records=r['records'][:t]+best['records'];full=sp.vstack((X[:t],best['X']),format='csr');hits=[i for i,v in enumerate(records) if v['hits']];assert len(hits)==r['report']['hits']+1
                name=f'donors/case_{index:02d}';sp.save_npz(root/(name+'.npz'),full);base.atomic_json(root/(name+'.json'),records)
                donor=dict(seed=case['seed'],phase=case['phase'],kind='new_burst',raw_feature=str(root/(name+'.npz')),records=str(root/(name+'.json')),end=len(records),target_hits=len(hits),advantage=1,forced_transitions=[t+i for i,v in enumerate(best['overrides']) if v],forced_action=action,source_receipt=case['receipt'],source_head=str(root/'carry.npy'),interpretation='whole reward-earned behavior, not individual causal labels')
                break
    out=dict(seed=case['seed'],donor=donor,trials=trials,prefix_cameras=prefix,suffix_cameras=spent,initials=s.initial_cameras,budget_censored=time.monotonic()>=deadline)
    base.atomic_json(root/'cases'/f'{index:02d}.json',out);return out


def run(root,discovery):
    root=Path(root);discovery=Path(discovery);m=base.read(discovery/'manifest.json');cases=[c for c in m['cases'] if c['kind']=='new'];assert len(cases)==4 and not root.exists()
    root.mkdir();(root/'cases').mkdir();(root/'donors').mkdir();(root/'carry.npy').write_bytes((discovery/'carry.npy').read_bytes());source=base.source_signature();sha=base.sha(Path(__file__))
    base.atomic_json(root/'manifest.json',dict(cases=cases,discovery=str(discovery),head_sha256=base.sha(root/'carry.npy'),source=source,script_sha256=sha,seconds=180,holds=[10,20,40],prune='four chunks once',stop_first_reward_success=True,fresh_admitted=False,no_privileged_inputs=True))
    began=time.monotonic();base.atomic_json(root/'work-clock.json',dict(started_wall_ns=time.time_ns(),deadline_monotonic=began+180))
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        fs=[pool.submit(task,str(root),c,i,began+179) for i,c in enumerate(cases)];results=[f.result() for f in fs]
    out=dict(status='complete' if all(not r['budget_censored'] for r in results) else 'bounded_partial',cases=results,positive_cases=sum(r['donor'] is not None for r in results),transition_cameras=sum(r['prefix_cameras']+r['suffix_cameras'] for r in results),initials=sum(r['initials'] for r in results),wall_seconds=time.monotonic()-began,source_frozen=source==base.source_signature() and sha==base.sha(Path(__file__)))
    base.atomic_json(root/'summary.json',out);print(json.dumps(out),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--discovery',required=True);a=p.parse_args();run(a.root,a.discovery)
