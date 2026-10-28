"""Bounded all-action discovery on existing TRAIN misses; own rewards only."""
import argparse, json, time, multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from pong_action_credit import clone_session
from rally_raw_capture import start
from run_rally_divergence_credit import branch
import run_practical_pong_v2 as base


def task(root, case, index, deadline):
    root=Path(root); r=base.read(Path(case['receipt'])); X=sp.load_npz(case['feature'])
    session=start(np.load(root/'carry.npy'),r['config']); snapshots={}
    for t in range(max(case['positions'])+1):
        if time.monotonic()>=deadline:break
        if t in case['positions']:snapshots[t]=clone_session(session)
        if t==max(case['positions']):break
        assert session.step(check_frozen=False)==r['records'][t]
    session.verify_frozen();prefix=session.transitions; spent=0;comparisons=[];donors=[]
    for t in case['positions']:
        if t not in snapshots or time.monotonic()>=deadline:break
        snapshot=snapshots[t];past=snapshot.episode_hits; greedy=r['records'][t]['action'];baseline=r['report']['hits']-past
        null=branch(snapshot,greedy,deadline,r,X) if index==0 and not comparisons else None
        if null:spent+=null['cameras'];assert null['exact_null_parity']
        outcomes={str(greedy):baseline};complete=True
        for action in range(3):
            if action==greedy:continue
            b=branch(snapshot,action,deadline,trace=True);spent+=b['cameras'];complete &= b['complete']
            if b['complete']:outcomes[str(action)]=b['subsequent_hits']
            if not b['complete'] or b['subsequent_hits']<=baseline:continue
            records=r['records'][:t]+b['_records'];full=sp.vstack((X[:t],b['_raw']),format='csr');hits=[i for i,v in enumerate(records) if v['hits']];name=f'donors/case_{index:02d}_t{t}_a{action}'
            sp.save_npz(root/(name+'.npz'),full);base.atomic_json(root/(name+'.json'),records)
            donors.append(dict(seed=case['seed'],kind=case['kind'],phase=case['phase'],raw_feature=str(root/(name+'.npz')),records=str(root/(name+'.json')),end=hits[-1]+1,target_hits=len(hits),advantage=b['subsequent_hits']-baseline,forced_transition=t,forced_action=action,source_receipt=case['receipt'],source_head=str(root/'carry.npy')))
        comparisons.append(dict(transition=t,past_hits=past,greedy=greedy,outcomes=outcomes,complete=bool(complete),null_parity=null['exact_null_parity'] if null else None))
        if donors:break
    result=dict(seed=case['seed'],kind=case['kind'],comparisons=comparisons,donors=donors,prefix_cameras=prefix,suffix_cameras=spent,initials=session.initial_cameras,complete=bool(comparisons) and all(v['complete'] for v in comparisons))
    base.atomic_json(root/'cases'/f'{index:02d}.json',result);return result


def run(root,current,bankfile):
    root=Path(root);current=Path(current);assert not root.exists();bank=base.read(Path(bankfile));old={e['seed']:base.read(Path(e['receipt'])) for e in bank['bank']}
    rows={r['seed']:(p,r) for p in sorted((current/'native_candidate/episodes').glob('*.json')) for r in [base.read(p)]}
    assert len(rows)==160 and all(r['complete'] for p,r in rows.values())
    best={s:old[s]['report']['hits'] for s in rows}
    for name in ('pong-rally-divergence-correction-native-completion-20261028-r1','pong-rally-divergence-correction-20261028-r2','pong-rally-divergence-correction-20261028-r3'):
        for p in (base.EXPERIMENTS/name/'native_candidate/episodes').glob('*.json'):
            r=base.read(p);s=r['seed'];best[s]=max(best[s],r['report']['hits'])
    cases=[]
    for kind in ('recovery','new'):
        for left in (True,False):
            eligible=[]
            for s,(p,r) in rows.items():
                h=r['report']['hits']
                if not h or not r['report']['losses'] or base.leftward(s)!=left:continue
                if kind=='recovery' and old[s]['report']['hits']<=h:continue
                if kind=='new' and h<best[s]:continue
                eligible.append(s)
            for s in sorted(eligible)[:2]:
                p,r=rows[s];last=max(i for i,v in enumerate(r['records']) if v['hits']);end=r['transitions'];mid=(last+1+end)//2
                first=next(i for i,(a,b) in enumerate(zip(old[s]['records'],r['records'])) if a['action']!=b['action']) if kind=='recovery' else min(last+6,end-1)
                f=Path(r['feature_file']);f=f if f.is_absolute() else current/f
                cases.append(dict(seed=s,kind=kind,phase='left' if left else 'right',receipt=str(p),feature=str(f),positions=list(dict.fromkeys((first,mid))),baseline_hits=r['report']['hits']))
    assert len(cases)==8
    root.mkdir();(root/'cases').mkdir();(root/'donors').mkdir();(root/'carry.npy').write_bytes((current/'proposal.npy').read_bytes())
    files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_current_discovery.py','run_rally_divergence_credit.py','pong_action_credit.py')}
    base.atomic_json(root/'manifest.json',dict(cases=cases,current_root=str(current),bank=str(bankfile),source=base.source_signature(),files=files,head_sha256=base.sha(root/'carry.npy'),seconds=180,fresh_admitted=False,selection='2 per initial direction per recovery/new group by seed; new equals/exceeds named recent policies; two reward/time selected states; all3 actions; stop at first reward improvement',no_privileged_features=True))
    began=time.monotonic();base.atomic_json(root/'work-clock.json',dict(started_wall_ns=time.time_ns(),deadline_monotonic=began+180))
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        fs=[pool.submit(task,str(root),c,i,began+179) for i,c in enumerate(cases)];results=[f.result() for f in fs]
    summary=dict(status='complete' if all(r['complete'] for r in results) else 'bounded_partial',cases=results,positive_cases=sum(bool(r['donors']) for r in results),positive_by_kind={k:sum(bool(r['donors']) for r in results if r['kind']==k) for k in ('recovery','new')},transition_cameras=sum(r['prefix_cameras']+r['suffix_cameras'] for r in results),initials=sum(r['initials'] for r in results),wall_seconds=time.monotonic()-began,source_frozen=files=={n:base.sha(Path(__file__).with_name(n)) for n in files})
    base.atomic_json(root/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--current',required=True);p.add_argument('--bank',required=True);a=p.parse_args();run(a.root,a.current,a.bank)
