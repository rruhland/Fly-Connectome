"""Validate exact-action reuse against two already executed native collections."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from rally_action_equivalence import certify_episode
import run_practical_pong_v2 as base


def run(source_root,target_root,output):
    began=time.monotonic();source_root=Path(source_root);target_root=Path(target_root)
    sm=base.read(source_root/'manifest.json');tm=base.read(target_root/'manifest.json')
    old=np.load(source_root/'proposal.npy');new=np.load(target_root/'proposal.npy');sha=base.sha(source_root/'proposal.npy')
    targets={r['seed']:r for p in (target_root/'native_candidate/episodes').glob('*.json') for r in [base.read(p)]}
    counts=dict(checked_episodes=0,certified_episodes=0,reused_camera_potential=0,exact_first_divergences=0);divergences=[];probe=None
    for p in sorted((source_root/'native_candidate/episodes').glob('*.json')):
        r=base.read(p);t=targets[r['seed']];X=sp.load_npz(source_root/r['feature_file'])
        kwargs=dict(source_weights_sha256=sha,source_fingerprint=sm['source'],candidate_fingerprint=tm['source'],candidate_config=t['config'])
        c=certify_episode(r,X,old,new,**kwargs);counts['checked_episodes']+=1
        if c['certified']:
            assert c['records']==t['records'],('record mismatch',r['seed'])
            for key in ('annotation','opportunity_events','report'):assert r[key]==t[key],(key,r['seed'])
            Y=sp.load_npz(target_root/t['feature_file']);assert X.shape==Y.shape and (X!=Y).nnz==0
            counts['certified_episodes']+=1;counts['reused_camera_potential']+=r['transitions']
        else:
            assert c['reason']=='candidate_action_changed',c
            i=c['first_disagreement'];assert len(t['records'])>i
            assert [v['action'] for v in r['records'][:i]]==[v['action'] for v in t['records'][:i]]
            assert r['records'][i]['action']!=t['records'][i]['action']
            counts['exact_first_divergences']+=1;divergences.append(i)
        if probe is None:probe=(r,X,kwargs)
    r,X,kwargs=probe
    changed=dict(kwargs,candidate_fingerprint={'wrong':'context'})
    assert not certify_episode(r,X,old,new,**changed)['certified']
    changed=dict(kwargs,candidate_config=dict(r['config'],opponent_speed=.123))
    assert not certify_episode(r,X,old,new,**changed)['certified']
    corrupt=copy.deepcopy(r);corrupt['records'][0]['q'][0]+=1
    assert certify_episode(corrupt,X,old,new,**kwargs)['reason']=='source_policy_record_mismatch'
    result=dict(passed=True,**counts,source_root=str(source_root),target_root=str(target_root),new_camera_observations=0,
                first_divergence_quantiles=np.quantile(divergences,[0,.25,.5,.75,1]).tolist() if divergences else [],
                first_divergences_within5_cameras=sum(i<5 for i in divergences),
                source_integrity_guards_passed=True,wall_seconds=time.monotonic()-began,
                evidence='Certified updated q records, outcomes, event sequences and raw feature trajectories exactly match independent native executions; every rejected episode first diverges at the predicted action.')
    base.atomic_json(output,result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source_root');p.add_argument('target_root');p.add_argument('output');a=p.parse_args();run(a.source_root,a.target_root,a.output)
