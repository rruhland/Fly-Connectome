"""Current-head roll-in and native-reward discovery for one curriculum slot."""
import argparse,json,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from rally_action_equivalence import certify_episode
from run_rally_incremental_cycle import discover_task
from run_rally_raw_optimizer import native_task
import run_practical_pong_v2 as base


def acquire_one(folder,entry,index,bank_path,role,phase,deadline):
    folder=Path(folder);bank=base.read(Path(bank_path));r=base.read(Path(entry['receipt']));sourceW=np.load(bank['current_head']);W=np.load(folder/'carry.npy');X=sp.load_npz(entry['raw_feature'])
    cert=certify_episode(r,X,sourceW,W,source_weights_sha256=bank['current_head_sha256'],source_fingerprint=bank['source'],candidate_fingerprint=base.source_signature(),candidate_config=r['config'])
    paid=0;initials=0
    if cert['certified']:
        r=dict(r,records=cert['records'],weights_sha256=base.sha(folder/'carry.npy'),feature_file=entry['raw_feature']);raw=entry['raw_feature'];origin='exact current-head episode certificate'
    else:
        r=native_task(str(folder/'current'),entry['seed'],index,deadline);paid=r['transitions'];initials=r['initial_cameras'];raw=str(folder/'current'/r['feature_file']);origin='new current-head native roll-in'
    receipt=folder/'rollin_receipts'/f'case_{index:02d}.json';base.atomic_json(receipt,r)
    result=dict(seed=entry['seed'],role=role,phase=phase,rollin_receipt=str(receipt),rollin_raw_feature=raw,rollin_origin=origin,rollin_transition_cameras=paid,rollin_initials=initials,source_episode_certificate={k:v for k,v in cert.items() if k!='records'},discovery=None)
    if not r['complete'] or not r['report']['losses'] or (r['report']['hits']>0)!=(role=='later'):
        result['reason']='current-head episode no longer has selected miss role, or roll-in was partial';base.atomic_json(folder/'acquisition_cases'/f'case_{index:02d}.json',result);return result
    last=max((i for i,v in enumerate(r['records']) if v['hits']),default=-1);floor=max(last+1,5 if role=='opening' else 0)
    positions=sorted({max(floor,r['transitions']-lead) for lead in (40,80,120) if max(floor,r['transitions']-lead)<r['transitions']})
    c=dict(seed=entry['seed'],phase=phase,role=role,receipt=str(receipt),raw_feature=raw,positions=positions)
    result['discovery']=discover_task(str(folder),c,index,deadline);base.atomic_json(folder/'acquisition_cases'/f'case_{index:02d}.json',result);return result


def acquire(folder,bank_path,current_head,role,phase,seconds=40,excluded=()):
    folder=Path(folder);bank_path=Path(bank_path);current_head=Path(current_head);bank=base.read(bank_path);assert not folder.exists();folder.mkdir(parents=True)
    for n in ('cases','witnesses','rollin_receipts','acquisition_cases','current/native_candidate/episodes','current/native_candidate/features'):(folder/n).mkdir(parents=True)
    (folder/'carry.npy').write_bytes(current_head.read_bytes());(folder/'current/proposal.npy').write_bytes(current_head.read_bytes())
    for dest in (folder/'radial-cap.json',folder/'current/radial-cap.json'):dest.write_bytes(Path(bank['radial_cap']).read_bytes())
    eligible=[]
    for entry in bank['bank']:
        r=base.read(Path(entry['receipt']))
        if entry['seed'] not in excluded and r['report']['losses'] and (r['report']['hits']>0)==(role=='later') and base.leftward(entry['seed'])==(phase=='left'):eligible.append(entry)
    selected=sorted(eligible,key=lambda e:e['seed'])[:2];files={n:base.sha(Path(__file__).with_name(n)) for n in ('rally_curriculum_acquire.py','run_rally_incremental_cycle.py','run_rally_incremental_pruned.py','run_rally_raw_optimizer.py','rally_action_equivalence.py')}
    manifest=dict(bank_path=str(bank_path),bank_sha256=base.sha(bank_path),current_head=str(current_head),current_head_sha256=base.sha(current_head),role=role,phase=phase,selected=selected,seconds=seconds,source=base.source_signature(),files=files)
    base.atomic_json(folder/'manifest.json',manifest);began=time.monotonic();deadline=began+seconds;base.atomic_json(folder/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline));results=[]
    if selected:
        with ProcessPoolExecutor(max_workers=2,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
            futures=[pool.submit(acquire_one,str(folder),entry,i,str(bank_path),role,phase,deadline-1) for i,entry in enumerate(selected)]
            for f in futures:results.append(f.result())
    witnesses=[r['discovery']['witness'] for r in results if r['discovery'] and r['discovery']['witness']]
    chosen=min(witnesses,key=lambda w:(w['contrasting_overrides'],w['remaining_forced'],w['seed'])) if witnesses else None
    cameras=sum(r['rollin_transition_cameras']+(r['discovery']['prefix_cameras']+r['discovery']['suffix_cameras'] if r['discovery'] else 0) for r in results);initials=sum(r['rollin_initials']+(r['discovery']['initial_cameras'] if r['discovery'] else 0) for r in results)
    summary=dict(results=results,witnesses=witnesses,selected_witness=chosen,transition_cameras=cameras,initial_cameras=initials,wall_seconds=time.monotonic()-began,source_frozen=manifest['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files})
    base.atomic_json(folder/'summary.json',summary);print(json.dumps(dict(acquisition_complete=True,role=role,phase=phase,contexts=len(results),witnesses=len(witnesses),selected=chosen,transition_cameras=cameras,initials=initials)),flush=True);return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder');p.add_argument('--bank',required=True);p.add_argument('--head',required=True);p.add_argument('--role',required=True);p.add_argument('--phase',required=True);a=p.parse_args();acquire(a.folder,a.bank,a.head,a.role,a.phase)
