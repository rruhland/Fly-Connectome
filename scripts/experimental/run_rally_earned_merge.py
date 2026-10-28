"""One verified own-behavior merge, fixed TRAIN sentinel, native outcome check."""
import argparse,json,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from pong_earned_trace_merge import merge
from run_rally_fixed_bank_check import run as native_check
from return_metrics import aggregate
import run_practical_pong_v2 as base


def run(directory,index_path,bank_path,reference_root):
    directory=Path(directory);directory.mkdir(exist_ok=True);assert not (directory/'work-clock.json').exists();index_path=Path(index_path);index=base.read(index_path);bank_path=Path(bank_path);bank=base.read(bank_path);reference_root=Path(reference_root);reference=np.load(index['reference_head']);assert base.sha(reference_root/'proposal.npy')==index['reference_sha256'];scale=np.load(index['feature_metric']);normal=base.read(base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json');cases=[]
    for entry in index['cases']:
        receipt=base.read(Path(entry['receipt']));assert receipt['weights_sha256']==entry['source_head_sha256']==base.sha(entry['source_head']);X=sp.load_npz(entry['raw_feature'])[:entry['end']].tocsr();actions=np.array([r['action'] for r in receipt['records'][:entry['end']]],dtype=np.int64);cases.append(dict(seed=entry['seed'],X=X,actions=actions,source_weights=np.load(entry['source_head'])))
    old={e['seed']:base.read(Path(e['receipt'])) for e in bank['bank']};current={base.read(p)['seed']:(p,base.read(p)) for p in (reference_root/'native_candidate/episodes').glob('*.json')};targets=list(dict.fromkeys(c['seed'] for c in index['cases']));sentinel=list(targets)
    # Fixed before solving: protect sampled behavior on both initial directions
    # and different achieved return depths, beyond the explicit merge targets.
    while len(sentinel)<32:
        phase=min(('left','right'),key=lambda p:sum(base.leftward(s)==(p=='left') for s in sentinel));remaining=[s for s in old if s not in sentinel and base.leftward(s)==(phase=='left')];remaining=remaining or [s for s in old if s not in sentinel];band=(len(sentinel)-len(targets))%4;group=[s for s in remaining if min(old[s]['report']['hits'],3)==band];sentinel.append(min(group or remaining))
    base.atomic_json(directory/'sentinel-schedule.json',dict(seeds=sentinel,merge_targets=targets,TRAIN_not_holdout=True,selection='all merge targets plus phase/depth-spread otherTRAIN starts, fixed before fitting'))
    files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_earned_merge.py','pong_earned_trace_merge.py','run_rally_fixed_bank_check.py')};base.atomic_json(directory/'manifest.json',dict(source=base.source_signature(),files=files,index_path=str(index_path),index_sha256=base.sha(index_path),bank_path=str(bank_path),reference_root=str(reference_root),reference_sha256=index['reference_sha256'],reward_tradeoff=index['reward_tradeoff'],seconds=60,no_known_feasible_anchor=True,no_invented_fallback=True))
    start=time.monotonic();base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=start,deadline_monotonic=start+60));candidate,report=merge(reference,cases,scale,normal['kappa'],start+60);base.atomic_json(directory/'merge-report.json',report);print(json.dumps(dict(merge=report)),flush=True)
    if candidate is None:base.atomic_json(directory/'summary.json',dict(status='no_verified_exact_merge',merge=report,new_observations=0));return
    base.save_array(directory/'proposal.npy',candidate);entries=[]
    for seed in sentinel:
        p,r=current[seed];feature=Path(r['feature_file']);feature=feature if feature.is_absolute() else reference_root/feature;entries.append(dict(seed=seed,receipt=str(p),raw_feature=str(feature)))
    sbank=dict(bank,current_head=str(reference_root/'proposal.npy'),current_head_sha256=index['reference_sha256'],bank=entries);base.atomic_json(directory/'sentinel-bank.json',sbank);native_check(directory/'sentinel',directory/'sentinel-bank.json',directory/'proposal.npy',native_seconds=120,alternate_native_root=reference_root);s=base.read(directory/'sentinel/summary.json');rows=[base.read(p) for p in (directory/'sentinel/native_candidate/episodes').glob('*.json')];other=[r for r in rows if r['seed'] not in targets];before_other=[current[r['seed']][1] for r in other];other_delta=sum(r['report']['hits'] for r in other)-sum(r['report']['hits'] for r in before_other);first_delta=s['candidate']['first_incoming_returns']-s['baseline']['first_incoming_returns'];advance=s['status']=='complete' and s['candidate']['own_returns']>s['baseline']['own_returns'] and other_delta>=-1 and first_delta>=-1
    summary=dict(status='sentinel_complete' if s['status']=='complete' else 'sentinel_partial',merge=report,sentinel=s,non_target_return_change=other_delta,first_return_change=first_delta,advance_to_full=advance,wall_seconds=time.monotonic()-start,rule='netTRAINgain, at most1lost return on the other15starts and at most1firstreturn loss; no perseedzero-loss veto');base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)
    if advance:native_check(directory/'full',bank_path,directory/'proposal.npy',cached_native_root=directory/'sentinel',fresh_root=str(directory/'fresh'),practical_later_screen=True,alternate_native_root=reference_root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--index',required=True);p.add_argument('--bank',required=True);p.add_argument('--reference-root',required=True);a=p.parse_args();run(a.directory,a.index,a.bank,a.reference_root)
