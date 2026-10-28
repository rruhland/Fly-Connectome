"""Cumulative measured preferences, target protection, fixed TRAIN sentinel."""
import argparse,json,time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from fly_connectome.control.policy import FrozenLinearPolicy
from pong_reward_witness_projection import project
from run_rally_fixed_bank_check import run as native_check
from return_metrics import aggregate
import run_practical_pong_v2 as base


def run(directory,screen,previous=()):
    directory=Path(directory);screen=Path(screen);assert not directory.exists();directory.mkdir();s=base.read(screen/'summary.json');sm=base.read(screen/'manifest.json');assert s['source_frozen'] and s['exact_null_parity']
    groups=[(Path(p),base.read(Path(p)/'summary.json')) for p in previous]+[(screen,s)];new=[r for r in s['results'] if r.get('eligible')]
    if len(new)<2:base.atomic_json(directory/'summary.json',dict(status='stopped_fewer_than_two_new_preferences',new_cases=len(new),new_observations=0));return
    original=base.read(base.EXPERIMENTS/'pong-rally-target-projection-20261028-r1/manifest.json');source=Path(original['witness_root']);w=original['witness'];X=sp.load_npz(source/w['path']);actions=np.load(source/w['actions_path']);normal=base.read(base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json');scale=np.load(Path(sm['bank_path']).parent/'feature-scale.npy');reference=np.load(screen/'candidate.npy');feasible=np.load(base.EXPERIMENTS/'pong-rally-diagonal-metric-20261028-r1/proposal.npy');pairs=[];positive=[]
    for folder,summary in groups:
        for row in summary['results']:
            if not row.get('eligible'):continue
            positive.append((folder,row))
            for pair in row['preferences']:pairs.append(dict(X=sp.load_npz(folder/row['legal_row']),good=pair['good'],bad=pair['bad']))
    files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_consequence_correction.py','pong_reward_witness_projection.py','run_rally_fixed_bank_check.py')};base.atomic_json(directory/'manifest.json',dict(source=base.source_signature(),files=files,screen_root=str(screen),previous_screens=list(map(str,previous)),reference_sha256=base.sha(screen/'candidate.npy'),new_cases=len(new),cumulative_pairs=len(pairs),full_target=w,projection_seconds=60,sentinel_seconds=120,native_seconds=500,sentinel_selection=str(screen/'sentinel-schedule.json'),sentinel_rule='at least4more total returns and3more later returns versus current experimental head, at most1first-return loss; TRAIN heuristic, report d7 comparison separately',no_blanket_old_action_locks=True))
    began=time.monotonic();candidate,report=project(reference,feasible,X,actions,scale,normal['kappa'],began+60,pairs);base.save_array(directory/'proposal.npy',candidate);base.atomic_json(directory/'projection-report.json',report);policy=FrozenLinearPolicy(candidate);deployment=[]
    for folder,row in positive:
        raw=sp.load_npz(folder/row['legal_row']);choice,_=policy.choose(raw.indices,raw.data);actual=row['outcomes'].get(str(choice));before=row['outcomes'][str(row['candidate_action'])];deployment.append(dict(screen=str(folder),seed=row['seed'],greedy_action=int(choice),measured_conditional_hits=actual,baseline_conditional_hits=before,strictly_beneficial=actual is not None and actual>before))
    base.atomic_json(directory/'deployment-check.json',dict(cases=deployment,new_observations=0));new_deployed=sum(r['screen']==str(screen) and r['strictly_beneficial'] for r in deployment)
    if new_deployed<2 or not all(r['strictly_beneficial'] for r in deployment):base.atomic_json(directory/'summary.json',dict(status='stopped_preferences_not_deployed',new_deployed=new_deployed));return
    # Reuse the fixed-bank evaluator for the32predeclared TRAIN contexts.
    bank=base.read(Path(sm['bank_path']));candidate_root=Path(sm['candidate_root']);current={base.read(p)['seed']:(p,base.read(p)) for p in (candidate_root/'native_candidate/episodes').glob('*.json')};seeds=base.read(screen/'sentinel-schedule.json')['seeds'];entries=[]
    for seed in seeds:
        p,r=current[seed];assert r['complete'] and r['weights_sha256']==sm['candidate_sha256'];feature=Path(r['feature_file']);feature=feature if feature.is_absolute() else candidate_root/feature;entries.append(dict(seed=seed,receipt=str(p),raw_feature=str(feature)))
    sentinel_bank=dict(bank,current_head=str(screen/'candidate.npy'),current_head_sha256=sm['candidate_sha256'],bank=entries);base.atomic_json(directory/'sentinel-bank.json',sentinel_bank);sentinel=directory/'sentinel';native_check(sentinel,directory/'sentinel-bank.json',directory/'proposal.npy',native_seconds=120);ss=base.read(sentinel/'summary.json');oldrows={e['seed']:base.read(Path(e['receipt'])) for e in bank['bank']};d7=aggregate([oldrows[seed]['annotation'] for seed in seeds]);gain=ss['candidate']['own_returns']-ss['baseline']['own_returns'];later=ss['candidate_more']['later_returns']-ss['baseline_more']['later_returns'];first=ss['candidate']['first_incoming_returns']-ss['baseline']['first_incoming_returns'];advance=ss['status']=='complete' and gain>=4 and later>=3 and first>=-1
    review=dict(status='sentinel_complete' if ss['status']=='complete' else 'sentinel_partial',sentinel_d7=d7,sentinel_current=ss['baseline'],sentinel_candidate=ss['candidate'],total_return_gain=gain,later_return_gain=later,first_return_gain=first,advance_to_full=advance,projection=report,wall_seconds=time.monotonic()-began);base.atomic_json(directory/'summary.json',review);print(json.dumps(review),flush=True)
    if advance:native_check(directory/'full',sm['bank_path'],directory/'proposal.npy',cached_native_root=sentinel,fresh_root=str(directory/'fresh'),practical_later_screen=True,alternate_native_root=candidate_root)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--screen',required=True);p.add_argument('--previous-screen',action='append',default=[]);a=p.parse_args();run(a.directory,a.screen,a.previous_screen)
