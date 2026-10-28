"""One earned repair with soft successful-action memory; native returns decide."""
import argparse,json,time
from pathlib import Path
import numpy as np
from pong_rally_raw_margin import fit
from run_rally_guarded_curriculum import earned_cases,evaluate,check_target
from run_rally_incremental_cycle import dataset
from run_rally_raw_optimizer import more
from return_metrics import aggregate
import run_practical_pong_v2 as base


def run(directory,bank_path,source,seed,metric_path):
    directory=Path(directory);bank_path=Path(bank_path);source=Path(source);metric_path=Path(metric_path);assert not directory.exists();directory.mkdir();bank=base.read(bank_path);assert bank['source']==base.source_signature()
    (directory/'carry.npy').write_bytes(Path(bank['current_head']).read_bytes());(directory/'radial-cap.json').write_bytes(Path(bank['radial_cap']).read_bytes());(directory/'feature-scale.npy').write_bytes(metric_path.read_bytes());W=np.load(directory/'carry.npy');entries=bank['bank'];baseline=[base.read(Path(e['receipt'])) for e in entries];byseed={r['seed']:r for r in baseline};entry=next(e for e in entries if e['seed']==seed)
    witnesses=[w for w in base.read(source/'summary.json')['witnesses'] if w['seed']==seed];assert len(witnesses)==1;witness=witnesses[0];assert witness['target_hits']>byseed[seed]['report']['hits'];positive,_,matching=dataset(source,W,[],[witness]);preservation=[]
    for e,r in zip(entries,baseline):preservation+=earned_cases(r,e['raw_feature'])
    files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_soft_activity_repair.py','pong_rally_raw_margin.py','run_rally_guarded_curriculum.py','run_rally_raw_optimizer.py')}
    manifest=dict(source=base.source_signature(),files=files,bank_path=str(bank_path),bank_sha256=base.sha(bank_path),carry_sha256=base.sha(directory/'carry.npy'),witness_root=str(source),witness=witness,feature_metric_sha256=base.sha(metric_path),ridge=1e-6,fit_seconds=90,target_seconds=30,native_seconds=500,earned_segments=len(preservation),no_hard_action_constraints=True,no_parameter_grid=True,no_backprop=True,no_fitting_to_fresh=True)
    base.atomic_json(directory/'manifest.json',manifest);began=time.monotonic();base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began));original=base.read(base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json');normal={k:original[k] for k in ('input_rms','kappa')};normal['feature_scale']=np.load(metric_path)
    delta,report=fit(W,positive,preservation,matching,normal,time.monotonic()+90,ridge=1e-6);base.save_array(directory/'delta.npy',delta);base.save_array(directory/'proposal.npy',W+delta);base.atomic_json(directory/'fit-report.json',report);print(json.dumps(dict(fit=report)),flush=True)
    target,_=check_target(directory,seed,byseed[seed]['report']['hits'],witness['role'],witness['phase'],bank,entry,byseed[seed]);base.atomic_json(directory/'target-result.json',target)
    if not target['accepted']:
        base.atomic_json(directory/'summary.json',dict(status='no_new_target_return',target=target,whole_bank_skipped=True,wall_seconds=time.monotonic()-began,source_frozen=files=={n:base.sha(Path(__file__).with_name(n)) for n in files}));return
    outcomes,cost=evaluate(directory,bank,entries,baseline,directory/'proposal.npy',directory/'target-native');complete=len(outcomes)==len(entries) and all(r['complete'] for r in outcomes);a=aggregate([r['annotation'] for r in baseline]);b=aggregate([r['annotation'] for r in outcomes]);am=more(baseline);bm=more(outcomes);changes=[dict(seed=r['seed'],before_hits=byseed[r['seed']]['report']['hits'],after_hits=r['report']['hits']) for r in outcomes if r['report']['hits']!=byseed[r['seed']]['report']['hits']]
    useful=complete and b['own_returns']>a['own_returns'] and bm['later_returns']>am['later_returns'];frozen=bank['source']==base.source_signature() and files=={n:base.sha(Path(__file__).with_name(n)) for n in files}
    summary=dict(status='complete' if complete else 'bounded_partial',baseline=a,candidate=b,baseline_more=am,candidate_more=bm,target=target,per_seed_hit_changes=changes,native=cost,fresh_eligible=useful,first_return_tradeoff_reported_not_vetoed=True,wall_seconds=time.monotonic()-began,source_frozen=frozen,carry_immutable=base.sha(directory/'carry.npy')==manifest['carry_sha256'],proposal_sha256=base.sha(directory/'proposal.npy'),no_automatic_promotion=True)
    base.atomic_json(directory/'summary.json',summary);base.atomic_json(directory/'completion-verification.json',dict(passed=complete and frozen,native_games=len(outcomes),new_verification_observations=0));print(json.dumps(summary),flush=True)
    if useful:
        from run_rally_raw_fresh_screen import run as fresh_run
        import run_rally_baseline_pong as schedule_module
        fresh=directory.with_name(directory.name+'-fresh');assert not fresh.exists();fresh.mkdir();schedule_module.ROOT=2397461809;schedule_module.TAG=62911;draw=schedule_module.schedule();seeds=draw.pop('train');draw.update(probe=seeds[:32],unused_reserved=seeds[32:],selection='fixed new full-game screen, d7 versus soft activity-metric repair; no fit/admission');base.atomic_json(fresh/'schedules.json',draw);fresh_run(fresh,Path(bank['current_head']).parent,directory)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--bank',required=True);p.add_argument('--source',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--metric',required=True);a=p.parse_args();run(a.directory,a.bank,a.source,a.seed,a.metric)
