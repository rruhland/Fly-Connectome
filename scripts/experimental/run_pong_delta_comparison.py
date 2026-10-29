"""One frozen learned-dynamics policy comparison, with bounded replay coverage."""
import argparse
from pathlib import Path
import time

from stateful_pong_comparison import prepare as generic_prepare, comparison as generic_comparison
from pong_delta_native import native_task
from run_rally_option_memory import ROOT, BANK, read, write, sha, parallel_jobs
from run_rally_raw_fresh_screen import paired_metrics
import run_practical_pong_v2 as base

DEST = ROOT / 'pong-delta-native-20261029-r1'
MODEL = ROOT / 'pong-learned-delta-planner-20261029-r1/model.npz'
ADAPTER = 'pong_delta_policy_adapter'


def comparison(baseline, candidate):
    result = generic_comparison(baseline, candidate)
    result.pop('sequential_audit_rows')
    result.update(raw_baseline_q_audit_rows=sum(r['full_stateful_audit']['rows'] for r in candidate),
        planner_replay_rows=sum(r['full_stateful_audit']['planner_replay_rows'] for r in candidate),
        full_chronological_games=sum(r['full_stateful_audit'].get('full_chronological_replay',False) for r in candidate),
        replay_coverage='allactualinputsretained;allrawbaselineQchecked;onefixedcompletegamechronological;othergamesfixed<=9historyreconstructedqueries')
    return result


def passed(stats, size):
    a,b = stats['baseline'],stats['candidate']; x,y=stats['baseline_more'],stats['candidate_more']
    return (b['own_returns']>a['own_returns'] and y['later_returns']>x['later_returns']
        and b['first_incoming_returns']==a['first_incoming_returns']
        and (size==8 or (y['tail']['2']>=x['tail']['2'] and y['later_fraction']>=x['later_fraction'])))


def prepare(model_path, seconds):
    model_path=Path(model_path)
    fit=read(model_path.parent/'fit-summary.json')
    if fit.get('model_sha256',fit.get('final_sha256')) != sha(model_path):
        raise ValueError('fit receipt must identify exact frozen endpoint')
    extra=[Path(__file__),Path(__file__).with_name('pong_delta_native.py'),
        Path(__file__).with_name('pong_learned_delta_planner.py'),
        Path(__file__).with_name('pong_smdp_adaptive_forest.py'),
        Path(__file__).with_name('pong_smdp_fast_training.py'),model_path.parent/'fit-summary.json',
        model_path.parent/'spec.json']
    generic_prepare(DEST,model_path,ADAPTER,63281,extra)
    spec=read(DEST/'spec.json')
    from pong_learned_delta_planner import TAIL_SHA
    from pong_delta_policy_adapter import load_model
    model=load_model(model_path)
    spec.update(seconds=seconds,clock='ONEtotalcomparisonclockincludingconditionalfreshandallreplaychecks',
        full_replay_seed=spec['screen8'][0],full_replay_required_complete=True,
        audit='alllegalinputs/actionsretained;allrawbaselineQchecked;ONEdesignatedTRAINgamefullchronological;othersfixed<=9reconstructedhistoryqueries;firstfreshcandidatealsofullifqualified',
        planner='H8W8primitive3actions;learnedmodelonly;C1option3tailateverydepth;executeonlyfirstcommand;actualhistory16;baselineuntilactualownhit',
        inference_profile=fit.get('inference_profile',fit.get('profile')),
        no_fresh_fitting=True,no_temperature_or_threshold_variants=True)
    spec['pinned_assets'][model.tail_path]=TAIL_SHA
    bank={entry['seed']:entry for entry in read(BANK)['bank']}
    for seed in spec['screen32']:
        p=bank[seed]['receipt'];spec['pinned_assets'][p]=sha(p)
    write(DEST/'spec.json',spec)
    print(dict(status='prepared_no_native',spec_sha256=sha(DEST/'spec.json'),model_sha256=sha(model_path),seconds=seconds),flush=True)


def run(approved):
    spec=read(DEST/'spec.json')
    assert approved==sha(DEST/'spec.json') and not (DEST/'work-clock.json').exists()
    assert spec['source']==base.source_signature()
    assert all(sha(p)==h for p,h in spec['files'].items()) and all(sha(p)==h for p,h in spec['pinned_assets'].items())
    began=time.monotonic();deadline=began+spec['seconds']
    write(DEST/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,
        deadline_monotonic=deadline,seconds=spec['seconds'],single_clock=True))
    bank={entry['seed']:entry for entry in read(BANK)['bank']}; candidate={}
    summary=dict(status='evaluating',new_transition_cameras=0,new_initial_cameras=0)
    for size in (8,32):
        folder=DEST/f'screen{size}';folder.mkdir();target=spec['screen8' if size==8 else 'screen32']
        jobs=[(spec['model'],str(folder),seed,i,deadline-1,True,seed==spec['full_replay_seed'])
            for i,seed in enumerate(target) if seed not in candidate]
        rs=parallel_jobs(native_task,jobs,deadline,progress=folder/'progress.json')
        summary['new_transition_cameras']+=sum(r['transitions'] for r in rs)
        summary['new_initial_cameras']+=sum(r['initial_cameras'] for r in rs)
        candidate.update({r['seed']:r for r in rs if not r.get('not_started')})
        complete=all(s in candidate and candidate[s]['complete'] for s in target)
        chosen=[candidate[s] for s in target if s in candidate]
        stats=comparison([read(bank[r['seed']]['receipt']) for r in chosen],chosen)
        stats.update(complete=complete,elapsed_clock_seconds=time.monotonic()-began)
        write(folder/'summary.json',stats);summary[f'screen{size}']=stats
        print(dict(stage=f'screen{size}',**stats),flush=True)
        if not complete:
            summary['status']=f'screen{size}_partial';break
        full=candidate[spec['full_replay_seed']]
        assert full['complete'] and full['full_stateful_audit']['full_chronological_replay']
        if not passed(stats,size):
            summary['status']=f'rejected_screen{size}';break
        if size==32:
            folder=DEST/'fresh';seeds=read(folder/'schedules.json')['probe']
            jobs=[(spec['model'],str(folder),seed,i,deadline-1,kind,kind and i==0)
                for i,seed in enumerate(seeds) for kind in (False,True)]
            rs=parallel_jobs(native_task,jobs,deadline,progress=folder/'progress.json')
            summary['new_transition_cameras']+=sum(r['transitions'] for r in rs)
            summary['new_initial_cameras']+=sum(r['initial_cameras'] for r in rs)
            groups={kind:{r['seed']:r for r in rs if not r.get('not_started') and (r['model_sha256'] is not None)==kind} for kind in (False,True)}
            complete=all(len(g)==32 and all(r['complete'] for r in g.values()) for g in groups.values())
            stats=dict(complete=complete,elapsed_clock_seconds=time.monotonic()-began,
                started_games={str(k):len(v) for k,v in groups.items()})
            if complete:
                a=[groups[False][s] for s in seeds];b=[groups[True][s] for s in seeds]
                stats.update(comparison(a,b),paired=paired_metrics(a,b),qualified=passed(comparison(a,b),32))
            write(folder/'summary.json',stats);summary.update(fresh=stats,status='fresh_complete' if complete else 'fresh_partial')
    summary.update(wall_seconds=time.monotonic()-began,post_deadline_audit_cleanup_seconds=max(0.,time.monotonic()-deadline),
        source_frozen=all(sha(p)==h for p,h in spec['files'].items()),
        assets_frozen=all(sha(p)==h for p,h in spec['pinned_assets'].items()),no_fit=True,no_automatic_promotion=True)
    write(DEST/'summary.json',summary)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','run'])
    p.add_argument('--model',type=Path,default=MODEL);p.add_argument('--seconds',type=int,default=600)
    p.add_argument('--approved-spec-sha256');a=p.parse_args()
    if a.operation=='prepare':prepare(a.model,a.seconds)
    else:run(a.approved_spec_sha256)
