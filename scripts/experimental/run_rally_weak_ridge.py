"""One justified same-data soft-margin refit with explicit weaker regularization."""
import argparse,json,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
import numpy as np
from rally_aggregate_data import load
from pong_rally_margin import fit
from pong_rally_imitation import diagnostics
from run_rally_baseline_pong import collect_one
from return_metrics import aggregate
import run_practical_pong_v2 as base

RIDGE=1e-6

def pins():return {n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_weak_ridge.py','rally_aggregate_data.py','pong_rally_margin.py','pong_rally_imitation.py','run_rally_baseline_pong.py')}


def run(directory,data_source):
    directory=Path(directory);source=Path(data_source);assert base.read(source/'completion-verification.json')['passed'];assert not directory.exists();directory.mkdir(parents=True)
    began=time.monotonic();base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began));overall=began+360
    for name in ('carry.npy','carry_theta.npy','search-transform.npy','radial-cap.json','old-transform.npy','extra-transform.npy'):(directory/name).write_bytes((source/name).read_bytes())
    theta,positive,preservation,matching,rows,memories=load(source)
    normal_path=base.EXPERIMENTS/'pong-action-credit-soft-preserve-20261025-r1/fit-report.json';normal=base.read(normal_path);normal={k:normal[k] for k in ('input_rms','kappa')}
    manifest=dict(source=base.source_signature(),files=pins(),data_source=str(source),carry_sha256=base.sha(directory/'carry.npy'),ridge=RIDGE,previous_ridge=.001,
        reason='same566D positive-only diagnostic reached99.96% at1e-6 vs58.7% at.001; test regularization underfitting, not a parameter grid',
        no_new_discovery=True,no_new_capture=True,no_holdout_fitted=True,work_seconds=360,native_seconds=300,native_camera_cap=96000,
        groups='same saved group/episode/segment weighting',positive_weight=1.,broad_success_weight=1.,matching_weight=.25,no_hardguards=True,no_radialshrink=True)
    base.atomic_json(directory/'manifest.json',manifest);base.atomic_json(directory/'schedules.json',base.read(source/'schedules.json'))
    delta,report=fit(theta,positive,preservation,matching,normal,ridge=RIDGE)
    allcases=[]
    for r in rows:
        with np.load(r['feature_path']) as f:X=f['X'];actions=f['actions']
        allcases.append(dict(seed=r['seed'],group='left' if base.leftward(r['seed']) else 'right',X=X,actions=actions))
    report['whole_baseline_distribution']=diagnostics(theta,delta,allcases);base.atomic_json(directory/'fit-report.json',report);print(json.dumps(dict(fit=report)),flush=True)
    proposal=theta+delta;base.save_array(directory/'proposal_theta.npy',proposal);base.save_array(directory/'proposal.npy',proposal[:,:214]@np.load(directory/'old-transform.npy')+proposal[:,214:]@np.load(directory/'extra-transform.npy'));base.save_array(directory/'applied_delta.npy',delta)
    native=directory/'native_candidate';native.mkdir();(native/'episodes').mkdir();(native/'features').mkdir()
    for source_name,name in [('proposal.npy','carry.npy'),('proposal_theta.npy','carry_theta.npy'),('search-transform.npy','search-transform.npy'),('radial-cap.json','radial-cap.json')]:(native/name).write_bytes((directory/source_name).read_bytes())
    start=time.monotonic();deadline=min(overall,start+300);base.atomic_json(directory/'native-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=start,deadline_monotonic=deadline));outcomes=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        pending=iter(enumerate(r['seed'] for r in rows));active={}
        def fill():
            while len(active)<4 and time.monotonic()<deadline-2:
                try:i,seed=next(pending)
                except StopIteration:break
                active[pool.submit(collect_one,str(native),seed,i,deadline-1,False)]=(i,seed)
        fill()
        while active:
            done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for f in done:active.pop(f);outcomes.append(f.result())
            base.atomic_json(directory/'progress.json',dict(games=len(outcomes),policy_cameras=sum(r['transitions'] for r in outcomes)));fill()
    def more(values):
        a=aggregate([r['annotation'] for r in values]);later=a['own_returns']-a['first_incoming_returns'];opps=a['own_opportunities']-a['first_incoming_opportunities']
        return dict(later_returns=later,later_opportunities=opps,later_fraction=later/opps if opps else None,tail={str(k):sum(r['report']['hits']>=k for r in values) for k in range(1,8)},timeouts=sum(r['report']['truncations'] for r in values),opponent_ended=sum(r['report']['wins'] for r in values))
    summary=dict(status='complete' if len(outcomes)==96 and all(r['complete'] for r in outcomes) else 'bounded_partial',baseline=aggregate([r['annotation'] for r in rows]),candidate=aggregate([r['annotation'] for r in outcomes]),baseline_more=more(rows),candidate_more=more(outcomes),
        train_transition_cameras=sum(r['transitions'] for r in outcomes),initial_cameras=sum(r['initial_cameras'] for r in outcomes),wall_seconds=time.monotonic()-began,
        source_frozen=manifest['files']==pins() and manifest['source']==base.source_signature(),ridge=RIDGE,proposal_sha256=base.sha(directory/'proposal.npy'),carry_immutable=base.sha(directory/'carry.npy')==manifest['carry_sha256'],no_fresh_evaluation=True,no_automatic_promotion=True)
    base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--data-source',default='/workspace/m1b-experiments/pong-rally-aggregate-20261027-r1');a=p.parse_args();run(a.directory,a.data_source)
