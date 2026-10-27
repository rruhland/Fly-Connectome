"""Explicit fixed-head completion of missing/partial native comparison games.

No fitting, selection, new seeds or changed candidate. Completed receipts remain
untouched; partial receipts/features are archived byte-for-byte before reruns.
"""
import argparse,json,multiprocessing as mp,shutil,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from run_rally_incremental_cycle import current_rows
from run_rally_raw_optimizer import native_task,more
from return_metrics import aggregate
import run_practical_pong_v2 as base


def run(directory,seconds=180):
    directory=Path(directory);old=base.read(directory/'summary.json');assert old['status']=='bounded_partial';manifest=base.read(directory/'manifest.json');source=Path(manifest['source_root']);W,control=current_rows(source)
    assert base.sha(directory/'proposal.npy')==old['proposal_sha256'] and manifest['source']==base.source_signature()
    folder=directory/'fixed-head-completion';folder.mkdir();shutil.copyfile(directory/'summary.json',folder/'original-bounded-summary.json');existing={};archived=[]
    for p in sorted((directory/'native_candidate/episodes').glob('*.json')):
        r=base.read(p);assert r['weights_sha256']==old['proposal_sha256']
        if r['complete']:existing[r['seed']]=r;continue
        destination=folder/'discarded-partials'/p.relative_to(directory);destination.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,destination)
        feature=directory/r['feature_file'];target=folder/'discarded-partials'/Path(r['feature_file']);target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(feature,target)
        archived.append(dict(seed=r['seed'],receipt=str(destination),feature=str(target),receipt_sha256=base.sha(destination),feature_sha256=base.sha(target),paid_transition_cameras=r['transitions'],paid_initials=r['initial_cameras']))
    jobs=[(i,r['seed']) for i,r in enumerate(control) if r['seed'] not in existing]
    phase=dict(source=base.source_signature(),source_files={n:base.sha(Path(__file__).with_name(n)) for n in ('complete_rally_native.py','run_rally_raw_optimizer.py')},candidate_sha256=old['proposal_sha256'],seconds=seconds,jobs=jobs,kept_complete_games=len(existing),archived_partial_receipts=archived,previous_paid_transition_cameras=old['native_transition_cameras'],previous_paid_initials=old['native_initials'],new_fitting=False,new_seeds=False)
    base.atomic_json(folder/'manifest.json',phase);began=time.monotonic();deadline=began+seconds;base.atomic_json(folder/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline));new=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        pending=iter(jobs);active={}
        def fill():
            while len(active)<4 and time.monotonic()<deadline-2:
                try:i,seed=next(pending)
                except StopIteration:break
                active[pool.submit(native_task,str(directory),seed,i,deadline-1)]=(i,seed)
        fill()
        while active:
            done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for f in done:active.pop(f);new.append(f.result())
            base.atomic_json(folder/'progress.json',dict(new_games=len(new),new_transition_cameras=sum(r['transitions'] for r in new)));fill()
    values=list(existing.values())+new;byseed={r['seed']:r for r in values};baseline={r['seed']:r for r in control};complete=len(values)==96 and all(r['complete'] for r in values);violations=[r['seed'] for r in values if r['complete'] and r['report']['hits']<baseline[r['seed']]['report']['hits']]
    summary=dict(old,status='complete' if complete else 'bounded_partial_after_completion',candidate=aggregate([r['annotation'] for r in values]),candidate_more=more(values),guarded_native_return_violations=violations,
        repair_outcomes=[dict(r,candidate_hits=byseed[r['seed']]['report']['hits'] if r['seed'] in byseed else None) for r in old['repair_outcomes']],
        newly_executed_games=old['newly_executed_games']+len(new),native_transition_cameras=old['native_transition_cameras']+sum(r['transitions'] for r in new),native_initials=old['native_initials']+sum(r['initial_cameras'] for r in new),
        logical_transition_cameras=sum(r['transitions'] for r in values),fixed_head_completion=dict(seconds=seconds,wall_seconds=time.monotonic()-began,new_games=len(new),new_transition_cameras=sum(r['transitions'] for r in new),new_initials=sum(r['initial_cameras'] for r in new),partial_attempts_charged_and_preserved=True,clock=str(folder/'work-clock.json')),
        initial_allocation_wall_seconds=old['wall_seconds'],wall_seconds=time.monotonic()-base.read(directory/'work-clock.json')['started_monotonic'],source_frozen=old['source_frozen'] and phase['source']==base.source_signature() and phase['source_files']=={n:base.sha(Path(__file__).with_name(n)) for n in phase['source_files']})
    assert base.sha(directory/'proposal.npy')==phase['candidate_sha256']
    base.atomic_json(folder/'summary.json',summary);base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--seconds',type=int,default=180);a=p.parse_args();assert 0<a.seconds<=180;run(a.directory,a.seconds)
