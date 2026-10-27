"""Fresh32 ordinary TRAIN games under an earned rally controller; no fitting."""
import argparse,json,multiprocessing as mp,time
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED
from pathlib import Path
from run_rally_baseline_pong import collect_one,schedule
from return_metrics import aggregate
import run_practical_pong_v2 as base


def run(directory,carry):
    directory=Path(directory);carry=Path(carry);directory.mkdir(parents=True,exist_ok=True);assert not (directory/'work-clock.json').exists()
    for name in ('episodes','features'):(directory/name).mkdir()
    for source,name in [('proposal.npy','carry.npy'),('proposal_theta.npy','carry_theta.npy'),('search-transform.npy','search-transform.npy'),('radial-cap.json','radial-cap.json')]:(directory/name).write_bytes((carry/source).read_bytes())
    drawn=schedule();chosen=drawn['train'][:32];drawn.update(train=chosen,unused_reserved=drawn['train'][32:],carry_source=str(carry/'proposal.npy'))
    base.atomic_json(directory/'schedules.json',drawn);files={n:base.sha(Path(__file__).with_name(n)) for n in ('run_rally_train_extend.py','run_rally_baseline_pong.py','pong_action_credit.py')}
    base.atomic_json(directory/'manifest.json',dict(files=files,source=base.source_signature(),carry_source=str(carry/'proposal.npy'),carry_sha256=base.sha(directory/'carry.npy'),work_seconds=150,training_camera_cap=32000,initials_separate=True,purpose='32 genuinely new ordinary TRAIN starts; no heldout qualification/no fitting'))
    began=time.monotonic();base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began));deadline=began+150;results=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        pending=iter(enumerate(chosen));active={}
        def fill():
            while len(active)<4 and time.monotonic()<deadline-2:
                try:i,seed=next(pending)
                except StopIteration:break
                active[pool.submit(collect_one,str(directory),seed,i,deadline-1,False)]=(i,seed)
        fill()
        while active:
            done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for f in done:active.pop(f);results.append(f.result())
            base.atomic_json(directory/'progress.json',dict(games=len(results),policy_cameras=sum(r['transitions'] for r in results)));fill()
    summary=dict(status='complete' if len(results)==32 and all(r['complete'] for r in results) else 'bounded_partial',aggregate=aggregate([r['annotation'] for r in results]),train_transition_cameras=sum(r['transitions'] for r in results),initial_cameras=sum(r['initial_cameras'] for r in results),wall_seconds=time.monotonic()-began,no_fitting=True)
    base.atomic_json(directory/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('directory');p.add_argument('--carry',default='/workspace/m1b-experiments/pong-rally-margin-20261027-r1');a=p.parse_args();run(a.directory,a.carry)
