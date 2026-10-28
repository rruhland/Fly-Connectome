"""One bounded, fixed-baseline epsilon collection. No fitting or tree runtime."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import random
import time
import numpy as np
import scipy.sparse as sp
import torch
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.control.scaling import policy_features
from pong_fixed_rbf import FreshRBFEvaluator, load_readout
from rally_raw_capture import BOOKKEEPING, SparseRows
import run_practical_pong_v2 as base

ROOT=Path('/workspace/m1b-experiments')
DEST=ROOT/'pong-value-collection-20261028-r1'
MODEL=ROOT/'pong-rally-joint-output-20261028-r1/full'
BANK=ROOT/'pong-rally-TRAIN-bank-20261028-r7/admission.json'
EXPECTED='40069e2deae2b745b83b6c193389fa740450856b67443339b12f0dbb9bf3c438'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')


class EpsilonCollector(FreshRBFEvaluator):
    def __init__(self,readout,config,action_seed):
        self.exploration_rng=np.random.default_rng(action_seed)
        self.choices=[]
        super().__init__(readout,config)

    def _choose_policy(self):
        ix,x=policy_features(self.latest_encoded,self.scaler)
        greedy,q=self.rbf_readout.choose(ix,x)
        p=np.full(3,.2/3);p[greedy]+=.8
        action=int(self.exploration_rng.choice(3,p=p))
        self.choices.append((action,greedy,q.copy(),p))
        return action,q


def task(job,deadline,spec):
    began=time.monotonic()
    if began>=deadline:
        return dict(job,status='not_started_deadline',transitions=0,initials=0,collected_hits=0)
    random.seed(BOOKKEEPING);np.random.seed(BOOKKEEPING);torch.manual_seed(BOOKKEEPING)
    readout=load_readout(MODEL/'rbf-model.npz')
    config=PongEvaluationConfig(seed=job['seed'],episodes=1,max_episode_ticks=2400,
        opponent_speed=.65,action_hold_frames=1,policy_scaling=read(MODEL/'radial-cap.json'))
    session=EpsilonCollector(readout,config,job['action_seed'])
    if time.monotonic()>=deadline:
        return dict(job,status='not_started_deadline',transitions=0,initials=0,collected_hits=0)
    session._begin_episode();raw=SparseRows();raw.add(session);records=[]
    while not session.complete and session.transitions<spec['per_start_transition_cap'] and time.monotonic()<deadline:
        previous=session.latest_encoded
        old_seconds=session.physics_seconds
        old_ticks=session.env.total_ticks
        row=session.step(check_frozen=False)
        assert session.latest_encoded is not previous, 'successor must be newly observed, including terminal camera'
        elapsed=session.physics_seconds-old_seconds
        ticks=session.env.total_ticks-old_ticks
        assert ticks==row['physics_ticks'] and abs(elapsed-row['elapsed_seconds'])<1e-12
        assert abs(elapsed-ticks/120)<1e-12
        records.append(dict(action=row['action'],hits=row['hits'],terminated=row['terminated'],
            truncated=row['truncated'],elapsed_seconds=row['elapsed_seconds'],physics_ticks=ticks))
        raw.add(session)
    session.verify_frozen()
    matrix=raw.matrix();n=len(records)
    assert matrix.shape==(n+1,4389) and n==session.transitions==len(session.choices)
    stem=f"episode_{job['index']:02d}_{job['seed']}"
    raw_path=DEST/(stem+'-raw-next.npz');score_path=DEST/(stem+'-transitions.npz')
    sp.save_npz(raw_path,matrix)
    choices=session.choices
    np.savez_compressed(score_path,
        actions=np.asarray([r['action'] for r in records],dtype=np.int64),
        greedy=np.asarray([r[1] for r in choices],dtype=np.int64),
        q=np.asarray([r[2] for r in choices],dtype=float).reshape(n,3),
        p=np.asarray([r[3] for r in choices],dtype=float).reshape(n,3),
        hits=np.asarray([r['hits'] for r in records],dtype=np.int64),
        terminated=np.asarray([r['terminated'] for r in records],dtype=bool),
        truncated=np.asarray([r['truncated'] for r in records],dtype=bool),
        elapsed_seconds=np.asarray([r['elapsed_seconds'] for r in records]),
        physics_ticks=np.asarray([r['physics_ticks'] for r in records],dtype=np.int64))
    collected_hits=sum(r['hits'] for r in records)
    report=session.summary()
    if session.complete:
        assert collected_hits==sum(r['hits'] for r in report['rallies'])
    result=dict(job,status='natural_or_environment_end' if session.complete else
        ('per_start_camera_cap' if n>=spec['per_start_transition_cap'] else 'deadline_partial'),
        transitions=n,initials=session.initial_cameras,collected_hits=collected_hits,
        natural_termination=bool(records and records[-1]['terminated']),
        environment_timeout=bool(records and records[-1]['truncated']),
        every_successor_saved=True,terminal_successor_saved_but_bootstrap_irrelevant=True,
        raw_feature_file=str(raw_path),transitions_file=str(score_path),
        raw_sha256=sha(raw_path),transitions_sha256=sha(score_path),
        physics_ticks=sum(r['physics_ticks'] for r in records),
        physics_seconds=sum(r['elapsed_seconds'] for r in records),
        nongreedy_choices=sum(r[0]!=r[1] for r in choices),
        report=report,wall_seconds=time.monotonic()-began,parameters_frozen=True)
    write(DEST/(stem+'.json'),result)
    return result


def prepare():
    assert not DEST.exists();DEST.mkdir()
    assert sha(MODEL/'rbf-model.npz')==EXPECTED
    rng=np.random.default_rng(2026102884)
    admitted=[r['seed'] for r in read(BANK)['bank']]
    phases={phase:sorted(s for s in admitted if base.leftward(s)==(phase=='left'))
            for phase in ('left','right')}
    selected={phase:rng.choice(values,16,replace=False).tolist() for phase,values in phases.items()}
    jobs=[]
    for i in range(16):
        for phase in ('left','right'):
            jobs.append(dict(index=len(jobs),seed=int(selected[phase][i]),phase=phase,
                action_seed=int(rng.integers(1,2**32,dtype=np.uint64))))
    spec=dict(purpose='action_diverse_frozen40069_TRAIN_only_collection_no_fit',
        jobs=jobs,epsilon=.2,greedy_probability=.8+.2/3,other_probability=.2/3,
        native_seconds=120,task_reserve_seconds=2,maximum_transition_cameras=20000,
        per_start_transition_cap=625,workers=4,unchanged_primitive_cadence=True,
        source_selection='32uniform fixed admitted TRAIN starts;16perphase; no outcome filtering',
        schedule_seed=2026102884,existing_model=str(MODEL),bank=str(BANK),bank_sha256=sha(BANK),
        pinned_assets={str(MODEL/name):sha(MODEL/name) for name in ('rbf-model.npz','proposal.npy','radial-cap.json')},
        source=base.source_signature(),collector_sha256=sha(__file__),
        helper_hashes={name:sha(Path(__file__).with_name(name)) for name in
            ('pong_fixed_rbf.py','pong_fixed_pairwise.py','rally_raw_capture.py','run_practical_pong_v2.py')},
        sampling='save EVERY raw state and actual successor/action/hit/elapsed/terminal; no fitting or subsampling',
        hit_accounting='sum actual collected transition hits; partial summary completed-rally hits are not authoritative',
        partial='retain correctly labeled quota/deadline traces, including final actual successor; no automatic resume',
        no_fresh_data=True,no_backprop=True,no_model_changes=True,no_tree_implementation=True,
        no_automatic_promotion=True,implementation_seconds_separate_estimate=300)
    write(DEST/'spec.json',spec)
    print(json.dumps(dict(status='prepared_no_native',spec_sha256=sha(DEST/'spec.json'),jobs=len(jobs))),flush=True)


def run(approved):
    spec_path=DEST/'spec.json';spec=read(spec_path)
    assert approved==sha(spec_path) and not (DEST/'work-clock.json').exists()
    assert spec['source']==base.source_signature() and spec['collector_sha256']==sha(__file__)
    assert sha(BANK)==spec['bank_sha256']
    assert all(sha(p)==h for p,h in spec['pinned_assets'].items())
    assert all(sha(Path(__file__).with_name(n))==h for n,h in spec['helper_hashes'].items())
    began=time.monotonic();deadline=began+spec['native_seconds']
    write(DEST/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,
        deadline_monotonic=deadline,seconds=spec['native_seconds']))
    results=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        futures=[pool.submit(task,job,deadline-spec['task_reserve_seconds'],spec) for job in spec['jobs']]
        for f in as_completed(futures):
            results.append(f.result())
            write(DEST/'progress.json',dict(completed_jobs=len(results),
                transition_cameras=sum(r['transitions'] for r in results),
                initial_cameras=sum(r['initials'] for r in results),
                collected_hits=sum(r['collected_hits'] for r in results)))
    summary=dict(status='collection_complete' if len(results)==32 and all(r['transitions']>0 for r in results)
        else 'collection_bounded_partial',jobs=sorted(results,key=lambda r:r['index']),
        transition_cameras=sum(r['transitions'] for r in results),
        initial_cameras=sum(r['initials'] for r in results),
        collected_hits=sum(r['collected_hits'] for r in results),
        natural_terminations=sum(r.get('natural_termination',False) for r in results),
        environment_timeouts=sum(r.get('environment_timeout',False) for r in results),
        quota_or_deadline_partial=sum(r['status'] in ('per_start_camera_cap','deadline_partial') for r in results),
        all32_started=all(r['transitions']>0 for r in results),wall_seconds=time.monotonic()-began,
        source_frozen=spec['source']==base.source_signature(),
        assets_unchanged=all(sha(p)==h for p,h in spec['pinned_assets'].items()),
        no_fitting=True,no_tree_implementation=True,no_model_promotion=True)
    assert summary['transition_cameras']<=20000
    write(DEST/'summary.json',summary)
    print(json.dumps({k:v for k,v in summary.items() if k!='jobs'}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('operation',choices=['prepare','collect'])
    parser.add_argument('--approved-spec-sha256');args=parser.parse_args()
    if args.operation=='prepare':prepare()
    else:run(args.approved_spec_sha256)
