"""Actual capped raw-input replay for a single same-data optimizer comparison.

Only existing ordinary TRAIN roll-ins and own reward-earned action sequences are
replayed. No physics values enter features or action targets. Sparse rows are the
exact policy_features output; sharing a prefix avoids paying for it twice.
"""
import argparse, json, multiprocessing as mp, random, time
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import torch
from fly_connectome.control.pong_evaluation import FreshPongEvaluator, PongEvaluationConfig
from fly_connectome.control.scaling import policy_features
from pong_action_credit import step_action
import run_practical_pong_v2 as base

SOURCE=base.EXPERIMENTS/'pong-rally-aggregate-20261027-r1'
EXPECTED='fe132e0dd2466494142ef4578b187e6856341dc1bb4a3803456ca16e7703b625'
BOOKKEEPING=904271833

class SparseRows:
    def __init__(self): self.data=[]; self.indices=[]; self.indptr=[0]
    def add(self, session):
        indices, values=policy_features(session.latest_encoded,session.scaler)
        self.indices.extend(indices.tolist()); self.data.extend(values.tolist()); self.indptr.append(len(self.data))
    def matrix(self):
        return sp.csr_matrix((np.asarray(self.data,dtype=np.float64),np.asarray(self.indices,dtype=np.int32),np.asarray(self.indptr,dtype=np.int64)),shape=(len(self.indptr)-1,4389))

def start(weights, config):
    random.seed(BOOKKEEPING); np.random.seed(BOOKKEEPING); torch.manual_seed(BOOKKEEPING)
    session=FreshPongEvaluator(weights,PongEvaluationConfig(**config)); session._begin_episode(); return session

def task(directory, job, deadline):
    began=time.monotonic(); directory=Path(directory); W=np.load(directory/'carry.npy'); buffer=SparseRows(); cameras=0; outputs=[]; qerror=0.; hit_indices=[]
    if job['kind']=='current':
        receipt=base.read(job['receipt']); session=start(W,receipt['config']); query=job.get('query'); stop=job['stop']
        for t in range(stop):
            if time.monotonic()>=deadline: break
            buffer.add(session); q=session._choose_policy()[1]; record=session.step(check_frozen=False); cameras+=1
            cached=receipt['records'][t]
            assert record==cached, ('native replay mismatch',job['seed'],t)
            if record['hits']: hit_indices.append(t)
            # Reconstruct using the actual sparse input rather than a projected inverse.
            lo,hi=buffer.indptr[-2:]; error=np.max(abs(W[:,np.asarray(buffer.indices[lo:hi],dtype=int)]@np.asarray(buffer.data[lo:hi])-q)); qerror=max(qerror,float(error))
        complete=cameras==stop
        current_path=f"capture/current_{job['seed']}.npz"; sp.save_npz(directory/current_path,buffer.matrix())
        outputs.append(dict(kind='current',seed=job['seed'],path=current_path,rows=cameras,complete=complete,hit_indices=hit_indices))
        if complete and query is not None:
            memory=job['memory']; assert stop==query
            with np.load(memory['feature_path']) as f: actions=f['actions'].astype(int)
            assert len(actions)>query and np.array_equal(actions[:query],[r['action'] for r in receipt['records'][:query]])
            actual_hits=len(hit_indices)
            for t in range(query,len(actions)):
                if time.monotonic()>=deadline: break
                buffer.add(session); r=step_action(session,int(actions[t])); cameras+=1
                if r['hits']: actual_hits+=1
            witness_complete=len(buffer.indptr)-1==len(actions)
            if witness_complete: assert actual_hits==memory['target_hits'] and r['hits']>0
            memory_path=f"capture/memory_{job['seed']}.npz"; sp.save_npz(directory/memory_path,buffer.matrix())
            outputs.append(dict(kind='memory',seed=job['seed'],path=memory_path,rows=len(buffer.indptr)-1,complete=witness_complete,actual_hits=actual_hits,source=memory['source']))
    else:
        memory=job['memory']; session=start(W,job['config'])
        with np.load(memory['feature_path']) as f: actions=f['actions'].astype(int)
        actions=actions[:memory.get('slice_end',len(actions))]; actual_hits=0
        for t,a in enumerate(actions):
            if time.monotonic()>=deadline: break
            buffer.add(session); r=step_action(session,int(a)); cameras+=1
            if r['hits']: actual_hits+=1
        complete=cameras==len(actions)
        if complete: assert actual_hits==memory['target_hits'] and r['hits']>0
        path=f"capture/memory_{job['seed']}.npz"; sp.save_npz(directory/path,buffer.matrix())
        outputs.append(dict(kind='memory',seed=job['seed'],path=path,rows=cameras,complete=complete,actual_hits=actual_hits,source=memory['source']))
    session.verify_frozen(); assert qerror<1e-10
    result=dict(seed=job['seed'],kind=job['kind'],outputs=outputs,cameras=cameras,initials=session.initial_cameras,maximum_q_error=qerror,wall_seconds=time.monotonic()-began,weights_unchanged=True)
    base.atomic_json(directory/'capture_receipts'/f"{job['kind']}_{job['seed']}.json",result); return result

def stage(directory):
    directory=Path(directory); assert not directory.exists(); directory.mkdir()
    (directory/'capture').mkdir(); (directory/'capture_receipts').mkdir()
    for name in ('carry.npy','radial-cap.json'):(directory/name).write_bytes((SOURCE/name).read_bytes())
    assert base.sha(directory/'carry.npy')==EXPECTED
    m=base.read(SOURCE/'manifest.json'); rows=[dict(base.read(p),receipt=p) for p in m['cached_current_receipts']+m['new_current_receipts']]
    memories=base.read(SOURCE/'reward-memory-index.json')['memories']; byseed={r['seed']:r for r in rows}; new={m['seed']:m for m in memories if m['source']=='current_burst_success'}
    witnesses={w['seed']:w for w in base.read(SOURCE/'discovery-result.json')['witnesses']}; jobs=[]
    for r in rows:
        hits=[i for i,v in enumerate(r['records']) if v['hits']]; query=witnesses[r['seed']]['transition'] if r['seed'] in new else None
        stop=max((hits[-1]+1 if hits else 0),query or 0)
        if stop or query is not None: jobs.append(dict(kind='current',seed=r['seed'],receipt=r['receipt'],stop=stop,query=query,memory=new.get(r['seed'])))
    for memory in memories:
        if memory['source']!='current_burst_success': jobs.append(dict(kind='old_memory',seed=memory['seed'],memory=memory,config=byseed[memory['seed']]['config']))
    assert len(new)==24 and len(memories)==37
    manifest=dict(source=base.source_signature(),capture_source_sha256=base.sha(Path(__file__)),carry_sha256=EXPECTED,data_source=str(SOURCE),raw_dimension=4389,optimizer_parameters=8778,rows=rows,memories=memories,jobs=jobs,
        work_seconds=600,no_new_search=True,no_holdout_fitted=True,zero_residual_mapping='literal carried raw W copy; additive raw residual, no projection remap',
        capture='actual policy_features indices/values after unchanged radial cap; shared native prefix; recorded own successful actions only',
        geometry='raw coordinate L2 penalty differs from DCT metric; same frozen normalization and ridge1e-6, not a pure capacity causal test')
    base.atomic_json(directory/'manifest.json',manifest); print(json.dumps(dict(staged=True,jobs=len(jobs),memories=len(memories),new_observations=0)),flush=True)

def run(directory):
    directory=Path(directory); m=base.read(directory/'manifest.json'); assert m['source']==base.source_signature() and m['capture_source_sha256']==base.sha(Path(__file__))
    assert not (directory/'work-clock.json').exists(); began=time.monotonic(); deadline=began+600
    base.atomic_json(directory/'work-clock.json',dict(started_wall_ns=time.time_ns(),started_monotonic=began,deadline_monotonic=deadline)); results=[]
    with ProcessPoolExecutor(max_workers=4,mp_context=mp.get_context('spawn'),initializer=base.initialize) as pool:
        pending=iter(m['jobs']);active={}
        def fill():
            while len(active)<4 and time.monotonic()<deadline-2:
                try:job=next(pending)
                except StopIteration:break
                active[pool.submit(task,str(directory),job,deadline-1)]=job
        fill()
        while active:
            done,_=wait(active,timeout=1,return_when=FIRST_COMPLETED)
            for f in done:active.pop(f);results.append(f.result())
            base.atomic_json(directory/'progress.json',dict(phase='raw_capture',jobs=len(results),cameras=sum(r['cameras'] for r in results)));fill()
    outputs=[o for r in results for o in r['outputs']]; summary=dict(status='complete' if len(results)==len(m['jobs']) and all(o['complete'] for o in outputs) else 'bounded_partial',outputs=outputs,transition_cameras=sum(r['cameras'] for r in results),initial_cameras=sum(r['initials'] for r in results),maximum_q_error=max(r['maximum_q_error'] for r in results),wall_seconds=time.monotonic()-began,source_frozen=m['source']==base.source_signature() and m['capture_source_sha256']==base.sha(Path(__file__)))
    base.atomic_json(directory/'capture-summary.json',summary);print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['stage','capture']);p.add_argument('directory');a=p.parse_args()
    (stage if a.operation=='stage' else run)(a.directory)
