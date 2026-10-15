"""Diagnostic seeds only: toy-specific fresh-process mid-delay recovery proof."""
import argparse,json,os,random,sqlite3,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.toy_session import ToySpec,ToySession


def worker(path,stop,mode):
    torch.set_num_threads(1)
    if mode=='resume': run=DurableRun(path)
    else:
        random.seed(991);np.random.seed(991);torch.manual_seed(991)
        run=DurableRun(path,session=ToySession(ToySpec('delayed',99031,4,diagnostic=True)),snapshot_transitions=7)
    replayed=run.replayed
    while run.session.transitions<stop:run.step()
    s=run.session
    result=dict(transitions=s.transitions,hash=fingerprint(s),replayed=replayed,
                episode=s.env.episode,delay_step=s.env.elapsed_steps,latched_side=s.env.selected,
                schedule=s.env.sides.tolist(),pending_action=s.learner.pending_action,
                rng_hash=fingerprint((s.learner.rng,s.env.rng,s.python_rng,s.numpy_rng,s.torch_rng)))
    if mode=='crash':
        print(json.dumps(result),flush=True)
        os._exit(29)
    run.stop();print(json.dumps(result))


def proof(path):
    path.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    def invoke(name,n,mode):
        r=subprocess.run([sys.executable,__file__,'--db',str(path/name),'--stop',str(n),'--mode',mode],
                         capture_output=True,text=True,timeout=max(.01,180-(time.perf_counter()-started)))
        assert r.returncode==(29 if mode=='crash' else 0),r.stderr
        return json.loads(r.stdout)
    full=invoke('full.db',18,'fresh');mid=invoke('split.db',10,'crash');resumed=invoke('split.db',18,'resume')
    assert mid['delay_step']==2 and mid['episode']==1 and mid['pending_action'] is not None
    assert resumed['replayed']==3 and full['hash']==resumed['hash'] and full['rng_hash']==resumed['rng_hash']
    with sqlite3.connect(path/'full.db') as a,sqlite3.connect(path/'split.db') as b:
        assert a.execute('SELECT * FROM replay_samples ORDER BY seq').fetchall()==b.execute('SELECT * FROM replay_samples ORDER BY seq').fetchall()
    result=dict(status='passed',new_transitions=36,replayed_transitions=3,wall_seconds=time.perf_counter()-started,
                full=full,midpoint=mid,resumed=resumed,diagnostic_seed=99031)
    (path/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--proof',type=Path);p.add_argument('--db',type=Path)
    p.add_argument('--stop',type=int);p.add_argument('--mode');a=p.parse_args()
    if a.proof:proof(a.proof)
    else:worker(a.db,a.stop,a.mode)
