"""Bounded crash/replay proof: fresh processes and real active learning state."""
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.session import LearningSession


def worker(path,stop,mode):
    torch.set_num_threads(1)
    if mode=='fresh' or mode=='crash':
        random.seed(33);np.random.seed(33);torch.manual_seed(33)
        session=LearningSession(max_episode_ticks=31)
        session.env.game.ball[0]=torch.tensor([.999,.8])
        session.env.game.ball_velocity[0]=torch.tensor([.55,0.])
        session.vision.reset_state();session.encoder.reset_state();session.previous.zero_()
        session.latest_encoded=session.observe(session.env.game.render(64,64)[0])
        session.learner.reset_episode()
        session.learner.start(session.latest_encoded['indices'].numpy(),session.latest_encoded['values'].numpy())
        run=DurableRun(path,session=session,snapshot_transitions=7)
    else:
        run=DurableRun(path)
    replayed=run.replayed
    while run.session.transitions < stop:
        run.step()
    if mode=='crash':
        # Kill during an uncommitted write after all requested transitions committed.
        # Last complete checkpoint is at 7, with committed tail 8 and 9.
        run.db.execute('BEGIN IMMEDIATE')
        run.db.execute("UPDATE replay_samples SET state_hash='uncommitted corruption'")
        os._exit(23)
    result=dict(count=run.session.transitions,hash=fingerprint(run.session),replayed=replayed,
                nonzero_weights=int(np.count_nonzero(run.session.learner.weights)),
                active_traces=int(np.count_nonzero(run.session.learner.traces)),
                pending_action=run.session.learner.pending_action,
                artifacts=run.db.execute('SELECT count(*) FROM replay_artifacts').fetchone()[0],
                bank_bytes=len(run.codec.artifact),snapshot_bytes=len(run.codec.dumps(run.session)),
                timing=run.timing_summary())
    run.stop()
    print(json.dumps(result))


def proof(directory):
    directory.mkdir(parents=True,exist_ok=False)
    started=time.perf_counter()
    def call(name,stop,mode):
        r=subprocess.run([sys.executable,__file__,'--db',str(directory/name),'--stop',str(stop),'--mode',mode],
                         capture_output=True,text=True,timeout=max(.01,180-(time.perf_counter()-started)))
        if mode=='crash':
            assert r.returncode==23,r.stderr
            return None
        if r.returncode:raise RuntimeError(r.stderr)
        return json.loads(r.stdout)
    full=call('full.db',16,'fresh')
    call('split.db',9,'crash')
    resumed=call('split.db',16,'resume')
    assert full['hash']==resumed['hash']
    assert resumed['replayed']==2 and resumed['artifacts']==1
    assert resumed['nonzero_weights'] and resumed['active_traces']
    import sqlite3
    with sqlite3.connect(directory/'full.db') as a,sqlite3.connect(directory/'split.db') as b:
        assert a.execute('SELECT * FROM replay_samples ORDER BY seq').fetchall()==b.execute('SELECT * FROM replay_samples ORDER BY seq').fetchall()
    result=dict(status='passed',comparison='16 uninterrupted vs 9 + crash + replay 2 + 7',
                advanced_transitions=32,replayed_transitions=2,wall_seconds=time.perf_counter()-started,
                full=full,resumed=resumed)
    (directory/'report.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proof',type=Path);p.add_argument('--db',type=Path)
    p.add_argument('--stop',type=int);p.add_argument('--mode',choices=['fresh','resume','crash'])
    args=p.parse_args()
    if args.proof:proof(args.proof)
    elif args.db and args.stop and args.mode:worker(args.db,args.stop,args.mode)
    else:p.error('provide --proof or --db/--stop/--mode')
