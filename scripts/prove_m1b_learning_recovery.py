"""Bounded controlled learning continuation, not Pong competence evidence."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fly_connectome.control import Journal
from fly_connectome.control.session import LearningSession, model_bytes
from prove_m1b_frozen_recovery import fingerprint


def learned_vision(session):
    return (session.vision.observer.weights,session.vision.observer.bias,
            tuple(getattr(session.vision.state.memory,n) for n in ('keys','values','metric')),
            session.vision.dynamics)


def worker(path, stop, resume):
    torch.set_num_threads(1)
    started = time.perf_counter()
    with Journal(path) as journal:
        saved = journal.load()
        if resume:
            if saved is None:
                raise ValueError('missing continuation')
            session = LearningSession.restore(saved[1])
            assert saved[0] == session.transitions-1
        else:
            if saved is not None:
                raise ValueError('fresh journal required')
            random.seed(1101); np.random.seed(1101); torch.manual_seed(1101)
            session = LearningSession(max_episode_ticks=31)
            # Controlled score fixture, not a training curriculum or learning result.
            # Camera state is refreshed so the forced physical setup is observed.
            session.env.game.ball[0] = torch.tensor([.999,.8])
            session.env.game.ball_velocity[0] = torch.tensor([.55,0.])
            session.vision.reset_state(); session.encoder.reset_state(); session.previous.zero_()
            session.latest_encoded = session.observe(session.env.game.render(64,64)[0])
            session.learner.reset_episode()
            session.learner.start(session.latest_encoded['indices'].numpy(),session.latest_encoded['values'].numpy())
        if not session.transitions < stop <= 48:
            raise ValueError('bounded proof only')
        before = fingerprint(learned_vision(session))
        journal.put_artifact(model_bytes())
        while session.transitions < stop:
            if time.perf_counter()-started > 120:
                raise TimeoutError('worker time limit')
            record = session.step()
            # Exercise captured global RNGs in addition to the active learner RNG.
            record['rng_probe'] = [random.random(),float(np.random.random()),float(torch.rand(()))]
            journal.commit(session.transitions-1,record,session.snapshot())
        assert fingerprint(learned_vision(session)) == before
        assert np.any(session.learner.weights != 0)
        assert np.any(session.learner.traces != 0)
        assert session.learner.pending_action is not None
        report = dict(transitions=session.transitions,hash=fingerprint(session),
                      nonzero_weights=int(np.count_nonzero(session.learner.weights)),
                      active_traces=int(np.count_nonzero(session.learner.traces)),
                      pending_action=session.learner.pending_action,q_old=session.learner.q_old,
                      incoming_discount=session.learner.incoming_discount,
                      episodes=session.episodes,elapsed_seconds=time.perf_counter()-started)
    print(json.dumps(report))


def proof(directory):
    directory.mkdir(parents=True,exist_ok=False)
    started = time.perf_counter()
    def launch(name,stop,resume=False):
        command=[sys.executable,__file__,'--db',str(directory/name),'--stop',str(stop)]
        if resume: command.append('--resume')
        result=subprocess.run(command,capture_output=True,text=True,
                              timeout=max(.01,180-(time.perf_counter()-started)))
        if result.returncode: raise RuntimeError(result.stderr)
        return json.loads(result.stdout)
    full=launch('full.db',32)
    split=launch('split.db',16)
    resumed=launch('split.db',32,True)
    assert full['hash']==resumed['hash'], 'continuation mismatch'
    with Journal(directory/'full.db') as a, Journal(directory/'split.db') as b:
        samples=a.samples()
        assert samples==b.samples(), 'audit mismatch'
        assert any(x['terminated'] for x in samples)
        assert any(x['truncated'] for x in samples)
    report=dict(status='passed',comparison='32 vs 16+16 fresh-process resume',
                integration_transitions=64,wall_seconds=time.perf_counter()-started,
                full=full,split=split,resumed=resumed,
                purpose='controlled algebra/lifecycle/recovery diagnostic; no competence claim',
                vision_learning=False)
    (directory/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--proof',type=Path); p.add_argument('--db',type=Path)
    p.add_argument('--stop',type=int); p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.proof: proof(args.proof)
    elif args.db and args.stop: worker(args.db,args.stop,args.resume)
    else: p.error('--proof or --db and --stop required')
