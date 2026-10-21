"""Real learned-memory dose export/recovery; no candidate training allocation."""
import json
import sqlite3
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import persistent_vision_worker as worker
import run_online_vision_drift as study
from fly_connectome.control.online_vision import PersistentVisionSession
from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from fly_connectome.vision import VisualStateEncoder


def fixture_job(tmp_path):
    torch.set_num_threads(1)
    dimension=VisualStateEncoder(64,64,.02).dimension
    weights=np.random.default_rng(497103).normal(0,.001,(3,dimension))
    np.save(tmp_path/'fixed.npy',weights)
    cfg=PongEvaluationConfig(497107,episodes=3,max_episode_ticks=25,opponent_speed=.65)
    seeds=[497107,497131,497137]
    initial=PersistentVisionSession(weights,cfg,learn_enabled=True,episode_seeds=seeds)
    worker.save_snapshot(tmp_path/'before.pkl',initial.retained_vision())
    job=dict(id='stream',policy='fixture',purpose='exposure',phase='exposure',
        config=asdict(cfg),episode_seeds=seeds,learn_enabled=True,weights_from='fixed.npy',
        snapshot_from='before.pkl',camera_dose=17,
        input_hashes={name:worker.sha(tmp_path/name) for name in ('fixed.npy','before.pkl')})
    manifest=dict(source=worker.source_signature(),runtime=worker.runtime_signature(),
        model_hash=worker.MODEL_HASH,vision_configuration=worker.VISION_CONFIGURATION,
        worker_sha256=worker.sha(worker.__file__),pinned_sources=study.pinned_sources(),jobs=[job])
    worker.atomic_json(tmp_path/'manifest.json',manifest)
    worker.atomic_json(tmp_path/'stream-spec.json',job)
    deadline_ns=time.time_ns()+600_000_000_000
    worker.atomic_json(tmp_path/'exposure-boundary.json',dict(deadline_wall_ns=deadline_ns))
    return job,deadline_ns/1e9


def rows(path):
    with sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True) as connection:
        return connection.execute('SELECT seq,record,state_hash,chain_hash FROM replay_samples ORDER BY seq').fetchall()


@pytest.mark.parametrize('crash_point',['before_snapshot','after_snapshot_before_report'])
def test_completed_mid_rally_dose_export_recovers_without_new_camera(tmp_path,monkeypatch,crash_point):
    job,deadline=fixture_job(tmp_path)
    original_save=worker.save_snapshot;original_json=worker.atomic_json
    def save(path,value):
        if path.name=='stream-retained.pkl' and crash_point=='before_snapshot':
            raise OSError('dose export crash')
        return original_save(path,value)
    def write(path,value):
        if path.name=='stream.json' and crash_point=='after_snapshot_before_report':
            raise OSError('report export crash')
        return original_json(path,value)
    monkeypatch.setattr(worker,'save_snapshot',save);monkeypatch.setattr(worker,'atomic_json',write)
    with pytest.raises(OSError,match='export crash'):
        worker.execute_job(str(tmp_path),job,deadline)
    before=rows(tmp_path/'stream.db');assert len(before)==17
    snapshot_path=tmp_path/'stream-retained.pkl'
    retained_bytes=None if not snapshot_path.exists() else snapshot_path.read_bytes()
    monkeypatch.setattr(worker,'save_snapshot',original_save);monkeypatch.setattr(worker,'atomic_json',original_json)
    cost=worker.execute_job(str(tmp_path),job,deadline)
    assert cost['ok'] and cost['complete']
    assert cost['new_camera_transitions']==cost['new_initial_images']==cost['new_visual_observations']==0
    assert cost['replayed_camera_transitions']==cost['replayed_visual_observations']==0
    assert before==rows(tmp_path/'stream.db')
    if retained_bytes is not None:assert snapshot_path.read_bytes()==retained_bytes
    report=worker.complete_report(tmp_path,job)
    assert report['complete'] and report['partial_final_episode'] and not report['episode_session_complete']
    assert report['session']['transitions']==17 and report['session']['episodes']==1
    assert report['session']['initial_cameras']==2
    assert report['raw_hits_including_partial_episode']==report['session']['hits']+report['partial_episode']['hits']
    assert report['retained_snapshot']['cumulative_learning_observations']==19
    snapshot=worker.load_snapshot(tmp_path,'stream-retained.pkl')
    assert snapshot.learning_observations==19
    assert any(len(model.long.keys)>0 for model in snapshot.vision.dynamics.values())
    original=worker.complete_report(tmp_path,job)
    assert worker.execute_job(str(tmp_path),job,deadline)['new_camera_transitions']==0
    assert worker.complete_report(tmp_path,job)==original
    # All artifacts referenced by a job must remain explicitly pinned.
    bad=dict(job,input_hashes={'fixed.npy':job['input_hashes']['fixed.npy']})
    manifest=json.loads((tmp_path/'manifest.json').read_text());manifest['jobs']=[bad]
    original_json(tmp_path/'manifest.json',manifest);original_json(tmp_path/'stream-spec.json',bad)
    with pytest.raises(ValueError,match='content pinned'):
        worker.validate_inputs(tmp_path,bad)
