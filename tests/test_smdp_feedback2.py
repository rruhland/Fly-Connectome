"""Admit every newly executed comparison, without duplicating cached games."""
from pathlib import Path
import json
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))
import run_smdp_feedback2 as feedback


def comparison_fixture(tmp_path, monkeypatch):
    comparison = tmp_path / 'comparison'; prior = tmp_path / 'prior'
    comparison.mkdir(); prior.mkdir()
    monkeypatch.setattr(feedback, 'COMPARISON', comparison)
    monkeypatch.setattr(feedback, 'PRIOR', prior)
    (prior / 'data-manifest.json').write_text(json.dumps({'added_traces':[]}))
    groups = [('matched32/c0', 24, feedback.C0_SHA), ('c1-screen8', 8, feedback.C1_SHA),
        ('matched32/c1', 24, feedback.C1_SHA)]
    first = None
    for folder, count, model in groups:
        destination = comparison / folder; destination.mkdir(parents=True)
        for index in range(count):
            scene_path = destination / f'candidate_{index:02d}-scenes.npz'
            np.savez_compressed(scene_path, entities=np.array([[0, 0, 0, 0, 1, 1]] * 2), indptr=[0, 1, 2])
            records = [dict(action=0, hits=0, terminated=False, truncated=False,
                elapsed_seconds=2 / 120, physics_ticks=2, q=[1., 0., 0.]),
                dict(action=0, hits=0, terminated=True, truncated=False,
                    elapsed_seconds=2 / 120, physics_ticks=2, q=[1., 0., 0.])]
            path = destination / f'candidate_{index:02d}.json'
            path.write_text(json.dumps(dict(model_sha256=model, records=records, complete=True,
                transitions=2, structured_feature_file=str(scene_path), seed=index)))
            if first is None:
                first = path
    (comparison / 'summary.json').write_text(json.dumps(dict(source_frozen=True,
        assets_frozen=True, new_transition_cameras=112)))
    return first, prior


def test_all_56_new_comparisons_and_no_cached_re_admission(tmp_path, monkeypatch):
    first, prior = comparison_fixture(tmp_path, monkeypatch)
    traces = list(feedback.comparison_traces())
    assert len(traces) == 56 and sum(len(t['actions']) for t in traces) == 112
    assert sum(t['family'].endswith('_c0') for t in traces) == 24
    assert sum(t['family'].endswith('_c1') for t in traces) == 32
    (prior / 'data-manifest.json').write_text(json.dumps({'added_traces':[{'files':[{'path':str(first)}]}]}))
    with pytest.raises(ValueError, match='cached'):
        list(feedback.comparison_traces())


def test_comparison_model_mismatch_is_rejected(tmp_path, monkeypatch):
    first, _ = comparison_fixture(tmp_path, monkeypatch)
    receipt = json.loads(first.read_text()); receipt['model_sha256'] = 'another-model'
    first.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match='model'):
        list(feedback.comparison_traces())
