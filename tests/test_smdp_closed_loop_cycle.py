"""The append operation preserves prior data, actual windows and unknown tails."""
from pathlib import Path
import sys
import json

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'experimental'))
import run_smdp_closed_loop_cycle as cycle
from run_smdp_closed_loop_cycle import Appender
from pong_smdp_adaptive_forest import state_features, discount


def scene(y=0.):
    return np.array([[y, -.2, 0., 0., 1., 1.]])


def old_data():
    return dict(states=np.array([state_features(scene(), 0, False, np.ones(4)),
        state_features(scene(.01), 1, False, np.ones(4))]), state_valid=np.ones(2, bool),
        state_start=np.zeros(2, np.int32), scales=np.ones(4),
        source=np.array([0], np.int32), successor=np.array([1], np.int32), option=np.array([3], np.int8),
        elapsed=np.array([.02]), kind=np.array([0], np.int8), start=np.array([0], np.int32),
        cameras=np.array([1], np.int8), mc=np.array([np.nan]))


def trace(seed=7, hit=True, natural=False, successor=True):
    return dict(name='actual', family='new', seed=seed, ordinal=0, past_hit=False,
        scenes=[scene(i / 100) for i in range(4 if successor else 3)],
        actions=np.ones(3, np.int8), hits=np.array([False, hit, False]),
        natural=np.array([False, False, natural]), truncated=np.array([False, False, not natural]),
        elapsed=np.full(3, 2 / 120), ticks=np.full(3, 2), baseline=np.ones(3, np.int8))


def test_append_preserves_old_bytes_and_reuses_start_identity():
    old = old_data(); before = {key:value.tobytes() for key,value in old.items()}
    appender = Appender(old, {7:0})
    appender.add(trace())
    data = appender.finish()
    assert all(value.tobytes() == before[key] for key,value in old.items())
    assert appender.seed_map == {7:0}
    assert np.array_equal(data['states'][:2], old['states'])
    assert np.all(data['start'] == 0)
    assert data['states'][4, 28] == 1  # Actual first hit updates later state context.
    first_macro = np.flatnonzero((data['source'] == 2) & (data['option'] == 1))[0]
    assert data['kind'][first_macro] == 1
    assert data['elapsed'][first_macro] == pytest.approx(4 / 120)
    assert data['mc'][first_macro] == pytest.approx(float(discount(4 / 120)))


def test_all_valid_new_windows_retained_while_mc_censored_tail_stays_unknown():
    appender = Appender(old_data(), {7:0})
    appender.add(trace(seed=11, hit=False, natural=False, successor=False))
    data = appender.finish()
    assert appender.seed_map == {7:0, 11:1}
    new = data['start'] == 1
    assert new.sum() == 2  # Two real baseline one-camera successors; no fabricated hold.
    assert np.all(data['option'][new] == 3)
    assert np.isnan(data['mc'][new]).all()
    assert np.all(data['kind'][new] == 0)
    assert appender.receipts[0]['rejected']['missing_successor_baseline'] == 1


def test_verified_natural_final_needs_no_successor_and_is_not_a_censor():
    appender = Appender(old_data(), {7:0})
    appender.add(trace(seed=11, hit=False, natural=True, successor=False))
    data = appender.finish(); new = data['start'] == 1
    assert np.all(data['mc'][new] == 0)
    terminal = new & (data['kind'] == 2)
    assert terminal.sum() == 4  # Three actual constant-prefix macros and final base step.
    assert np.all(data['successor'][terminal] == -1)


def test_tick_timing_mismatch_is_rejected():
    appender = Appender(old_data(), {7:0}); actual = trace()
    actual['elapsed'][0] = .1
    with pytest.raises(ValueError, match='tick'):
        appender.add(actual)


def test_collection_must_finish_and_all_receipts_and_data_must_match(tmp_path, monkeypatch):
    monkeypatch.setattr(cycle, 'COLLECTION', tmp_path)
    with pytest.raises(RuntimeError, match='active'):
        cycle.validate_completed_collection()
    folder = tmp_path / 'traces'; folder.mkdir()
    scene_path = folder / 'site_00-raw-scenes.npz'; scene_path.write_bytes(b'scene')
    data_path = folder / 'site_00-transitions.npz'; data_path.write_bytes(b'data')
    receipt = dict(model_sha256=cycle.C0_SHA, transitions=3, scene_file=str(scene_path),
        transition_file=str(data_path), scene_sha256=cycle.sha(scene_path), transition_sha256=cycle.sha(data_path))
    receipt_path = folder / 'site_00.json'; receipt_path.write_text(json.dumps(receipt))
    entry = dict(model_sha256=cycle.C0_SHA, initial_cameras=1, transitions=3,
        receipt_file=str(receipt_path), receipt_sha256=cycle.sha(receipt_path))
    summary = dict(status='all64complete', source_frozen=True, assets_frozen=True,
        traces=[entry], started=1, transition_cameras=3)
    (tmp_path / 'summary.json').write_text(json.dumps(summary))
    assert cycle.validate_completed_collection()['transition_cameras'] == 3
    data_path.write_bytes(b'changed')
    with pytest.raises(ValueError, match='hash'):
        cycle.validate_completed_collection()
