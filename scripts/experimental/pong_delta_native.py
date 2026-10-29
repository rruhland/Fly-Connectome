"""Frozen planner native execution with one full replay and fixed spot checks.

All actual inputs/actions/rewards remain saved. Most games reconstruct controller
state from actual past actions, avoiding a second full expensive planning pass.
"""
from pathlib import Path
import time

import numpy as np
import scipy.sparse as sp

from fly_connectome.control.pong_evaluation import PongEvaluationConfig
from stateful_pong_comparison import StatefulEvaluator
from pong_fixed_rbf import FreshRBFEvaluator, load_readout
from pong_continuous_visual import canonical_scene, save_scenes
from rally_raw_capture import SparseRows
from return_metrics import ReturnCounter, native_count_status
from run_rally_option_memory import MODEL, EXPECTED, read, write, sha, seed_rngs
import pong_delta_policy_adapter as adapter


def fixed_audit_indices(records, limit=8):
    """Fixed index/first-hit schedule; never choose favorable policy outcomes."""
    n = len(records)
    if not n:
        return []
    selected = set(np.linspace(0, n - 1, min(limit, n), dtype=int).tolist())
    hit = next((i for i,r in enumerate(records) if r['hits']), None)
    if hit is not None and hit + 1 < n:
        selected.add(hit + 1)
    return sorted(selected)


def restore_actual_state(controller, records, index):
    controller.ordinal = index
    controller.past_hit = any(r['hits'] for r in records[:index])
    controller.pending_action = None
    history = np.zeros((16, 3), dtype=np.float64)
    previous = [r['action'] for r in records[max(0, index - 16):index]]
    if previous:
        history[-len(previous):] = np.eye(3)[previous]
    controller.history = history


def audit_trace(model, baseline, records, decisions, scenes, raw, full):
    n = len(records)
    if not (n == len(decisions) == len(scenes) == raw.shape[0]):
        raise ValueError('no silent omissions in planner audit')
    selected = set(range(n) if full else fixed_audit_indices(records))
    controller = adapter.new_controller(model)
    past_hit = False; changes = prehit = planned = 0
    for i,(scene,record,expected) in enumerate(zip(scenes,records,decisions)):
        lo, hi = raw.indptr[i:i + 2]
        action, q = baseline.choose(raw.indices[lo:hi], raw.data[lo:hi])
        if not np.array_equal(q, np.asarray(record['q'])) or action != expected['baseline_action']:
            raise AssertionError('saved baseline Q/action differs from actual raw features')
        if expected['ordinal'] != i or expected['phase'] != i % 5 or expected['past_hit'] != past_hit:
            raise AssertionError('public history/phase differs from actual trajectory')
        if not past_hit:
            prehit += 1
            if record['action'] != action or expected['mode'] != 'baseline_before_hit':
                raise AssertionError('baseline-before-actual-hit contract violated')
        if i in selected:
            if not full:
                restore_actual_state(controller, records, i)
            got, choice = adapter.choose(controller, scene, action, q)
            if got != record['action'] or choice != expected:
                raise AssertionError(('frozen planner replay mismatch', i))
        if full:
            adapter.observe(controller, record)
        changes += record['action'] != action
        planned += expected['mode'] in ('planned_primitive', 'baseline_predicted_best')
        past_hit |= bool(record['hits'])
    return dict(passed=True, rows=n, raw_baseline_q_rows=n,
        planner_replay_rows=len(selected), planner_replay_indices=sorted(selected),
        full_chronological_replay=bool(full), reconstructed_spotchecks=not full,
        changed_actions=int(changes), planned_cameras=int(planned), baseline_before_hit_cameras=prehit,
        macro_starts=0, held_cameras=0, new_cameras=0)


def native_task(model_path, directory, seed, index, deadline, candidate=True, full_audit=False):
    directory = Path(directory); began = time.monotonic(); seed_rngs()
    if time.monotonic() >= deadline:
        return dict(seed=seed, complete=False, not_started=True, transitions=0, initial_cameras=0)
    baseline = load_readout(MODEL / 'rbf-model.npz')
    config = PongEvaluationConfig(seed=seed, episodes=1, max_episode_ticks=2400,
        opponent_speed=.65, policy_scaling=read(MODEL / 'radial-cap.json'))
    model = adapter.load_model(model_path) if candidate else None
    session = StatefulEvaluator(baseline, model, adapter, config) if candidate else FreshRBFEvaluator(baseline, config)
    if time.monotonic() >= deadline:
        return dict(seed=seed, complete=False, not_started=True, transitions=0, initial_cameras=0)
    session._begin_episode(); game=session.env.game; old=game.step
    counter=ReturnCounter(); records=[]; raw=SparseRows(); scenes=[]
    def instrument(drive, **kwargs):
        return counter.step(game, old, drive, **kwargs)
    game.step=instrument
    try:
        while not session.complete and time.monotonic() < deadline:
            raw.add(session); scenes.append(canonical_scene(session.latest_encoded))
            records.append(session.step(check_frozen=False))
    finally:
        del game.step
    session.verify_frozen(); report=session.summary()
    last=report['rallies'][-1] if report['rallies'] else dict(truncated=False, terminated=False,
        reward=session.episode_return, seconds=session.episode_seconds)
    annotation=counter.summary(game, **{k:last[k] for k in ('truncated','terminated','reward','seconds')})
    stem=directory / f"{'candidate' if candidate else 'carry'}_{index:02d}"
    X=raw.matrix(); sp.save_npz(str(stem)+'-raw.npz',X); save_scenes(str(stem)+'-scenes.npz',scenes)
    decisions=session.decisions if candidate else []
    native_seconds=time.monotonic()-began; audit_began=time.monotonic()
    audit=audit_trace(model,baseline,records,decisions,scenes,X,full_audit) if candidate else dict(
        passed=True, rows=len(records), planner_replay_rows=0, new_cameras=0)
    session.verify_frozen()
    result=dict(seed=seed, complete=session.complete, report=report, annotation=annotation,
        records=records, decisions=decisions, full_stateful_audit=audit,
        audit_kind='full_one_designated_game_other_fixed_spotchecks' if candidate else 'baseline',
        transitions=session.transitions, initial_cameras=session.initial_cameras,
        model_sha256=sha(model_path) if candidate else None, baseline_sha256=EXPECTED,
        transport_kind=adapter.CONTRACT if candidate else 'visual-transport-v1',
        feature_file=str(stem)+'-raw.npz', structured_feature_file=str(stem)+'-scenes.npz',
        native_and_loading_seconds=native_seconds, audit_seconds=time.monotonic()-audit_began,
        wall_seconds=time.monotonic()-began)
    if report['complete']:
        assert native_count_status(annotation,report)['valid_terminal_counts']
    write(str(stem)+'.json',result)
    return result
