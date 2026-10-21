"""Metadata/recovery boundaries; no new camera observations in these tests."""
import copy
import json
import sys
import time
from concurrent.futures import Future
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts/experimental'))
import run_online_vision_drift as study


def recipe():
    return dict(version='persistent-fixed-policy-drift-v1',
        seed_roots=dict(exposure=104729,probes=104743,prequential=104759),
        seed_tag=104761,bootstrap_seed=104773,
        wall_seconds=dict(total=100,exposure=20,probes=40,prequential=20,scoring_reserve=20))


def test_seed_expansion_is_disjoint_deterministic_and_excludes_actual_ids():
    spec=recipe();study.validate_recipe(spec)
    first=study.expand(spec,[])
    assert first==study.expand(copy.deepcopy(spec),[])
    assert len(set(sum(first.values(),[])))==192
    banned=first['probes'][:3]
    second=study.expand(spec,banned)
    assert not set(sum(second.values(),[]))&set(banned)
    assert not set(sum(second.values(),[]))&set(study.FINAL_SEEDS)


@pytest.mark.parametrize('mutation',[
    lambda r:r['seed_roots'].update(probes=r['seed_roots']['exposure']),
    lambda r:r['seed_roots'].update(probes=study.FINAL_SEEDS[0]),
    lambda r:r['wall_seconds'].update(total=10),
    lambda r:r['wall_seconds'].update(probes=float('nan')),
])
def test_recipe_rejects_colliding_or_unbounded_registration(mutation):
    spec=recipe();mutation(spec)
    with pytest.raises(ValueError):study.validate_recipe(spec)


def test_draft_guard_precedes_all_clocks_and_session_construction(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'DRAFT_ONLY',True)
    with pytest.raises(RuntimeError,match='unallocated'):
        study.supervise(tmp_path)
    assert not list(tmp_path.iterdir())


def test_returned_failed_cost_is_durable_and_frozen_clock_cannot_reset(tmp_path,monkeypatch):
    manifest=dict(recipe=recipe(),started_wall_ns=time.time_ns(),jobs=[])
    job=dict(id='sample')
    monkeypatch.setattr(study,'jobs_for',lambda *args:[job])
    monkeypatch.setattr(study.worker,'complete_report',lambda *args:None)
    class Pool:
        def submit(self,*args):
            future=Future();future.set_result(dict(ok=False,job='sample',error='declared task failure'))
            return future
    original=study.execute_phase(tmp_path,manifest,'exposure',Pool())
    assert original['failed'] and not original['complete']
    assert study.execute_phase(tmp_path,manifest,'exposure',None)==original
    path=tmp_path/'exposure-boundary.json'
    for field,value in [('failed',False),('started_wall_ns',original['started_wall_ns']+1)]:
        changed=dict(original,**{field:value});study.atomic_json(path,changed)
        with pytest.raises(ValueError,match='changed'):
            study.execute_phase(tmp_path,manifest,'exposure',None)
        study.atomic_json(path,original)


def test_authoritative_outcome_restores_only_identical_summary_without_scoring(tmp_path,monkeypatch):
    evidence=dict(files={'manifest.json':'fixed'},jobs={})
    summary=dict(status='complete',wall_seconds=95,overall_budget_met=True)
    study.atomic_json(tmp_path/'study-completion.json',dict(summary=summary,evidence=evidence))
    monkeypatch.setattr(study,'outcome_evidence',lambda *args:evidence)
    assert study.retained_outcome(tmp_path,{})==summary
    assert json.loads((tmp_path/'summary.json').read_text())==summary
    assert study.retained_outcome(tmp_path,{})==summary
    study.atomic_json(tmp_path/'summary.json',dict(summary,wall_seconds=1000))
    with pytest.raises(ValueError,match='summary changed'):
        study.retained_outcome(tmp_path,{})
    study.atomic_json(tmp_path/'summary.json',summary)
    monkeypatch.setattr(study,'outcome_evidence',lambda *args:dict(evidence,extra='mutated'))
    with pytest.raises(ValueError,match='evidence changed'):
        study.retained_outcome(tmp_path,{})


def test_unpinned_public_summary_is_rejected(tmp_path):
    study.atomic_json(tmp_path/'summary.json',dict(status='complete'))
    with pytest.raises(ValueError,match='authoritative'):
        study.retained_outcome(tmp_path,{})
