"""Small diagnostic fixtures only; registered held-out schedules are never run."""
import copy
import random

import numpy as np
import pytest
import torch
from fly_connectome.control.codec import fingerprint, frozen_tensors
from fly_connectome.control.fresh_evaluation import (
    EvaluationConfig,FreshFrozenEvaluator,VisibleChoiceEvaluationTask,preserve_global_rngs)
from fly_connectome.control.sarsa import TrueOnlineSarsa
from fly_connectome.vision import VisualStateEncoder,load_default


def weights():
    return np.full((2,VisualStateEncoder(64,64,.02).dimension),.001)


def rng_hash():
    return fingerprint((random.getstate(),np.random.get_state(),torch.get_rng_state()))


def test_registered_seed_mapping_only_no_competence_runs():
    assert [EvaluationConfig.registered('immediate',i).seed for i in range(3)] == [3101,3102,3103]
    assert [EvaluationConfig.registered('delayed',i).seed for i in range(3)] == [3201,3202,3203]
    assert EvaluationConfig.registered('immediate',0).episodes == 20
    assert EvaluationConfig.registered('delayed',0).period == .02


def test_exact_fresh_initialization_and_pinned_prior():
    torch.set_num_threads(1)
    config = EvaluationConfig('delayed',99001,2)
    evaluator = FreshFrozenEvaluator(weights(),config)
    assert evaluator.task.episode == -1 and not evaluator.task.active
    assert evaluator.task.selected is None and evaluator.task.elapsed_steps == 0
    assert evaluator.encoder.sample is None and evaluator.encoder.history == {}
    assert evaluator.vision.state.tracker.frame == -1
    assert evaluator.vision.histories == {} and evaluator.vision.pending == {}
    assert not evaluator.previous.any() and evaluator.encoded is None
    assert evaluator.records == [] and evaluator.transitions == 0
    reference = load_default()
    holder = type('Holder',(),{})(); holder.vision=reference
    assert fingerprint(frozen_tensors(evaluator)) == fingerprint(frozen_tensors(holder))
    expected = np.random.default_rng(config.seed).permutation([0,1])
    np.testing.assert_array_equal(evaluator.task.sides,expected)
    evaluator.begin_episode()
    expected_frame = torch.zeros(64,64,dtype=torch.bool)
    col=(16,48)[int(expected[0])]
    expected_frame[15:18,col-1:col+2]=True
    assert torch.equal(evaluator.previous,expected_frame)
    assert evaluator.encoded['sample_period_seconds'] == .02
    assert evaluator.vision.state.tracker.frame == 0 and evaluator.encoder.sample == 0


def test_training_state_rng_isolated_and_same_seed_repeats_without_updates(monkeypatch):
    torch.set_num_threads(1)
    learner=TrueOnlineSarsa(2,weights().shape[1],weights=weights(),seed=777)
    learner.start([0],[1.])
    learner.advance(.5,.02,[1],[1.])  # Algebra fixture, not toy competence training.
    before=fingerprint(learner)
    global_before=rng_hash()
    first=FreshFrozenEvaluator(learner.frozen().weights,EvaluationConfig('delayed',99002,2))
    def forbidden(*args,**kwargs): raise AssertionError('learner update called in evaluation')
    monkeypatch.setattr(TrueOnlineSarsa,'advance',forbidden)
    expected=[first.step() for _ in range(2)]
    assert any(value!=0 for row in expected for value in row['q'])
    assert first.learned_hash()==first.initial_learned_hash
    # Deliberately unrelated task trajectory; second evaluator must start from seed alone.
    first.previous.fill_(True)
    first.vision.histories[999]=[]
    second=FreshFrozenEvaluator(learner.frozen().weights,EvaluationConfig('delayed',99002,2))
    actual=[second.step() for _ in range(2)]
    assert expected==actual
    assert fingerprint(learner)==before and rng_hash()==global_before
    copied=learner.weights.copy()
    second.policy.weights[:]=0
    np.testing.assert_array_equal(learner.weights,copied)


def test_every_episode_clears_all_transient_history():
    torch.set_num_threads(1)
    evaluator=FreshFrozenEvaluator(weights(),EvaluationConfig('immediate',99003,2))
    evaluator.step()
    evaluator.previous.fill_(True)
    evaluator.vision.histories[999]=[]
    evaluator.encoder.history[999]=(torch.ones(2),-100,None,None)
    evaluator.begin_episode()
    assert 999 not in evaluator.vision.histories and 999 not in evaluator.encoder.history
    assert evaluator.vision.state.tracker.frame==0 and evaluator.encoder.sample==0
    assert torch.equal(evaluator.previous,evaluator.task.render())


def test_delayed_visible_consequence_cadence_and_terminal_reward():
    env=VisibleChoiceEvaluationTask(EvaluationConfig('delayed',99004,2))
    initial=env.reset()
    selected=int(env.sides[0])
    frame,reward,terminal,dt=env.step(selected)
    assert reward==0 and not terminal and dt==.02
    assert torch.equal(frame[:20],initial[:20])
    assert frame[43:46].sum()==21
    for tick in range(2,9):
        next_frame,reward,terminal,dt=env.step(1-selected)
        assert torch.equal(next_frame,frame)  # Opposite command holds the latched side.
        assert dt==.02 and terminal==(tick==8)
        assert reward==(1. if tick==8 else 0.)
    assert env.elapsed_steps==8 and env.selected==selected


def test_rng_guard_restores_exception_path():
    before=rng_hash()
    with pytest.raises(RuntimeError):
        with preserve_global_rngs():
            random.random();np.random.random();torch.rand(())
            raise RuntimeError('injected')
    assert rng_hash()==before


def test_chunk_cap_validation_and_no_partial_reset():
    torch.set_num_threads(1)
    evaluator=FreshFrozenEvaluator(weights(),EvaluationConfig('delayed',99005,2))
    for count,seconds in [(201,180),(200,181),(0,1)]:
        with pytest.raises(ValueError): evaluator.run_chunk(count,seconds)
    first=evaluator.run_chunk(1,1)
    assert len(first['records'])==1 and not first['complete']
    assert evaluator.task.elapsed_steps==1
    second=evaluator.run_chunk(1,1)
    assert len(second['records'])==1 and evaluator.task.elapsed_steps==2
