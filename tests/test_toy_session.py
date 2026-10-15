import copy
import numpy as np
import pytest
import torch
from fly_connectome.control.codec import fingerprint
from fly_connectome.control.durable import DurableRun
from fly_connectome.control.toy_session import ToySpec,ToySession


def test_registered_balanced_identical_schedules_without_running_them():
    from fly_connectome.control.fresh_evaluation import VisibleChoiceTask
    for task,seeds,episodes in [('immediate',range(2101,2104),200),('delayed',range(2201,2204),25)]:
        for seed in seeds:
            a=VisibleChoiceTask(ToySpec(task,seed,episodes,lam=.8))
            b=VisibleChoiceTask(ToySpec(task,seed,episodes,lam=0))
            np.testing.assert_array_equal(a.sides,b.sides)
            assert len(a.sides)==episodes
            assert abs(int(sum(a.sides==0))-int(sum(a.sides==1)))<=1
    for task,base in [('immediate',3101),('delayed',3201)]:
        for seed in range(base,base+3):
            env=VisibleChoiceTask(ToySpec(task,seed,20,mode='evaluate'))
            assert sum(env.sides==0)==sum(env.sides==1)==10


def test_immediate_complete_trajectories_and_weights_match_between_lambda_arms():
    torch.set_num_threads(1)
    a=ToySession(ToySpec('immediate',99021,4,lam=.8,diagnostic=True))
    b=ToySession(ToySpec('immediate',99021,4,lam=0.,diagnostic=True))
    assert [a.step() for _ in range(4)]==[b.step() for _ in range(4)]
    np.testing.assert_array_equal(a.learner.weights,b.learner.weights)
    assert a.complete and b.complete
    assert a.initial_cameras==b.initial_cameras==4


def test_toy_interrupted_resume_inside_delayed_episode(tmp_path):
    torch.set_num_threads(1)
    spec=ToySpec('delayed',99022,4,diagnostic=True)
    run=DurableRun(tmp_path/'toy.db',session=ToySession(spec),snapshot_transitions=7)
    for _ in range(10):run.step()
    assert run.session.env.elapsed_steps==2 and run.session.learner.pending_action is not None
    before=fingerprint(run.session)
    schedule=run.session.env.sides.copy()
    selected=run.session.env.selected
    run.close_without_checkpoint()
    resumed=DurableRun(tmp_path/'toy.db')
    assert resumed.replayed==3 and fingerprint(resumed.session)==before
    np.testing.assert_array_equal(resumed.session.env.sides,schedule)
    assert resumed.session.env.episode==1 and resumed.session.env.elapsed_steps==2
    assert resumed.session.env.selected==selected
    resumed.stop()


def test_toy_configuration_tampering_rejected():
    torch.set_num_threads(1)
    session=ToySession(ToySpec('delayed',99023,2,diagnostic=True))
    session.spec_hash='wrong'
    with pytest.raises(ValueError,match='configuration'):
        ToySession.validate_restored(session)


def test_evaluation_durable_restore_keeps_shared_perception_aliases(tmp_path):
    torch.set_num_threads(1)
    session=ToySession(ToySpec('delayed',99024,2,mode='evaluate',policy_kind='initial',diagnostic=True))
    run=DurableRun(tmp_path/'evaluation.db',session=session)
    run.step();run.step()
    before=fingerprint(run.session)
    run.close_without_checkpoint()
    resumed=DurableRun(tmp_path/'evaluation.db')
    assert resumed.replayed==2 and fingerprint(resumed.session)==before
    assert resumed.session.vision is resumed.session.evaluator.vision
    assert resumed.session.previous is resumed.session.evaluator.previous
    resumed.stop()


def test_pilot_manifest_exact_caps_and_only_registered_seeds():
    import sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'/'experimental'))
    from run_registered_toy_pilot import jobs
    manifest=jobs()
    assert len(manifest)==36 and len({row['id'] for row in manifest})==36
    specs=[ToySpec(**row['spec']) for row in manifest]
    for spec in specs:spec.validate();assert not spec.diagnostic
    assert sum(s.transition_budget for s in specs if s.mode=='learn')==2400
    assert sum(s.transition_budget for s in specs if s.mode=='evaluate')==2160
