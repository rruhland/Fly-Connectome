"""Narrow declared first-contact fresh greedy evaluation contract."""
import copy
import pytest
from test_pong_development_driver import driver


def job():
    return dict(kind='evaluate',purpose='fresh_evaluation',evaluation_target='first-contact-v1',first_hit_cutoff=True,
        config=dict(seed=173193,episodes=1,policy_kind='trained',action_hold_frames=1,action_seed=None))


def test_declared_first_contact_acceptance_and_legacy_training_boundary():
    j=job();assert driver.validate_first_hit_job(j)
    j.pop('evaluation_target');j['purpose']='training_fitness';assert driver.validate_first_hit_job(j)
    j['first_hit_cutoff']=False;assert not driver.validate_first_hit_job(j)


@pytest.mark.parametrize('mutation',[
    lambda j:j.pop('evaluation_target'),lambda j:j.update(evaluation_target='other'),
    lambda j:j.update(first_hit_cutoff=False),lambda j:j.update(actor={}),lambda j:j.update(projected_greedy={}),
    lambda j:j['config'].update(policy_kind='random'),lambda j:j['config'].update(action_seed=19),
    lambda j:j['config'].update(action_hold_frames=2),lambda j:j['config'].update(blank_camera=True),
    lambda j:j['config'].update(skip_baseline_perception=True),lambda j:j.update(camera_intervention={}),
    lambda j:j['config'].update(episodes=2)])
def test_first_contact_rejects_unmatched_or_nonordinary_paths(mutation):
    j=job();mutation(j)
    with pytest.raises(ValueError,match='first-hit cutoff'):driver.validate_first_hit_job(j)
