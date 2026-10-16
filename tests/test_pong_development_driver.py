import importlib.util
from pathlib import Path
import numpy as np
import pytest


spec=importlib.util.spec_from_file_location('pong_driver',Path(__file__).parents[1]/'scripts/experimental/run_pong_development.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def test_declared_allocation_does_not_use_final_holdout():
    learning,evaluation=driver.jobs()
    assert len(learning)==6 and sum(job['transitions'] for job in learning)==36000
    assert len(evaluation)==15 and sum(job['config']['episodes'] for job in evaluation)==180
    assert {job['config']['seed'] for job in evaluation} == {driver.DEV_SEED}
    assert not set(driver.TRAIN_SEEDS).intersection(driver.FINAL_SEEDS)


def test_weight_export_recovers_boundary_crash_and_rejects_later_substitution(tmp_path):
    weights=np.array([[.1,.2]])
    driver.export_checkpoint(tmp_path,'run',2000,weights)
    np.testing.assert_array_equal(np.load(tmp_path/'run_at2000.npy'),weights)
    driver.export_checkpoint(tmp_path,'run',2000,weights)
    with pytest.raises(ValueError,match='disagree'):
        driver.export_checkpoint(tmp_path,'run',2000,weights+1)
    (tmp_path/'run_at2000.npy').unlink()
    with pytest.raises(ValueError,match='missing past checkpoint'):
        driver.export_checkpoint(tmp_path,'run',2001,weights)
    assert not (tmp_path/'run_at2000.npy').exists()


def test_declared_longer_checkpoints_recover_without_substituting_later_weights(tmp_path):
    weights=np.array([[.3,.4]])
    driver.export_checkpoint(tmp_path,'long',15000,weights,checkpoints=(15000,30000))
    np.testing.assert_array_equal(np.load(tmp_path/'long_at15000.npy'),weights)
    driver.export_checkpoint(tmp_path,'long',30000,weights+1,checkpoints=(15000,30000))
    (tmp_path/'long_at15000.npy').unlink()
    with pytest.raises(ValueError,match='missing past checkpoint'):
        driver.export_checkpoint(tmp_path,'long',30000,weights+1,checkpoints=(15000,30000))
