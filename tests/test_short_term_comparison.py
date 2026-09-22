import sys
from pathlib import Path
import numpy as np
import pytest
import hashlib
import torch

sys.path.insert(0,str(Path(__file__).parents[1]/'scripts'))
from short_term_learning import compare_targets


def test_internal_changes_are_reported_without_changing_scored_target():
    reference=np.zeros((12,4),np.float32);candidate=reference.copy()
    candidate[3,2]=1
    assert compare_targets(candidate,reference,np.array([True,False,False,False]),1)==1


@pytest.mark.parametrize('changed',[0,1])
def test_sensory_or_scored_target_changes_fail(changed):
    reference=np.zeros((12,4),np.float32);candidate=reference.copy()
    candidate[3,changed]=1
    with pytest.raises(AssertionError):
        compare_targets(candidate,reference,np.array([True,False,False,False]),1)


@pytest.mark.parametrize('change',['none','weights','stimulus','artifact','missing'])
def test_cached_evaluation_rejects_changed_inputs_or_artifacts(tmp_path,change):
    from short_term_learning import validate_cached
    path=tmp_path/'evaluation.npz';np.savez(path,prediction=np.zeros((2,1)))
    weights=torch.tensor([.5]);stimulus='fixed-stimulus'
    row=dict(weights_sha256=hashlib.sha256(weights.numpy().tobytes()).hexdigest(),
        stimulus_sha256=stimulus,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    if change=='weights':weights.add_(.1)
    elif change=='stimulus':stimulus='another-stimulus'
    elif change=='artifact':np.savez(path,prediction=np.ones((2,1)))
    elif change=='missing':path.unlink()
    if change=='none':validate_cached(row,path,weights,stimulus)
    else:
        with pytest.raises((AssertionError,FileNotFoundError)):
            validate_cached(row,path,weights,stimulus)
