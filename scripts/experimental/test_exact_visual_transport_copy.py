"""Focused copy/alias and already-captured transport checks; no native cameras."""
import copy
from pathlib import Path
import pickle

import pytest
import torch

from fly_connectome.control.codec import fingerprint
from fly_connectome.vision.interface import VisualStateEncoder
from exact_visual_transport_copy import ExactCopyEncoder, guarded_transport_copy, identical_encoder_code

FIXTURE=Path('/workspace/m1b-experiments/pong-runtime-profile-20261029-r1/captured_transport_representatives.pkl')


def test_repeated_tensor_is_owned_and_keeps_object_alias():
    value=torch.arange(12,dtype=torch.float32)
    source=[dict(first=value,second=value)]
    result=guarded_transport_copy(source)
    assert result[0]['first'] is result[0]['second']
    assert fingerprint(result)==fingerprint(copy.deepcopy(source))
    result[0]['first'][0]=-99
    assert value[0]==0


def test_distinct_shared_storage_views_use_alias_preserving_fallback():
    value=torch.arange(12,dtype=torch.float32)
    for second in (value[4:],torch.empty(0).set_(value.untyped_storage(),4,(8,),(1,))):
        result=guarded_transport_copy([dict(first=value[:8],second=second)])
        first,last=result[0]['first'],result[0]['second']
        assert first.untyped_storage()._cdata==last.untyped_storage()._cdata
        assert last.storage_offset()==4
        first[4]=-99
        assert last[0]==-99
        assert value[4]==4


def test_manual_grad_is_copied_and_owned_by_fallback():
    value=torch.arange(12,dtype=torch.float32)
    value.grad=torch.ones_like(value)
    result=guarded_transport_copy([dict(tensor=value)])[0]['tensor']
    assert not result.requires_grad
    assert torch.equal(result.grad,value.grad)
    result.grad[0]=-99
    assert value.grad[0]==1


def test_complete_encode_matches_archived_transport_without_mutating_inputs():
    if not FIXTURE.exists():
        pytest.skip('local captured transport proof fixture is unavailable')
    assert identical_encoder_code()
    with FIXTURE.open('rb') as stream:
        samples=pickle.load(stream)
    assert len(samples)==4
    for sample in samples:
        state,frame=sample['state'],sample['frame']
        source_hash=fingerprint((state,frame))
        left=VisualStateEncoder(64,64,.02);right=ExactCopyEncoder(64,64,.02)
        assert fingerprint(left.encode(state,frame))==fingerprint(right.encode(state,frame))
        assert fingerprint(vars(left))==fingerprint(vars(right))
        assert fingerprint((state,frame))==source_hash
