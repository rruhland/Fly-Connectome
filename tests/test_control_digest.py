"""The faster encoding must retain complete content and graph-alias checks."""
import copy
import numpy as np
import pytest
import torch
from fly_connectome.control.codec import fingerprint


def test_order_and_primitive_identity_do_not_change_digest():
    first=''.join(['a long ', 'dictionary key'])
    second=''.join(['a long', ' dictionary key'])
    assert first == second and first is not second
    assert fingerprint({first:[first,second], 'x':1}) == fingerprint({'x':1,second:[second,second]})


def test_aliases_and_cycles_are_explicit():
    value=[torch.tensor([1.,2.])]
    assert fingerprint([value,value]) != fingerprint([value,copy.deepcopy(value)])
    cyclic=[];cyclic.append(cyclic)
    assert fingerprint(cyclic) == fingerprint(copy.deepcopy(cyclic))


@pytest.mark.parametrize('kind',['tensor','array'])
def test_value_dtype_shape_and_numpy_view_mutations_are_visible(kind):
    value=torch.arange(6,dtype=torch.float32) if kind=='tensor' else np.arange(6,dtype=np.float32)
    before=fingerprint(value)
    assert fingerprint(value.reshape(2,3)) != before
    assert fingerprint(value.double() if kind=='tensor' else value.astype(np.float64)) != before
    view=value.numpy() if kind=='tensor' else value.view()
    view[1]+=1  # Torch version counters do not see writes through a NumPy view.
    assert fingerprint(value) != before


def test_noncontiguous_empty_and_gradient_tensors():
    source=torch.arange(12.,requires_grad=True).reshape(3,4)
    value=source[:,::2]
    assert fingerprint(value) == fingerprint(value.detach().contiguous())
    assert fingerprint(torch.empty(0,3)) != fingerprint(torch.empty(3,0))
    assert fingerprint(np.empty((0,3))) != fingerprint(np.empty((3,0)))


def test_rng_and_bytes_changes_are_visible():
    rng=np.random.default_rng(9);before=fingerprint(rng);rng.random()
    assert fingerprint(rng) != before
    assert fingerprint([b'a',b'bc']) != fingerprint([b'ab',b'c'])


# Frozen scalar reference for the unchanged graph-v2 byte format.
from collections import deque
import hashlib, io, pickle
from fly_connectome.control.codec import GRAPH_DIGEST

def scalar_reference(value):
    """Complete canonical graph digest, including aliases and RNG state.

    Binary tensor contents avoid expanding the whole graph into hexadecimal JSON.
    Pickle only encodes the canonical primitive tree; explicit references above
    remove cycles, so disabling its memo prevents primitive object identity from
    changing the digest. This is a versioned hash format, never deserialized here.
    """
    seen = {}
    retained = []  # Keep temporary RNG state tensors alive while assigning alias IDs.
    def visit(obj):
        if obj is None or isinstance(obj, (str, bool, int, float)):
            return obj
        identity = id(obj)
        if identity in seen:
            return ['ref', seen[identity]]
        seen[identity] = len(seen)
        retained.append(obj)
        if isinstance(obj, torch.Tensor):
            return ['tensor', str(obj.dtype), list(obj.shape), obj.numpy(force=True).tobytes()]
        if isinstance(obj, np.random.Generator):
            return ['numpy-generator', visit(obj.bit_generator.state)]
        if isinstance(obj, np.ndarray):
            return ['array', str(obj.dtype), list(obj.shape), obj.tobytes()]
        if isinstance(obj, bytes):
            return ['bytes', obj]
        if isinstance(obj, torch.Generator):
            return ['torch-generator', visit(obj.get_state())]
        if isinstance(obj, dict):
            return ['dict', [[visit(k), visit(v)] for k, v in sorted(obj.items(), key=lambda kv: repr(kv[0]))]]
        if isinstance(obj, (list, tuple, deque)):
            return [type(obj).__name__, [visit(x) for x in obj]]
        if hasattr(obj, '__dict__'):
            fields = vars(obj)
            if '_variance_version' in fields:
                fields = dict(fields)
                # Torch version counters are process-local cache invalidation hints.
                fields['_variance_version'] = fields['_variance_version'] == fields['_variance_keys']._version
            return [type(obj).__module__+'.'+type(obj).__qualname__, visit(fields)]
        raise TypeError(type(obj))
    stream = io.BytesIO()
    writer = pickle.Pickler(stream, protocol=5)
    writer.fast = True
    writer.dump(visit(value))
    return hashlib.sha256(GRAPH_DIGEST.encode()+b'\0'+stream.getvalue()).hexdigest()


@pytest.mark.parametrize('dtype',[torch.bool,torch.int64,torch.float32,torch.float64,torch.complex64])
@pytest.mark.parametrize('container',[list,tuple,deque])
def test_bulk_tensor_nodes_match_frozen_scalar_reference(dtype,container):
    rows=[torch.ones((2,),dtype=dtype) for _ in range(40)]
    obj=[rows[3],container(rows+[rows[3],rows[0]]),container([rows[0]]*40)]
    assert fingerprint(obj)==scalar_reference(obj)
    before=fingerprint(obj)
    rows[17].numpy()[0]=0
    assert fingerprint(obj)!=before and fingerprint(obj)==scalar_reference(obj)


def test_bulk_noncontiguous_empty_mixed_and_subclass_fallback():
    rows=[torch.arange(12.,requires_grad=True).reshape(3,4)[:,::2] for _ in range(40)]
    for obj in [rows,[torch.empty(0,3) for _ in range(40)],rows+[torch.zeros(3)],
                rows+[torch.zeros(3,2,dtype=torch.float64)],
                [torch.ones(2,dtype=torch.complex64).conj() for _ in range(40)]]:
        assert fingerprint(obj)==scalar_reference(obj)
    class ModifiedNumpy(torch.Tensor):
        def numpy(self,*args,**kwargs):
            return super().numpy(*args,**kwargs)+1
    rows=[torch.zeros(2).as_subclass(ModifiedNumpy) for _ in range(40)]
    assert fingerprint(rows)==scalar_reference(rows)


@pytest.mark.parametrize('container',[list,tuple,deque])
def test_bulk_stack_validation_preserves_views_nan_bytes_and_external_aliases(container):
    bits=np.array([0,0x8000000000000000,0x7ff8000000000001,0xfff8000000000013],dtype=np.uint64)
    rows=[torch.from_numpy(bits.view(np.float64).copy()) for _ in range(40)]
    for values in (rows,[torch._neg_view(x) for x in rows],
                   [torch.complex(x,x).conj() for x in rows],
                   [x[::2] for x in rows],[x[:1].expand(3) for x in rows]):
        obj={'first':values[0],'sequence':container(values+[values[0]]),'last':values[-1]}
        assert fingerprint(obj)==scalar_reference(obj)
    obj={'numpy_alias':rows[0].numpy(),'sequence':container(rows)}
    before=fingerprint(obj);rows[0].numpy()[0]=123.
    assert fingerprint(obj)!=before and fingerprint(obj)==scalar_reference(obj)


def test_bulk_stack_failure_preserves_scalar_unsupported_layout_behavior():
    rows=[torch.ones(2) for _ in range(40)]
    rows[-1]=torch.sparse_coo_tensor(torch.tensor([[0]]),torch.tensor([1.]),(2,))
    for function in (fingerprint,scalar_reference):
        with pytest.raises(TypeError):function(rows)
