"""Exact transport-copy helper: no encoder arithmetic changes.

This is deliberately separate from previously pinned runtime sources and never
installs itself into a runner. Unsupported metadata, views and shared-storage groups retain
standard deepcopy. Distinct tensor object storage aliases trigger whole-list
fallback, so a cloned leaf never silently splits a storage alias group.
"""
import copy
import types

import torch
from fly_connectome.vision.interface import VisualStateEncoder

_ATOMS = (type(None), bool, int, float, str)


def _eligible(value, tensors, storages, containers):
    kind = type(value)
    if kind in _ATOMS:
        return True
    if kind in (list, dict):
        if id(value) in containers:
            return True
        containers.add(id(value))
        if kind is list:
            return all(_eligible(v, tensors, storages, containers) for v in value)
        return all(type(k) is str and _eligible(v, tensors, storages, containers) for k, v in value.items())
    if kind is not torch.Tensor:
        return False
    identity = id(value)
    if identity in tensors:
        return True
    if (value.device.type != 'cpu' or value.layout != torch.strided
            or value.requires_grad or value.grad is not None or not value.is_leaf or value._base is not None
            or value.is_conj() or value.is_neg() or value.is_quantized
            or value.is_inference() or value.__dict__ or not value.is_contiguous()
            or value.storage_offset() != 0):
        return False
    storage = value.untyped_storage()
    if storage.nbytes() != value.numel()*value.element_size():
        return False
    # _cdata distinguishes independent empty storages too; data_ptr() cannot.
    storage_id = storage._cdata
    if storage_id in storages and storages[storage_id] != identity:
        return False
    tensors.add(identity)
    storages[storage_id] = identity
    return True


def _copy(value, memo):
    kind = type(value)
    if kind in _ATOMS:
        return value
    identity = id(value)
    if identity in memo:
        return memo[identity]
    if kind is torch.Tensor:
        result = value.clone(memory_format=torch.preserve_format)
        memo[identity] = result
        return result
    if kind is list:
        result = []
        memo[identity] = result
        result.extend(_copy(v, memo) for v in value)
        return result
    result = {}
    memo[identity] = result
    result.update((k, _copy(v, memo)) for k, v in value.items())
    return result


def guarded_transport_copy(value):
    """Match ordinary transport deepcopy while conservatively retaining aliases."""
    if not _eligible(value, set(), {}, set()):
        return copy.deepcopy(value)
    return _copy(value, {})


# Reuse the *identical* original function code. Only the function-local `copy`
# global differs; no global module mutation and no rewritten encoder arithmetic.
_original = VisualStateEncoder.encode.__wrapped__
_globals = dict(_original.__globals__)
_globals['copy'] = types.SimpleNamespace(deepcopy=guarded_transport_copy)
_encode = types.FunctionType(_original.__code__, _globals, 'encode',
                             _original.__defaults__, _original.__closure__)
_encode.__kwdefaults__ = _original.__kwdefaults__


class ExactCopyEncoder(VisualStateEncoder):
    encode = torch.no_grad()(_encode)


def identical_encoder_code():
    return ExactCopyEncoder.encode.__wrapped__.__code__ is _original.__code__


VERSION = 'exact-visual-transport-copy-v1'


def install_exact_transport_copy(session):
    """Install on a pristine evaluator before its first camera, preserving fields.

    The caller must pin this NEW helper in its own staged runtime identity. No
    existing runner is modified and no policy input, choice or sensor work is
    removed. Installation preserves the initialized learned-contents hash.
    """
    from fly_connectome.control.codec import fingerprint

    old = session.encoder
    if (type(old) is not VisualStateEncoder or session.transitions != 0
            or session.initial_cameras != 0 or old.sample is not None):
        raise ValueError('exact transport copy must be installed before the first camera')
    before = session.learned_hash()
    if before != session.initial_learned_hash:
        raise RuntimeError('frozen contents changed before encoder installation')
    candidate = ExactCopyEncoder(old.height, old.width, old.period)
    if fingerprint(vars(candidate)) != fingerprint(vars(old)):
        raise ValueError('encoder installation requires pristine default fields')
    session.encoder = candidate
    after = session.learned_hash()
    if after != before:
        session.encoder = old
        raise RuntimeError('encoder installation unexpectedly changed learned contents')
    session.initial_learned_hash = after
    return dict(version=VERSION, installed_before_first_camera=True,
                identical_encoder_code_object=identical_encoder_code(),
                frozen_hash_before=before, frozen_hash_after=after)
