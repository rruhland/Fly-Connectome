"""Trusted local continuation codec and deterministic graph comparisons."""
from collections import deque
import hashlib
import io
import json
import pickle
import numpy as np
import torch

GRAPH_DIGEST = 'm1b-graph-v2:pickle5-no-memo'

def fingerprint(value):
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
            # Batch the bytes scan only; canonical nodes and graph reference IDs
            # remain identical to the scalar format, including repeated aliases.
            if len(obj) >= 32:
                first = obj[0]
                supported = {torch.bool, torch.uint8, torch.int8, torch.int16,
                             torch.int32, torch.int64, torch.float16,
                             torch.float32, torch.float64, torch.complex64,
                             torch.complex128}
                if (type(first) is torch.Tensor and first.device.type == 'cpu'
                        and first.layout == torch.strided and first.dtype in supported
                        and first.numel() <= 64
                        and len(obj) * first.numel() * first.element_size() <= 1048576):
                    shape, dtype = first.shape, first.dtype
                    if all(type(x) is torch.Tensor and x.dtype == dtype for x in obj):
                        # Stack validates shape/device/layout before any child is
                        # assigned a graph ID. force=True retains scalar handling
                        # of negative/conjugate and noncontiguous logical values.
                        try:
                            with torch.no_grad():
                                raw = torch.stack(tuple(obj)).numpy(force=True).tobytes()
                        except (RuntimeError, TypeError, NotImplementedError):
                            raw = None
                        if raw is not None:
                            size = first.numel() * first.element_size()
                            dtype_text, shape_list = str(dtype), list(shape)
                            nodes = []
                            for index, x in enumerate(obj):
                                identity = id(x)
                                if identity in seen:
                                    nodes.append(['ref', seen[identity]])
                                else:
                                    seen[identity] = len(seen)
                                    retained.append(x)
                                    nodes.append(['tensor', dtype_text, shape_list,
                                                  raw[index * size:(index + 1) * size]])
                            return [type(obj).__name__, nodes]
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



def frozen_tensors(session):
    vision = session.vision
    tensors = {'observer.weights':vision.observer.weights, 'observer.bias':vision.observer.bias}
    contract=getattr(session,'visual_memory_contract',None)
    if contract is not None:
        # Optional exact-class contract only. The default frozen bank is unchanged;
        # mutable memories remain in full per-camera graph hashes/checkpoints.
        from .online_vision import PersistentVisionSession,MUTABLE_VISION_CONTRACT
        from .pendulum_session import PendulumLearningSession
        if type(session) not in (PersistentVisionSession,PendulumLearningSession) or contract!=MUTABLE_VISION_CONTRACT:
            raise ValueError('unknown mutable visual-memory session contract')
        projection=getattr(session,'projection',None)
        if projection is not None:tensors['policy_projection.matrix']=projection.matrix
        return tensors
    for name in ('keys','values','metric'):
        tensors['memory.'+name] = getattr(vision.state.memory,name)
    projection=getattr(session,'projection',None)
    if projection is not None:tensors['policy_projection.matrix']=projection.matrix
    for horizon, model in sorted(vision.dynamics.items()):
        for branch in ('short','long'):
            for name in ('keys','values'):
                tensors[f'dynamics.{horizon}.{branch}.{name}'] = getattr(getattr(model,branch),name)
    return tensors


def artifact_bytes(tensors):
    # Encode object aliases explicitly; reject views rather than silently breaking them.
    nodes, references, identities = [], {}, {}
    for name, value in sorted(tensors.items()):
        if value.device.type != 'cpu' or value._base is not None:
            raise ValueError('frozen anchors must be standalone CPU tensors')
        if id(value) not in identities:
            identities[id(value)] = len(nodes)
            array = value.detach().contiguous().numpy()
            nodes.append([str(array.dtype),list(array.shape),array.tobytes().hex()])
        references[name] = identities[id(value)]
    return json.dumps([references,nodes],sort_keys=True,separators=(',',':')).encode()


class FrozenCodec:
    """One content-addressed immutable bank; per-checkpoint references reuse nodes.

    RNGs and calibration/history are not deduplicated: they stay in active state.
    Deserialization preserves repeated references to each immutable tensor.
    """
    def __init__(self, artifact):
        self.artifact = artifact
        self.key = hashlib.sha256(artifact).hexdigest()

    @classmethod
    def from_session(cls, session):
        return cls(artifact_bytes(frozen_tensors(session)))

    def dumps(self, session):
        tensors = frozen_tensors(session)
        if artifact_bytes(tensors) != self.artifact:
            raise ValueError('frozen vision tensors changed')
        names = {}
        for name, tensor in sorted(tensors.items()):
            names.setdefault(id(tensor),name)
        key = self.key
        class Writer(pickle.Pickler):
            def persistent_id(self, obj):
                if id(obj) in names:
                    return ('frozen-tensor-v1',key,names[id(obj)])
                return None
        stream = io.BytesIO()
        Writer(stream,protocol=5).dump(session)
        return stream.getvalue()

    def loads(self, payload):
        references, descriptions = json.loads(self.artifact)
        nodes = [torch.from_numpy(np.frombuffer(bytes.fromhex(raw),dtype=dtype).copy().reshape(shape))
                 for dtype,shape,raw in descriptions]
        tensors = {name:nodes[index] for name,index in references.items()}
        key = self.key
        class Reader(pickle.Unpickler):
            def persistent_load(self, identifier):
                tag, artifact_key, name = identifier
                if tag != 'frozen-tensor-v1' or artifact_key != key or name not in tensors:
                    raise ValueError('unknown frozen reference')
                return tensors[name]
        return Reader(io.BytesIO(payload)).load()
