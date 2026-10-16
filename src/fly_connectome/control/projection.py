"""Declared fixed linear learner coordinates; no fit, labels or backpropagation."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from .capping import CAP_VERSION,cap_values

PROJECTION_VERSION='fixed-linear-policy-projection-v1:cap-after'


def validate_projection(config):
    try:
        shape=config['shape']
        valid=(config['version']==PROJECTION_VERSION and isinstance(config['matrix_file'],str)
               and bool(config['matrix_file']) and len(shape)==2
               and all(type(n) is int and n>0 for n in shape)
               and all(isinstance(config[k],str) and len(config[k])==64
                       and all(c in '0123456789abcdef' for c in config[k])
                       for k in ('file_sha256','array_sha256'))
               and isinstance(config['max_norm2'],(int,float))
               and np.isfinite(config['max_norm2']) and config['max_norm2']>0
               and isinstance(config['calibration'],dict))
        json.dumps(config,sort_keys=True,allow_nan=False)
    except (KeyError,TypeError,ValueError):valid=False
    if not valid:raise ValueError('declared finite fixed projection artifact and post-projection cap required')


class FixedPolicyProjection:
    """Own the exact matrix as a complete hashed, immutable checkpoint anchor.

    A file is needed only for construction. Recovery embeds/resolves the tensor
    through FrozenCodec; it does not reload an external file or refit a matrix.
    """
    def __init__(self,input_dimension,config):
        validate_projection(config)
        if config['shape'][1]!=input_dimension:raise ValueError('projection input dimension differs from raw transport')
        path=Path(config['matrix_file']);payload=path.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=config['file_sha256']:raise ValueError('projection file hash differs')
        matrix=np.load(path,allow_pickle=False)
        if (not isinstance(matrix,np.ndarray) or matrix.dtype!=np.dtype('float64') or list(matrix.shape)!=config['shape']
                or not np.isfinite(matrix).all()
                or hashlib.sha256(matrix.tobytes()).hexdigest()!=config['array_sha256']):
            raise ValueError('projection array shape/dtype/content differs')
        self.config=copy.deepcopy(config)
        self.matrix=torch.tensor(matrix,dtype=torch.float64,device='cpu')
        self.output_dimension=int(matrix.shape[0])
        self.input_dimension=int(input_dimension)
        self.indices=np.arange(self.output_dimension,dtype=np.int64)
        self.indices.flags.writeable=False
        self.max_norm2=float(config['max_norm2'])
        raw=json.dumps(self.config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        self.configuration_hash=hashlib.sha256(raw).hexdigest()

    def validate_embedded(self):
        """Recovery checks the embedded content without depending on its old path."""
        validate_projection(self.config)
        raw=json.dumps(self.config,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
        if (self.output_dimension!=self.config['shape'][0] or self.input_dimension!=self.config['shape'][1]
                or self.max_norm2!=self.config['max_norm2']
                or hashlib.sha256(raw).hexdigest()!=self.configuration_hash
                or self.indices.dtype!=np.dtype('int64')
                or not np.array_equal(self.indices,np.arange(self.output_dimension,dtype=np.int64))
                or self.matrix.device.type!='cpu' or self.matrix.layout!=torch.strided or self.matrix.dtype!=torch.float64
                or list(self.matrix.shape)!=self.config['shape']
                or self.matrix.requires_grad or self.matrix._base is not None
                or hashlib.sha256(self.matrix.numpy().tobytes()).hexdigest()!=self.config['array_sha256']):
            raise ValueError('embedded fixed projection changed')

    def features(self,encoded):
        indices=np.asarray(encoded['indices'].numpy())
        values=np.asarray(encoded['values'].numpy(),dtype=np.float64)
        if (indices.ndim!=1 or values.shape!=indices.shape or indices.dtype.kind not in 'iu'
                or not np.isfinite(values).all() or (indices<0).any()
                or (indices>=self.input_dimension).any() or len(np.unique(indices))!=len(indices)):
            raise ValueError('unique raw indices and finite values required for projection')
        with np.errstate(over='raise',invalid='raise'):
            values=self.matrix.numpy()[:,indices]@values
        return self.indices,cap_values(values,self.max_norm2)

    def mapped_weights(self,weights):
        weights=np.asarray(weights,dtype=np.float64)
        if weights.ndim!=2 or weights.shape[1]!=self.output_dimension or not np.isfinite(weights).all():
            raise ValueError('finite matching projected weights required')
        with np.errstate(over='raise',invalid='raise'):result=weights@self.matrix.numpy()
        if not np.isfinite(result).all():raise FloatingPointError('nonfinite mapped weights')
        return result

    def mapped_evaluation(self,weights):
        """Raw positive radial scaling preserves mapped greedy decisions, not Q values."""
        mapped=self.mapped_weights(weights)
        provenance=dict(version=PROJECTION_VERSION,array_sha256=self.config['array_sha256'],
            file_sha256=self.config['file_sha256'],shape=list(self.config['shape']),
            training_cap='squared norm bound after projection',
            training_max_norm2=self.max_norm2,configuration_hash=self.configuration_hash,
            projected_weights_sha256=hashlib.sha256(np.asarray(weights,dtype=np.float64).tobytes()).hexdigest(),
            mapped_weights_sha256=hashlib.sha256(mapped.tobytes()).hexdigest())
        cap=dict(version=CAP_VERSION,max_norm2=self.max_norm2,base_transform=None,
                 calibration=dict(mapped_projection=provenance,
                     semantics='positive raw radial bound preserves greedy argmax; Q scales differ from training'))
        return mapped,cap
