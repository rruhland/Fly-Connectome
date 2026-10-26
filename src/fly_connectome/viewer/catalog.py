"""Curated, read-only checkpoints. Historical cohorts are never a learning curve."""
from dataclasses import dataclass
from pathlib import Path
import hashlib
import json
import numpy as np
from ..control.session import MODEL_HASH, VISION_CONFIGURATION, model_bytes, source_signature


@dataclass(frozen=True)
class Checkpoint:
    id: str
    title: str
    date: str
    description: str
    weights: np.ndarray
    scaling: dict
    digest: str
    evidence: dict

    def public(self):
        return dict(id=self.id,title=self.title,date=self.date,description=self.description,
                    digest=self.digest,evidence=self.evidence)


def read(path):
    return json.loads(Path(path).read_text())


def load_catalog(experiments,export=None):
    root=Path(experiments);export=Path(export) if export else root/'pong-interception-checkpoint-20261025'
    model_bytes()  # Verify the same frozen visual prior used in the benchmark.
    declared=read(export/'checkpoint.json')
    if declared['source_signature']!=source_signature():
        raise ValueError('export runtime source signature differs from the installed benchmark path')
    if declared['vision_configuration']!=VISION_CONFIGURATION:
        raise ValueError('export visual configuration mismatch')
    if hashlib.sha256((export/'radial-cap.json').read_bytes()).hexdigest()!=declared['files']['radial-cap.json']:
        raise ValueError('export policy scaling hash mismatch')
    if declared['vision_model_sha256']!=MODEL_HASH:
        raise ValueError('export visual prior differs from the installed model')
    if hashlib.sha256((export/'default.pt').read_bytes()).hexdigest()!=MODEL_HASH:
        raise ValueError('export visual prior hash mismatch')
    definitions=[
        ('first','First learned returns','21 OCT','The first return curriculum. A learned starting point, before the phase-specific skills.',
         root/'pong-first-return-curriculum-20261021-r1','policy_endpoint.npy',
         dict(metrics=[dict(label='First returns',hits=41,n=64)],cohort='Historical 64 ordinary games; separate cohort.',note='Early return acquisition. Not directly comparable to later phase-specific cohorts.')),
        ('direct','Direct interception','22 OCT','A rewarded specialist for balls served directly toward the learned paddle.',
         root/'pong-leftward-population-20261022-r1','endpoint.npy',
         dict(metrics=[dict(label='Direct returns',hits=56,n=64)],cohort='Historical 64 direct starts; separate cohort.',note='A real skill donor. This stage sacrificed later contacts and is not a combined controller.')),
        ('opponent','Opponent-return skill','22 OCT','Learning the other opening: the ball reaches the opponent before coming back.',
         root/'pong-rightward-interception-20261022-r1','endpoint.npy',
         dict(metrics=[dict(label='After opponent',hits=44,n=64),dict(label='Direct returns',hits=50,n=64)],cohort='Historical 64 starts per phase; separate cohort.',note='Phase acquisition with retention still unresolved. These cohorts differ from the later matched comparison.')),
        ('r4','Meshed skills · r4','25 OCT','Own successful trajectories were combined into a stronger shared controller.',
         root/'pong-local-repair-20261025-r4','selected.npy',
         dict(metrics=[dict(label='Direct returns',hits=118,n=128),dict(label='After opponent',hits=113,n=128)],cohort='Same fixed 128-start cohorts as the final checkpoint.',note='Matched baseline for the final comparison. Preserved unchanged.')),
        ('final','Causal credit · final','25 OCT','Single-action outcome credit, with soft preservation of already learned behavior.',
         export,'weights.npy',
         dict(metrics=[dict(label='Direct returns',hits=125,n=128),dict(label='After opponent',hits=117,n=128)],cohort='Fixed larger check: two new 64-start cohorts per phase.',note='RIGHT varies across cohorts: including the earlier probe, 142/160 opponent-return starts succeeded (88.75%). No distribution-wide 90% guarantee.',
              full='Ordinary full games: 28/32 first returns, 11/32 with repeated returns. Rally improvement and learning from scratch are not established.')),
    ]
    checkpoints={};unavailable=[]
    for identity,title,date,description,folder,name,evidence in definitions:
        path=folder/name
        if not path.exists():
            if identity=='final':raise FileNotFoundError('final exported policy is missing')
            unavailable.append(identity);continue
        if identity in ('first','direct','opponent'):
            historical=read(folder/'manifest.json')
            if historical['model_hash']!=MODEL_HASH or historical['vision_configuration']!=VISION_CONFIGURATION:
                raise ValueError('historical checkpoint visual prior is incompatible: '+identity)
        raw=path.read_bytes();digest=hashlib.sha256(raw).hexdigest()
        if identity=='final' and digest!=declared['weights_sha256']:
            raise ValueError('final exported policy hash mismatch')
        weights=np.load(path,allow_pickle=False)
        if weights.shape!=(3,4389) or not np.isfinite(weights).all():
            raise ValueError('invalid frozen checkpoint '+identity)
        weights=np.array(weights,dtype=np.float64,copy=True);weights.setflags(write=False)
        scaling=read(folder/'radial-cap.json')
        checkpoints[identity]=Checkpoint(identity,title,date,description,weights,scaling,digest,evidence)
    return checkpoints,unavailable
