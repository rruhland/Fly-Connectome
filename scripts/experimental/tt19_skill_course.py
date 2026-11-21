"""Adjust only the physical feeder spread while retaining rallies and both service turns."""
from types import MethodType
import numpy as np


def install_feeder(arena, course):
    if course not in ('center','wide'):
        raise ValueError('unknown feeder course')
    original=arena._opening
    previous_rng=arena._rng
    previous_server=arena.server
    centers={}
    for role in (-1,1):
        rows=[]
        for seed in range(64):
            arena.server=role
            arena._rng=np.random.default_rng([seed,10813])
            original()
            rows.append(np.r_[arena.ball.copy(),arena.velocity.copy()])
        centers[role]=np.mean(rows,axis=0)
    arena._rng=previous_rng
    arena.server=previous_server
    scale=np.full(6,.35 if course=='center' else .78)
    scale[[0,3]]=.20 if course=='center' else .65
    def serve(self):
        original()
        c=centers[self.server]
        self.ball[:]=c[:3]+scale[:3]*(self.ball-c[:3])
        self.velocity[:]=c[3:]+scale[3:]*(self.velocity-c[3:])
    arena._opening=MethodType(serve,arena)
