"""Connect causal high-rate stereo estimates to the unchanged TT83 body twin.

A host collects actual RGB exposures and supplies a flight fit. This adapter
never receives the live simulator state. It exposes the standard OpticalClinic
tracker interface with a legacy fallback when no fresh dense fit is usable.
"""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class LatestFlight:
    estimate: object
    time: float

    @property
    def bounced(self):
        return self.estimate.bounce is not None and self.estimate.bounce <= self.time

    @property
    def velocity(self):
        return self.advance(self.time)[1]

    def advance(self, now):
        pv,_=self.estimate.state(now)
        return pv[:3],pv[3:]


class PublicTracker:
    def __init__(self, legacy):
        self.legacy=legacy
        self.dense=None
        self.counts={"dense":0,"legacy":0}
        # OpticalClinic queries tracker.filter.rotation after rebuilding the
        # ball orientation at identity. The public latents are world-frame.
        self.filter=self

    @property
    def latest(self):
        return self.dense if self.dense is not None else self.legacy.latest

    def update(self, now, x_legacy, dense_result):
        self.dense=None
        if dense_result is not None:
            (x,w), estimate = dense_result
            x,w=np.asarray(x),np.asarray(w)
            if (x.shape==(6,) and w.shape==(3,) and np.isfinite(x).all() and np.isfinite(w).all()
                    and x[4]<-.85 and 0<=now-estimate.time<=.245):
                self.dense=LatestFlight(estimate,float(now))
                self.counts["dense"]+=1
                return x.copy()
        self.counts["legacy"]+=1
        return x_legacy

    def rotation(self,now):
        if self.dense is not None:
            return self.dense.estimate.state(now)[1]
        model=getattr(self.legacy,"filter",None)
        if hasattr(model,"rotation"):
            return model.rotation(now)
        return np.zeros(3)
