"""Time-windowed, causal joint p/v/spin fit for high-resolution stereo centers."""
from collections import deque
import numpy as np
from unittest.mock import patch
from . import tt39_spin as physics


class DenseFlight:
    def __init__(self,period,settings):
        self.period=period;self.settings=dict(settings)
        # Conservative center noise at 6x actual optical magnification.
        self.sigma=np.array([.0003,.0005,.0003]);self.reset()

    def reset(self):
        self.samples=deque(maxlen=int(np.ceil(.52/self.period))+2)
        self.last_fit=None;self.last_stamp=None

    def ingest(self,now,point):
        if self.samples and (now<=self.samples[-1][0] or now-self.samples[-1][0]>.055):self.samples.clear()
        if point is not None and np.isfinite(point).all():self.samples.append((float(now),np.asarray(point).copy()))

    def solve(self,t,p,pack,when,bounce,mode='coupled'):
        with patch.object(physics,'SIGMA',self.sigma):
            return physics.solve(t,p,pack,self.settings,mode,when,bounce)

    def evidence(self):
        if len(self.samples)<6:return None
        times,points=map(np.asarray,zip(*self.samples,strict=True));now=times[-1]
        recent=times>=now-.05
        if recent.sum()<3 or np.median(np.diff(points[recent,1])/np.diff(times[recent]))>-.85:return None
        # Candidate rebound is an observed local minimum with descent and rise.
        # Physical time intervals, rather than a camera-rate-specific frame count.
        candidates=[]
        for k in np.flatnonzero((times>=now-.20)&(points[:,1]>-1.42)&(points[:,1]<-.03)&(points[:,2]<.815)):
            tk=times[k];pre=(times>=tk-.085)&(times<=tk-.012);post=(times>=tk+.012)&(times<=now)
            local=(abs(times-tk)<.014)
            if pre.sum()<3 or post.sum()<3 or points[k,2]>points[local,2].min()+1e-7:continue
            before=np.polyfit(times[pre]-tk,points[pre,2],1)[0]
            after=np.polyfit(times[post]-tk,points[post,2],1)[0]
            if before<-.45 and after>.28:candidates.append(k)
        if candidates:
            k=candidates[-1];first=np.searchsorted(times,times[k]-.35);best=None
            for offset in (-1.,-.5,0.,.5,1.):
                when=times[k]+offset*self.period;t=times[first:]-when;p=points[first:]
                pack=physics.model(t,p,split=.010)
                if pack is None:continue
                fit=self.solve(t,p,pack,when,when)
                loss=fit.residual+.10*abs(fit.position[2]-self.settings['height'])
                if fit.pre[2]<-.4 and fit.post[2]>.2 and (best is None or loss<best[0]):best=(loss,fit)
            if best is not None and best[0]<.0025:return best[1]
        # Fit the longest coherent, already observed inbound arc. Curvature is
        # allowed to identify transverse spin instead of being rejected as noise.
        for duration in (.35,.23,.14,.075):
            start=np.searchsorted(times,now-duration);t=times[start:]-now;p=points[start:]
            if len(t)<6 or t[-1]-t[0]<.040:continue
            pack=physics.model(t,p)
            if pack is None:continue
            fit=self.solve(t,p,pack,now,None,'arc')
            if fit.pre[1]<-.85 and fit.residual<.0015:return fit
        return None

    def fit(self,now):
        if not self.samples or now-self.samples[-1][0]>.045:return None
        stamp=self.samples[-1][0]
        if stamp!=self.last_stamp:
            self.last_stamp=stamp;self.last_fit=self.evidence()
        if self.last_fit is None:return None
        return self.last_fit.state(now),self.last_fit
