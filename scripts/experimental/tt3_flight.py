"""Learn repeatable image-space accelerations and reversal surfaces from unlabelled tracks."""
from collections import defaultdict, deque
import numpy as np

try:
    from .tt3_learning import FlightHistory, TableLearner, approach_normal
except ImportError:
    from tt3_learning import FlightHistory, TableLearner, approach_normal


class MotionEvidence:
    def __init__(self):
        self.surfaces = defaultdict(lambda: deque(maxlen=80))
        self.accelerations = deque(maxlen=150)
        self.gravity = None

    def add_acceleration(self, value):
        self.accelerations.append(value.copy())
        if len(self.accelerations) >= 4:
            self.gravity = np.median(np.asarray(self.accelerations),axis=0)

    def boundary(self, k, sign):
        rows = self.surfaces.get((k, sign))
        if not rows or len(rows) < 2:
            return None
        pos = np.array([r[0] for r in rows])
        # A cluster of near-coincident reversals is evidence of a surface.
        neighbors = (abs(pos[:,None]-pos[None,:]) < .014).sum(axis=1)
        center = pos[int(neighbors.argmax())]
        chosen = [r for r in rows if abs(r[0]-center) < .014]
        return np.median(chosen,axis=0) if len(chosen) >= 2 else None


class ForecastHistory(FlightHistory):
    def __init__(self, owner, evidence):
        super().__init__()
        self.owner = owner
        self.evidence = evidence
        self.tail = deque(maxlen=13)
        self.recent = {}
        self.previous_track = None
        self.corrected = 0

    def update(self, track, frame):
        estimate = super().update(track,frame)
        if track is None or not track.observed:
            self.tail.clear()
            return estimate
        if track.id != self.previous_track:
            self.tail.clear()
            self.recent.clear()
        self.previous_track=track.id
        self.tail.append(np.asarray(track.position))
        if len(self.tail) >= 9:
            self._measure(frame)
        if estimate is None or self.owner.ready is None:
            return estimate
        position, velocity, acceleration = estimate
        normal = self.owner.normal
        if normal is None:
            normal = approach_normal(position,velocity,self.owner.ready)
        closing = velocity @ normal
        if abs(closing) < 1e-5:
            return estimate
        horizon = -((position-self.owner.ready)@normal) / closing
        if not 1. < horizon < 75.:
            return estimate
        # Infer physics in every coordinate. A coordinate is corrected only
        # after repeated visual evidence, with no baked-in gravity axis.
        adjusted = velocity.copy()
        accel = acceleration.copy()
        for k in range(len(position)):
            if abs(normal[k]) > .5 or self.evidence.gravity is None:
                continue
            g = self.evidence.gravity[k]
            if abs(g)<.00016:
                continue
            g = np.clip(g,-.0025,.0025)
            if len(self.tail)>=4:
                n=min(5,len(self.tail))
                samples=np.asarray(self.tail)[-n:,k]
                slope=np.dot(np.arange(n)-(n-1)/2,samples)/(n*(n*n-1)/12)
                v=slope+g*(n-1)/2
            else:
                v=velocity[k]
            stop=position[k]+v*horizon+.5*g*horizon*horizon
            for sign in (-1,1):
                boundary=self.evidence.boundary(k,sign)
                if boundary is None:
                    continue
                wall,restitution=boundary
                discriminant=v*v+2*g*(wall-position[k])
                if discriminant <= 0:
                    continue
                impact=sign*np.sqrt(discriminant)
                t=(impact-v)/g
                if 0.<t<horizon:
                    rest=horizon-t
                    stop=wall-restitution*impact*rest+.5*g*rest*rest
                    self.corrected+=1
                    break
            adjusted[k]=(stop-position[k])/horizon
            accel[k]=0.
        return position,adjusted,accel

    def _measure(self, frame):
        y=np.asarray(self.tail)
        # Slopes on either side of the middle camera instant.
        left=(y[-5]-y[-9])/4
        right=(y[-1]-y[-5])/4
        smooth=np.all(np.abs(right-left)<.009)
        if smooth and frame%3==0:
            self.evidence.add_acceleration((right-left)/4)
        for k,(v,w) in enumerate(zip(left,right,strict=True)):
            if v*w >= 0 or min(abs(v),abs(w))<.0018 or abs(v-w)<.012:
                continue
            sign=1 if v>0 else -1
            key=(k,sign)
            if frame-self.recent.get(key,-100)<8:
                continue
            self.recent[key]=frame
            middle=y[-8:-1,k]
            surface=middle.max() if sign==1 else middle.min()
            ratio=np.clip(abs(w/v), .4,1.3)
            self.evidence.surfaces[key].append((float(surface),float(ratio)))


class ForecastLearner(TableLearner):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.observed_motion=MotionEvidence()

    def reset(self, option=None):
        super().reset(option)
        self.flight=ForecastHistory(self,self.observed_motion)

class SmoothForecast(ForecastHistory):
    """Estimate each trajectory coordinate separately so a turn in one axis
    does not discard the measured velocity in all the others.
    """
    def __init__(self,owner,evidence):
        super().__init__(owner,evidence)
        self.segments=[]
        self.last_at=None
        self.derivative={}
        self.extrapolations=0

    def update(self,track,frame):
        if track is None or not track.observed:
            return None
        point=np.asarray(track.position)
        if self.last_at is None or frame-self.last_at>3 or track.id!=self.previous_track:
            self.segments=[deque(maxlen=19) for _ in point]
            self.tail.clear()
            self.recent.clear()
        self.previous_track=track.id
        self.frame=self.last_at=frame
        self.tail.append(point)
        if len(self.tail)>=9:
            self._measure(frame)
        center=point.copy()
        speed=np.array(track.velocity)
        gravity=self.evidence.gravity
        acceleration=np.zeros(len(point)) if gravity is None else np.where(abs(gravity)>.00014,gravity,0.)
        for k,values in enumerate(self.segments):
            g=acceleration[k]
            if len(values)>=4:
                last_t,last_obs=values[-1]
                prior=self.derivative.get(k)
                if prior is not None:
                    prediction=prior[0]+prior[1]*(frame-last_t)+.5*g*(frame-last_t)**2
                    if abs(point[k]-prediction)>.0055:
                        values.clear()
                        values.append((last_t,last_obs))
            values.append((frame,point[k]))
            if len(values)>=2:
                times=np.array([t-frame for t,_ in values])
                detrended=np.array([y for _,y in values])-.5*g*times**2
                dt=times-times.mean()
                slope=dt@detrended/(dt@dt)
                here=detrended.mean()-slope*times.mean()
                center[k],speed[k]=here,slope
                self.derivative[k]=(here,slope)
        ready=self.owner.ready
        if ready is None:
            return center,speed,acceleration.copy()
        normal=self.owner.normal if self.owner.normal is not None else approach_normal(center,speed,ready)
        closing=speed@normal
        if abs(closing)<1e-6:
            return center,speed,acceleration.copy()
        horizon=-(center-ready)@normal/closing
        if not 1.<horizon<75.:
            return center,speed,acceleration.copy()
        velocity=speed.copy()
        accel=acceleration.copy()
        for k,g in enumerate(acceleration):
            if abs(g)<.00014 or abs(normal[k])>.5:
                continue
            for sign in (-1,1):
                learned=self.evidence.boundary(k,sign)
                if learned is None:continue
                wall,restitution=learned
                v=speed[k]
                discriminant=v*v+2*g*(wall-center[k])
                if discriminant<=0:continue
                impact=sign*np.sqrt(discriminant)
                when=(impact-v)/g
                if 0.<when<horizon:
                    remaining=horizon-when
                    final=wall-restitution*impact*remaining+.5*g*remaining**2
                    velocity[k]=(final-center[k])/horizon
                    accel[k]=0.
                    self.extrapolations+=1
                    break
        return center,velocity,accel


class SmoothLearner(ForecastLearner):
    def reset(self, option=None):
        super().reset(option)
        self.flight=SmoothForecast(self,self.observed_motion)
