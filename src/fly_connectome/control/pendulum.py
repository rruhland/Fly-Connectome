"""Small camera-only unstable upright pendulum; no controller or task labels.

Angles/velocities belong only to environment physics. Policy observations are
ordinary rendered pixels; native reward is elapsed survival time. This module
does not create a model, expand experimental seeds, or launch an allocation.
"""
from dataclasses import dataclass
import math
import numpy as np
import torch
from .environment import CameraTransition


@dataclass(frozen=True)
class PendulumPhysics:
    dt: float=1/120
    gravity: float=8.
    torque: float=12.
    damping: float=.1
    failure_angle: float=math.pi/3
    initial_angle_min: float=.03
    initial_angle_max: float=.12
    initial_velocity_max: float=.2

    def validate(self):
        values=vars(self)
        if (any(isinstance(v,bool) or not isinstance(v,(int,float)) or not math.isfinite(v) for v in values.values())
                or min(self.dt,self.gravity,self.torque,self.failure_angle)<=0):
            raise ValueError('finite positive pendulum physics required')
        if (self.damping<0 or not 0<self.initial_angle_min<=self.initial_angle_max<self.failure_angle<math.pi
                or self.initial_velocity_max<0):
            raise ValueError('upright initial distribution and genuine failure boundary required')


def rk4_step(angle,velocity,drive,config):
    """Double-precision integration under one fixed legal drive, one physics tick."""
    def acceleration(a,v):return config.gravity*math.sin(a)+config.torque*drive-config.damping*v
    dt=config.dt;k1a=velocity;k1v=acceleration(angle,velocity)
    k2a=velocity+dt*k1v/2;k2v=acceleration(angle+dt*k1a/2,velocity+dt*k1v/2)
    k3a=velocity+dt*k2v/2;k3v=acceleration(angle+dt*k2a/2,velocity+dt*k2v/2)
    k4a=velocity+dt*k3v;k4v=acceleration(angle+dt*k3a,velocity+dt*k3v)
    return angle+dt*(k1a+2*k2a+2*k3a+k4a)/6,velocity+dt*(k1v+2*k2v+2*k3v+k4v)/6


class PendulumCamera:
    """Three torques, 120 Hz physics / 50 Hz camera and explicit raw end flags.

    Five-second time limits are truncations of continuing balance, not success
    terminals. The final rendered frame is returned before an explicit reset.
    A new episode restarts its 2/3-tick camera cadence and draws independent
    signed angle magnitude and angular velocity from the private seeded RNG.
    """
    actions=(-1.,0.,1.)

    def __init__(self,seed=0,*,max_episode_ticks=600,config=PendulumPhysics()):
        config.validate()
        if type(seed) is not int or seed<0:
            raise ValueError('nonnegative private environment seed required')
        if (type(max_episode_ticks) is not int or max_episode_ticks<1
                or not math.isclose(config.dt,1/120,rel_tol=0,abs_tol=1e-15)):
            raise ValueError('positive finite tick limit and declared 120 Hz physics required')
        self.config=config;self.max_episode_ticks=max_episode_ticks
        self.rng=np.random.default_rng(seed)
        self.total_ticks=0;self.decisions=self.episode_ticks=0;self.needs_reset=False
        self.y,self.x=torch.meshgrid(torch.arange(64,dtype=torch.float64),
                                   torch.arange(64,dtype=torch.float64),indexing='ij')
        self._draw_initial()

    def _draw_initial(self):
        magnitude=float(self.rng.uniform(self.config.initial_angle_min,self.config.initial_angle_max))
        sign=-1. if int(self.rng.integers(2))==0 else 1.
        self.angle=sign*magnitude
        self.velocity=float(self.rng.uniform(-self.config.initial_velocity_max,self.config.initial_velocity_max))

    def render(self):
        """Binary 64×64 pivot, one-pixel rod, radius-two bob; no annotations."""
        pivot_x,pivot_y,length=32.,40.,24.
        dx=length*math.sin(self.angle);dy=-length*math.cos(self.angle)
        x=self.x-pivot_x;y=self.y-pivot_y
        along=((x*dx+y*dy)/(length*length)).clamp(0,1)
        rod=(x-along*dx).square()+(y-along*dy).square()<=.55**2
        bob=(x-dx).square()+(y-dy).square()<=2.**2
        pivot=x.square()+y.square()<=1.25**2
        return (rod|bob|pivot).clone()

    def reset(self):
        if not self.needs_reset:raise ValueError('reset is explicit after a true end boundary')
        self._draw_initial();self.episode_ticks=self.decisions=0;self.needs_reset=False
        return self.render()

    def step(self,action):
        if self.needs_reset:raise ValueError('reset required after failure or time-limit truncation')
        if type(action) is not int or not 0<=action<len(self.actions):
            raise ValueError('legal torque index required')
        budget=(self.decisions+1)*12//5-self.decisions*12//5
        terminated=truncated=False
        for ticks in range(1,budget+1):
            self.angle,self.velocity=rk4_step(self.angle,self.velocity,self.actions[action],self.config)
            if not math.isfinite(self.angle) or not math.isfinite(self.velocity):
                raise FloatingPointError('nonfinite pendulum integration')
            self.episode_ticks+=1;self.total_ticks+=1
            terminated=abs(self.angle)>=self.config.failure_angle
            truncated=not terminated and self.episode_ticks>=self.max_episode_ticks
            if terminated or truncated:break
        self.decisions+=1;self.needs_reset=terminated or truncated
        elapsed=ticks*self.config.dt
        return CameraTransition(self.render(),elapsed,elapsed,terminated,truncated,ticks,0)
