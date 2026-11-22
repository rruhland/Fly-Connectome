"""Fit a causal incoming flight, including table contact, to a short stereo history."""
from copy import deepcopy
from xml.etree import ElementTree as ET

import numpy as np
from mujoco import rollout

from .tt4_arena import AIR_DRAG, BALL_INERTIA, BALL_MASS, MAGNUS_COEFFICIENT, SPIN_DECAY, TABLE_HEIGHT, mujoco
from .tt20_curriculum import model_arena

FRAME = np.array((0., 0., TABLE_HEIGHT))
# Independent, normalized coordinates for a finite difference update.
STEP = np.array([.0006]*3 + [.012]*3 + [6.]*3)


class FlightInverse:
    def __init__(self):
        arena = model_arena()
        try:
            original = arena._model_document()
            doc = ET.Element('mujoco')
            for name in ('option', 'default'):
                doc.append(deepcopy(original.find(name)))
            world = ET.SubElement(doc,'worldbody')
            for name in arena.table_panels:
                world.append(deepcopy(original.find(f".//geom[@name='{name}']")))
            world.append(deepcopy(original.find(".//body[@name='ball']")))
            self.model = mujoco.MjModel.from_xml_string(ET.tostring(doc, encoding='unicode'))
            self.micro = round(arena.control_dt / self.model.opt.timestep)
        finally:
            arena.close()
        self.data = mujoco.MjData(self.model)

    def simulate(self, theta, times, xyz):
        n = len(theta)
        frames = int(times[-1])
        t = np.arange(frames*self.micro) / self.micro
        # Drag uses the slope between received observations. A second pass is unnecessary
        # for ranking near-identical incoming paths over this short window.
        d = np.diff(xyz,axis=0) / (np.diff(times)[:,None]/30)
        slopes = np.column_stack([np.interp(t, .5*(times[:-1]+times[1:]), a) for a in d.T])
        v = np.broadcast_to(slopes,(n,len(t),3))
        spin = theta[:,None,6:]
        drag = -AIR_DRAG * np.linalg.norm(v,axis=2,keepdims=True)*v
        forces = np.zeros((n,len(t),12))
        forces[:,:,6:9] = BALL_MASS*(drag + MAGNUS_COEFFICIENT*np.cross(spin,v))
        forces[:,:,9:] = -BALL_INERTIA * SPIN_DECAY * spin
        initial = np.zeros((n,14))
        initial[:,1:4] = theta[:,:3]+FRAME
        initial[:,4] = 1
        initial[:,8:14] = theta[:,3:]
        states,_ = rollout.rollout(self.model,self.data,initial,forces,control_spec=mujoco.mjtState.mjSTATE_XFRC_APPLIED)
        states = np.concatenate((initial[:,None,:],states[:,self.micro-1::self.micro]), axis=1)
        return states[:,times,:]

    @staticmethod
    def segment(history):
        samples = [(i, np.asarray(p)) for i,p in history[-21:] if p is not None]
        if len(samples)<6 or samples[-1][0] != history[-1][0]:
            return None
        times, xyz = map(np.asarray,zip(*samples))
        # Cut after a flight reversal or missing interval. The table bounce stays inside the fit.
        v = np.diff(xyz[:,1])/np.diff(times)
        breaks = np.flatnonzero((v > -.012) | (np.diff(times)>3))
        start = 0 if not len(breaks) else int(breaks[-1]+1)
        times,xyz=times[start:], xyz[start:]
        if len(times)<6 or times[-1]-times[0]<5 or times[-1]-times[0]>24:
            return None
        return times-times[0],xyz

    def fit(self,history):
        segment=self.segment(history)
        if segment is None:
            return None
        times, xyz=segment
        dt=times[:3]/30.
        x=np.column_stack((np.ones(3),dt))
        p0,v0=np.linalg.lstsq(x,xyz[:3]+.5*dt[:,None]**2*np.array((0,0,9.81)),rcond=None)[0]
        theta = np.r_[p0,v0,[0.,0.,0.]]
        # Stereo noise is on the millimetre scale. The isotropic prior regularizes spin
        # directions that are poorly observable from a single airborne arc.
        prior_scale = np.r_[np.full(3, .01),np.full(3, .9),np.full(3,55.)]
        original=theta.copy()
        def residual(p):
            states=self.simulate(p,times,xyz)
            optical=(states[:,:,1:4]-FRAME-xyz)/.0025
            prior=(p-original)/prior_scale
            return np.concatenate((optical.reshape(len(p),-1),prior),axis=1),states
        best_states=None
        for _ in range(5):
            variants=np.vstack((theta,theta+np.diag(STEP),theta-np.diag(STEP)))
            err,states=residual(variants)
            best_states=states[0]
            jac=(err[1:10]-err[10:]).T/2
            move=np.linalg.solve(jac.T@jac+.12*np.eye(9),-jac.T@err[0])
            attempts=theta + np.array([1.,.45,.15])[:,None]*np.clip(move*STEP,
                -np.r_[[.01]*3,[.40]*3,[40.]*3],np.r_[[.01]*3,[.40]*3,[40.]*3])
            err2,states2=residual(attempts)
            losses=np.sum(err2**2,axis=1)
            i=int(np.argmin(losses))
            if losses[i]>=err[0]@err[0]:
                break
            theta=attempts[i]
            best_states=states2[i]
        rtn=best_states[-1]
        residual_mm=1000*np.sqrt(np.mean(np.sum((best_states[:,1:4]-FRAME-xyz)**2,axis=1)))
        if residual_mm>6 or np.linalg.norm(rtn[1:4]-FRAME-xyz[-1])>.012:
            return None
        spin=np.empty(3)
        mujoco.mju_rotVecQuat(spin,rtn[11:14],rtn[4:8])
        return rtn[1:4]-FRAME,rtn[8:11],spin,residual_mm
