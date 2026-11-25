"""Native motor babbling and learned reachable body inverse for the centered SO-101."""

import argparse
import ctypes
from functools import lru_cache
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory
from time import perf_counter
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from . import tt72_retarget_trajectory as tt72
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import NOMINAL_NO_LOAD

ROOT = Path('artifacts/tt88_inverse_body')
PLAN = ROOT / 'protocol.json'
DATA = ROOT / 'babble.npy'
HOLD = ROOT / 'babble-held.npy'
DT = tt72.DT
mpc = tt72.mpc
NATIVE = Path(__file__).with_name('tt88_babble.cpp')
POSE = ('shoulder_pan','shoulder_lift','elbow_flex','wrist_flex','wrist_roll','gripper')


@lru_cache(maxsize=1)
def kernel():
    folder = TemporaryDirectory(prefix='tt88-babble-')
    path = Path(folder.name) / 'motor.so'
    package = Path(mujoco.__file__).parent
    lib = next(package.glob('libmujoco.so.*'))
    subprocess.run(['c++','-std=c++17','-O3','-ffp-contract=off','-shared','-fPIC',
                    '-I',str(package/'include'),str(NATIVE),str(lib),'-o',str(path)], check=True)
    dll = ctypes.CDLL(str(path))
    p,i,u,d = ctypes.c_void_p,ctypes.c_int,ctypes.c_uint64,ctypes.c_double
    common = (p,p,p,p,p,p,p,d,i,i)
    babble, bench = dll.tt88_babble, dll.tt88_evaluate
    babble.argtypes, bench.argtypes = (*common,i,i,u,p), (*common,p,i,i,p)
    babble.restype = bench.restype = None
    return babble, bench, folder


class Body:
    """Same learner joints/drives, grasp constraint, and camera masses, without a ball."""
    def __init__(self, arena):
        x = arena._model_document()
        world = x.find('worldbody')
        for child in list(world):
            if child.tag != 'body' or child.get('name') not in ('learner_base', 'learner_paddle'):
                world.remove(child)
        for geom in world.iter('geom'):
            geom.set('contype','0')
            geom.set('conaffinity','0')
        for tag, keep in (('tendon', lambda e: False),
                          ('equality', lambda e: e.get('name')=='tt19_grasp'),
                          ('actuator', lambda e: e.get('name','').startswith('learner_'))):
            node = x.find(tag)
            for e in list(node):
                if not keep(e):
                    node.remove(e)
        self.m = mujoco.MjModel.from_xml_string(ET.tostring(x,encoding='unicode'))
        self.d = mujoco.MjData(self.m)
        self.names = ['learner_'+p for p in POSE]
        self.p = np.asarray([self.m.joint(n).qposadr[0] for n in self.names],dtype=np.intc)
        self.v = np.asarray([self.m.joint(n).dofadr[0] for n in self.names],dtype=np.intc)
        if [self.m.actuator(n).id for n in self.names] != list(range(6)):
            raise ValueError('Mismatched near-side actuator ordering')
        axes = arena._arm_ids[1][2]
        self.step = np.ascontiguousarray(arena._joint_increments[axes])
        self.lo, self.hi = (np.ascontiguousarray(z[axes]) for z in (arena._joint_low,arena._joint_high))
        self.itr = int(round(arena.control_dt / arena.model.opt.timestep))
        self.stride = self.step * self.itr
        self.racket = self.m.geom('learner_racket').id
        self.kp = self.m.actuator_gainprm[:5,0].copy()
        self.kv = -self.m.actuator_biasprm[:5,2].copy()
        self.force = self.m.actuator_forcerange[:5,1].copy()
        self.read(arena)
        self.common = (self.m._address,self.d._address,self.p.ctypes.data,self.v.ctypes.data,self.step.ctypes.data,
                       self.lo.ctypes.data,self.hi.ctypes.data,NOMINAL_NO_LOAD,self.itr,self.racket)

    def read(self, arena):
        source = arena.model
        for n in [*self.names,'learner_paddle_free']:
            width = 7 if n.endswith('_free') else 1
            i,j = self.m.joint(n).qposadr[0],source.joint(n).qposadr[0]
            self.d.qpos[i:i+width] = arena.data.qpos[j:j+width]
            width = 6 if n.endswith('_free') else 1
            i,j = self.m.joint(n).dofadr[0],source.joint(n).dofadr[0]
            self.d.qvel[i:i+width] = arena.data.qvel[j:j+width]
            self.d.qacc_warmstart[i:i+width] = arena.data.qacc_warmstart[j:j+width]
        self.d.ctrl[:] = arena.data.ctrl[arena._arm_ids[1][2]]
        self.d.time = arena.data.time
        mujoco.mj_forward(self.m,self.d)

    @property
    def state(self):
        return np.r_[self.d.qpos[self.p[:5]],self.d.qvel[self.v[:5]],self.d.ctrl[:5]]

    def collect(self, path, episodes, ticks, seed):
        out = np.lib.format.open_memmap(path,'w+',dtype=np.float32,shape=(episodes,ticks,35))
        kernel()[0](*self.common,episodes,ticks,seed,out.ctypes.data)
        out.flush()

    def eval(self, commands):
        u = np.ascontiguousarray(commands,dtype=float)
        if u.ndim != 3 or u.shape[2] != 5:
            raise ValueError('Expected batch of five-axis normalized commands')
        out = np.empty((len(u),22))
        kernel()[1](*self.common,u.ctypes.data,len(u),u.shape[1],out.ctypes.data)
        return out


def prepare(episodes):
    ROOT.mkdir(parents=True,exist_ok=True)
    ready = tt72.receipt()[0][0.]
    plan = dict(schema='tt88-babble-1', ready=ready, episodes=episodes, ticks=240, train_seed=880021,
                held_seed=888017, held_episodes=12, code_sha256=_hash(NATIVE),
                input='Five stock articulated joint positions and velocities, issued reference, next signed rates.',
                target='Physical next joint position and velocity plus issued reference, even on exploratory misses. '
                       'No ball or far-side body. Intact dynamic learner camera/gripper and free welded paddle.')
    if PLAN.exists() and _read(PLAN)!=plan:
        raise ValueError('Motor babble plan changed')
    if not PLAN.exists():
        _write(PLAN,plan)
    with mpc.centered(ready):
        a = mpc.arena(width=64,height=48)
        try:
            a.reset(render=False)
            body = Body(a)
            for p,n,seed in ((HOLD,plan['held_episodes'],plan['held_seed']),(DATA,episodes,plan['train_seed'])):
                if p.exists():
                    continue
                start = perf_counter()
                tmp = p.with_suffix('.tmp.npy')
                body.collect(tmp,n,plan['ticks'],seed)
                tmp.replace(p)
                print(p,n*plan['ticks'],round(perf_counter()-start,1),flush=True)
        finally:
            a.close()


MODEL = ROOT / 'motor.npz'
BENCH = ROOT / 'body-benchmark.json'
POS_SCALE, FACE_SCALE, VEL_SCALE = .025, .19, .32


class PlantFit:
    """Contact-free one-tick motor data, used without native rollouts by the inverse."""

    def __init__(self, body, model=MODEL):
        self.lo, self.hi = body.lo[:5], body.hi[:5]
        self.stride = body.stride[:5]
        self.kp, self.kv, self.force = body.kp, body.kv, body.force
        rng = np.random.default_rng(880037)
        basic = self.encode(np.zeros((1,15)),np.zeros((1,5))).shape[1]
        # Fixed random features and a ridge regression, no backpropagation.
        self.w = (rng.normal(size=(basic,208)) / np.sqrt(basic)).astype('f')
        self.bias = rng.uniform(-2.,2.,size=208).astype('f')
        ids = np.arange(basic)
        self.i, self.j = np.triu_indices(basic,k=0)
        normal = (ids < 35)
        keep = normal[self.i] & normal[self.j]
        self.i, self.j = self.i[keep], self.j[keep]
        if model.exists():
            with np.load(model) as data:
                self.coeff = data['coeff']
        else:
            self.coeff = None

    def encode(self, start, u):
        q, v, ref = np.split(start,3,axis=1)
        upper = self.force * np.clip(1.-v/NOMINAL_NO_LOAD,0.,1.)
        lower = -self.force * np.clip(1.+v/NOMINAL_NO_LOAD,0.,1.)
        drive = np.clip(ref+.61*self.stride*u,self.lo,self.hi)
        q_mid = q+.34*DT*v
        demand = self.kp*(drive-q_mid)-self.kv*v
        effective = np.clip(demand,lower,upper)/self.force
        limit_a = np.clip((self.lo - q - .78*DT*v)/.055,-1.,1.)
        limit_b = np.clip((q + .78*DT*v - self.hi)/.055,-1.,1.)
        return np.c_[(q-(self.lo+self.hi)/2.)/((self.hi-self.lo)/2.),
                     v/3.8, np.clip((ref-q)/.34,-3.,3.), u/.8,
                     effective, limit_a, limit_b, np.sin(q), np.cos(q), np.clip(demand/self.force,-2.3,2.3)].astype('f')

    def features(self, start, u):
        f = self.encode(start,u)
        return np.c_[np.ones(len(f),dtype='f'),f,f[:,self.i]*f[:,self.j],np.tanh(f@self.w+self.bias)]

    def fit(self, data):
        d = data.reshape(-1,35)
        d = d[np.isfinite(d).all(axis=1)]
        d = d[np.max(abs(d[:,25:30]),axis=1)<9.5]
        starts = d[:,:15]
        # Give fast changes, close-limit moves, and useful body motion weight
        # without discarding static holds or all failed explorations.
        impulse = np.linalg.norm(d[:,25:30]-d[:,5:10],axis=1)
        delta = np.c_[d[:,25:30]-d[:,5:10],
                   25.*(d[:,20:25]-d[:,:5]-.5*DT*(d[:,25:30]+d[:,5:10]))]
        dim = self.features(starts[:1],d[:1,15:20]).shape[1]
        gram, right = np.zeros((dim,dim)), np.zeros((dim,10))
        for i in range(0,len(d),4096):
            j = min(i+4096,len(d))
            weight = np.sqrt(.75 + np.minimum(impulse[i:j],2.5)*.32)[:,None]
            f = self.features(starts[i:j],d[i:j,15:20])*weight
            gram += f.T @ f
            right += f.T @ (weight*delta[i:j])
        self.coeff = np.linalg.solve(gram+np.diag(np.r_[.1,np.full(dim-1,.27)]),right).astype('f')
        return len(d)

    def next(self, start, u):
        q, v, ref = np.split(start,3,axis=1)
        dv, correction = np.split(self.features(start,u)@self.coeff,2,axis=1)
        nv = np.clip(v+dv,-7.,7.)
        nq = np.clip(q+DT*.5*(v+nv)+.04*correction,self.lo-.004,self.hi+.004)
        return np.c_[nq,nv,np.clip(ref+self.stride*u,self.lo,self.hi)]

    def references(self, state, choices):
        """Learned propagation of a batch of bounded absolute-reference programs."""
        start = np.broadcast_to(state,(len(choices),15)).copy()
        commands = np.empty_like(choices)
        for t in range(choices.shape[1]):
            u = np.clip((np.clip(choices[:,t],self.lo,self.hi)-start[:,10:])/self.stride,-.8,.8)
            commands[:,t] = u
            start = self.next(start,u)
        return start[:,:10],commands


def coordinates(n, uv):
    """Offset within the blade surface in world-upright tangent coordinates."""
    up = np.array([0.,0.,1.]) - n*n[2]
    if np.linalg.norm(up)<1e-5:
        up=np.array([0.,1.,0.])-n*n[1]
    up /= np.linalg.norm(up)
    return uv[0]*np.cross(n,up)+uv[1]*up


def world_goal(end, uv):
    p,n,w,v = np.split(end[10:],[3,6,9])
    offset = coordinates(n,uv)
    return np.r_[p+offset,n,v+np.cross(w,offset)]


class Inverse:
    """Map encoder state + feasible future blade patch to smooth stock commands."""

    def __init__(self, body, arena, model=MODEL):
        self.motor = PlantFit(body,model)
        self.ik = tt72.FullPose(arena)

    def features(self, joints, uv):
        q, v = joints[:5], joints[5:10]
        p,R,J,W = self.ik.frame(q,True)
        normal = R[:,2]
        offset = coordinates(normal,uv)
        return np.r_[p+offset,normal,J@v+np.cross(W@v,offset)]

    def attainable(self, state, target, uv, ticks):
        """Physical geometry maps a requested patch to possibly distinct arm modes."""
        q0 = state[:5]
        desired_n = target[3:6]
        initial = [q0,q0+np.array([.44,-.38,.43,-.33,.40]),
                   q0+np.array([-.44,.38,-.43,.33,-.40]),
                   q0+np.array([.23,-.60,.79,.43,-.60])]
        center = target[:3]-coordinates(desired_n,uv)
        rows=[]
        for start in initial:
            q,_ = self.ik.fit_frame(center,
                      self.frame_for(desired_n),start,tangent=.0007,stay=.0021,n=50)
            _,R,J,W = self.ik.frame(q,True)
            at = coordinates(R[:,2],uv)
            Y = J+np.cross(W.T,at).T
            v = Y.T @ np.linalg.solve(Y@Y.T+.018**2*np.eye(3),target[6:9])
            v = np.clip(v,-5,5)
            v[(q<self.motor.lo+.007)&(v<0)] *= .15
            v[(q>self.motor.hi-.007)&(v>0)] *= .15
            outcome = self.features(np.r_[q,v],uv)
            geom = np.linalg.norm((outcome-target)/np.repeat([POS_SCALE,FACE_SCALE,VEL_SCALE],3))
            travel = np.linalg.norm((q-q0)/(ticks*DT*3.6+.08))
            if not any(np.linalg.norm(q-r[1])<.10 for r in rows):
                rows.append((geom+.18*travel,q,v))
        return sorted(rows,key=lambda z:z[0])[:2]

    @staticmethod
    def frame_for(n):
        u=coordinates(n,(1.,0.))
        return np.column_stack([u,np.cross(n,u),n])

    def nominal(self, state, q, v, ticks, lead=.042):
        """A conventional kinematics/servo baseline, with a fixed latency lead."""
        t = np.arange(1,ticks+1)/ticks
        total = ticks*DT
        path = ((2*t**3-3*t*t+1)[:,None]*state[:5] + (3*t*t-2*t**3)[:,None]*q +
                total*((t**3-2*t*t+t)[:,None]*state[5:10] + (t**3-t*t)[:,None]*v))
        dx = ((6*t*t-6*t)[:,None]*state[:5] + (6*t-6*t*t)[:,None]*q)/total
        dx += (3*t*t-4*t+1)[:,None]*state[5:10] + (3*t*t-2*t)[:,None]*v
        return np.clip(path+lead*dx,self.motor.lo,self.motor.hi)

    def solve(self, state, desired, uv, ticks, rounds=3):
        """Gauss–Newton over the learned motor fit, not the live/native motor."""
        fits=self.attainable(state,desired,uv,ticks)
        if not fits:
            raise ValueError('Body atlas returned no geometry candidate')
        candidates=[]
        baseline=self.nominal(state,*fits[0][1:],ticks)
        tau=np.arange(1,ticks+1)/ticks
        basis=np.c_[np.minimum(1.,1.5*tau),1.40*(tau**3-.22*tau),4*tau*(1.-tau)]
        epsilon=.046
        axis=epsilon*np.eye(15)
        residual_scale=np.repeat([POS_SCALE,FACE_SCALE,VEL_SCALE],3)

        def predict(theta, ref):
            g=ref+np.einsum('tb,nbj->ntj',basis,theta.reshape(-1,3,5))
            reached,commands=self.motor.references(state,g)
            effects=np.array([self.features(z,uv) for z in reached])
            return (effects-desired)/residual_scale,commands

        for _,q,v in fits:
            ref=self.nominal(state,q,v,ticks)
            theta=np.zeros(15)
            y, u=predict(theta[None],ref)
            best=float(np.linalg.norm(y))
            best_u=u[0]
            for k in range(rounds):
                err,_=predict(theta[None]+axis,ref)
                J=(err-y)/epsilon
                direction=-J @ np.linalg.solve(J.T@J+.28*np.eye(9),y[0])
                steps=np.clip(theta+np.array([.55,1.,1.55])[:,None]*np.clip(direction,-.27,.27),-.72,.72)
                trial,program=predict(steps,ref)
                loss=np.linalg.norm(trial,axis=1)
                i=int(np.argmin(loss))
                if loss[i]>=best:
                    break
                theta, y, best, best_u = steps[i], trial[i:i+1], float(loss[i]), program[i]
            candidates.append((best,best_u))
        return min(candidates,key=lambda r:r[0])[1],self.clamp_program(state,baseline)

    def clamp_program(self,state,refs):
        current=state[10:].copy()
        out=[]
        for ref in refs:
            u=np.clip((ref-current)/self.motor.stride,-.8,.8)
            current=np.clip(current+self.motor.stride*u,self.motor.lo,self.motor.hi)
            out.append(u)
        return np.asarray(out)


def fit():
    p=_read(PLAN)
    with mpc.centered(p['ready']):
        a=mpc.arena(width=64,height=48)
        try:
            plant=PlantFit(Body(a))
            n=plant.fit(np.load(DATA,mmap_mode='r'))
            np.savez_compressed(MODEL,coeff=plant.coeff,
                                source=np.asarray(_hash(DATA)),protocol=np.asarray(_hash(PLAN)))
            d=np.load(HOLD).reshape(-1,35)
            pred=plant.next(d[:,:15],d[:,15:20])
            print('fit',n,'held one-tick joint err',np.median(np.linalg.norm(pred[:,:5]-d[:,20:25],axis=1)),
                  'velocity err',np.median(np.linalg.norm(pred[:,5:10]-d[:,25:30],axis=1)),flush=True)
        finally:
            a.close()


def move(body,controls):
    limit=body.m.actuator_forcerange.copy()
    force=body.m.actuator_forcerange
    try:
        for u in controls:
            for _ in range(body.itr):
                body.d.ctrl[:5]=np.clip(body.d.ctrl[:5]+u[:5]*body.step[:5],body.lo[:5],body.hi[:5])
                motion=body.d.qvel[body.v]/NOMINAL_NO_LOAD
                force[:,0]=limit[:,0]*np.clip(1.+motion,0.,1.)
                force[:,1]=limit[:,1]*np.clip(1.-motion,0.,1.)
                mujoco.mj_step(body.m,body.d)
    finally:
        force[:]=limit


def errors(v, target, uv):
    w=world_goal(v,uv)
    n=max(-1.,min(1.,w[3:6]@target[3:6]))
    return dict(joint_rms=None,pos_m=float(np.linalg.norm(w[:3]-target[:3])),
                angle_deg=float(np.rad2deg(np.arccos(n))),velocity_mps=float(np.linalg.norm(w[6:]-target[6:])),
                weighted=float(np.linalg.norm((w-target)/np.repeat([POS_SCALE,FACE_SCALE,VEL_SCALE],3))),
                achieved=w.tolist())


def compare_task(a,body,agent,oracle,uv,ticks,meta):
    y=body.eval(oracle[None])[0]
    desired=world_goal(y,uv)
    state=body.state
    start=perf_counter()
    plan, nominal=agent.solve(state,desired,uv,ticks)
    elapsed=perf_counter()-start
    reached=body.eval(np.stack([nominal,plan]))
    out=dict(**meta,ticks=ticks,state=state.tolist(),target=desired.tolist(),patch=uv.tolist(),
             resolution=['p_x','p_y','p_z','signed_n_x','signed_n_y','signed_n_z','patch_v_x','patch_v_y','patch_v_z'],
             learned_query_s=elapsed,held_attaining_rates=oracle.tolist(),
             learned_rates=plan.tolist(),baseline_rates=nominal.tolist())
    for k,z in zip(('baseline','learned'),reached):
        r=errors(z,desired,uv)
        r['joint_rms']=float(np.sqrt(np.mean((z[:5]-y[:5])**2)))
        r['joint_velocity_rms']=float(np.sqrt(np.mean((z[5:10]-y[5:10])**2)))
        out[k]=r
    return out


def bench():
    p=_read(PLAN)
    if np.load(MODEL)['protocol'].item()!=_hash(PLAN):
        raise ValueError('Plant model and training protocol differ')
    with mpc.centered(p['ready']),tt72.lateral.fixture.school.enabled(cameras=False):
        a=mpc.arena(width=64,height=48)
        body=Body(a)
        inv=Inverse(body,a)
        signature=dict(protocol=_hash(PLAN),model=_hash(MODEL),held_sha256=_hash(HOLD))
        report=_read(BENCH) if BENCH.exists() else dict(**signature,complete=False,rows=[])
        if any(report.get(k)!=v for k,v in signature.items()):
            raise ValueError('The benchmark inputs changed')
        if report['complete']:
            a.close()
            return
        try:
            a.reset(render=False)
            d=np.load(HOLD,mmap_mode='r')
            # Whole held motor episodes are release-disjoint and independent
            # of fitting. Predeclared episode offsets, horizons and patches.
            rng=np.random.default_rng(889913)
            for i,ep in enumerate(d):
                for trial,start in enumerate((26,71,132,185)):
                    body.read(a)
                    move(body,ep[:start,15:20])
                    h=(6,9,13,17)[(trial+i)%4]
                    uv=rng.uniform(-.035,.035,2)
                    if any(x['kind']=='held-babble' and x['episode']==i and x['start']==start for x in report['rows']):
                        continue
                    r=compare_task(a,body,inv,ep[start:start+h,15:20],uv,h,
                                   dict(kind='held-babble',episode=i,start=start))
                    report['rows'].append(r)
                _write(BENCH,report)
                print('attainment held episode',i,'baseline/model med',
                      [round(float(np.median([x[tag]['weighted'] for x in report['rows']])),3)
                       for tag in ('baseline','learned')],flush=True)
            # Only the public command tape establishes an exact attained body
            # target. Exclude any ball-hit substep from its generated target.
            stock=_read(tt72.ROOT/'equivalence.json.gz')
            row=next(r for r in stock['rows'] if r['case']['id']==9254)
            spec=row['case']; src=_read(Path(spec['source']))
            c,old,anchor=tt72.qual.get(src,spec)
            physical=[r for r in row['physically_tested'] if r['contact'] and r['contact']['radial_m']<.058
                      and not r['clearance']['penetration_samples']]
            for index,r in enumerate(physical):
                hit=int(np.floor(r['contact']['time_s']/DT))-1
                for ticks in (6,10,15):
                    start=hit-ticks
                    if any(x['kind']=='9254-attained' and x['episode']==index and x['start']==start for x in report['rows']):
                        continue
                    tt72.tape.setup(a,c,old,anchor)
                    for u in r['commands'][:start]:
                        a.exchange(u,mpc.FAR_COMMAND)
                    body.read(a)
                    u=np.asarray(r['commands'][start:hit])[:,:5]
                    target=np.asarray(body.eval(u[None])[0])
                    for y in r['commands'][start:hit]:
                        a.exchange(y,mpc.FAR_COMMAND)
                    read=tt72.blade(a)
                    exact=np.r_[read['q'][:5],read['qd'][:5],read['p'],np.asarray(read['R'])[:,2],read['twist']]
                    # The integration state maps exactly. The full arena
                    # geom transform is the last substep, unlike mj_forward;
                    # position error is reported as well as encoder parity.
                    q_check=float(np.linalg.norm(exact[:5]-target[:5]))
                    if q_check>.0025:
                        raise ValueError(f'Reduced motor does not reproduce full arm, q error {q_check:.5f}')
                    r0=compare_task(a,body,inv,u,np.array([.024,-.014]),ticks,
                        dict(kind='9254-attained',episode=index,start=start,full_native_error_q=q_check,
                             full_native_error_p=float(np.linalg.norm(exact[10:13]-target[10:13]))))
                    report['rows'].append(r0)
                    _write(BENCH,report)
                    print('9254 attained horizon',index,ticks,round(r0['baseline']['weighted'],3),
                          round(r0['learned']['weighted'],3),flush=True)
            report['complete']=True
            _write(BENCH,report)
        finally:
            a.close()


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('babble','fit','bench'))
    parser.add_argument('--episodes',type=int,default=640)
    args=parser.parse_args()
    {'babble':lambda:prepare(args.episodes),'fit':fit,'bench':bench}[args.stage]()
