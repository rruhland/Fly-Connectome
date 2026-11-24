"""Opponent capacity control with a second finite-mass paddle instead of the SO-101 blade."""

import argparse
from pathlib import Path
from time import process_time
import xml.etree.ElementTree as xml

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_partner import CARRIAGE, MASS, INERTIA, _drive
from .tt29_partner_fast import FastPracticePartner
from .tt29_rally_physics import measured_launches, mujoco
from .tt37_opponent import LONG_CAP, MAX_CONTROL_STEPS, LiveArena
from .tt37_opponent_benchmark import PLAN, OUT, mark


CARRIAGE_NEAR = 'tt37_capacity_near'
ROT = np.array([-1.,-1.,1.])
QUAT = np.array([0.,0.,0.,1.])
CENTRAL = (0., -1.27, 1.05, 3.60)
WIDE = ((-.48,.48),(-1.32,-1.04),(.96,1.15),(2.9,4.35))


def rotated(q):
    out = np.empty(4)
    mujoco.mju_mulQuat(out,QUAT,q)
    return out


class MirroredFlight:
    def __init__(self, court):
        self.court = court

    def __getattr__(self,name):
        return getattr(self.court,name)

    @property
    def ball(self):
        return self.court.ball*ROT

    @property
    def velocity(self):
        return self.court.velocity*ROT

    @property
    def spin(self):
        return self.court.spin*ROT


class SyntheticReceiver(FastPracticePartner):
    """Privileged diagnostic striker, not a trainable or articulated pupil."""

    def __init__(self, court):
        super().__init__(court,targets=[CENTRAL],guard_table=True)
        self.court = court
        self.arena = MirroredFlight(court)
        j = court.model.joint(CARRIAGE_NEAR)
        self.qpos,self.qvel = int(j.qposadr[0]),int(j.dofadr[0])
        self.body = int(court.model.body(CARRIAGE_NEAR).id)
        self.forces = np.zeros(6)

    def goal(self):
        if self.court.near_mode=='central':
            return np.array(CENTRAL)
        bounds = np.array(WIDE)
        seed = [self.court.card['seed'],self.reception,374728]
        return np.random.default_rng(seed).uniform(bounds[:,0],bounds[:,1])

    def advance(self):
        a=self.court
        p=a.data.qpos[self.qpos:self.qpos+7]
        q=np.r_[p[:3]*ROT,rotated(p[3:])]
        v=a.data.qvel[self.qvel:self.qvel+6].copy()
        v[:3] *= ROT
        now = float(a.data.time)
        if a._last_hitter==-1 and a.velocity[1]<-.32:
            if self.reception!=a.rival_contacts:
                self.reception = a.rival_contacts
                self.target = self.goal()
                self.plan_stage,self.stroke = 0,None
            stage = 1+int(a.ball[1]<.20)+int(a.ball[1]<-.40)
            obstacles = a._racket_ids[-1] in a._previous_touches or a._geom_ids['table'] in a._previous_touches
            if not obstacles and self.stroke is None and stage>self.plan_stage:
                self.plan_stage = stage
                self.stroke, rec = self._plan(self.target,q[:3],v[:3],now)
                self.reports.append(dict(time_s=now,stage=stage,reception=self.reception,**rec))
        elif self.stroke is not None and now>self.stroke.start+self.stroke.eta+.15:
            self.stroke=None
        _drive(self.forces,q,v,now,self.impact.dt,self.stroke,self.home,self.home_quat)
        self.court.data.xfrc_applied[self.body,:3] = self.forces[:3]*ROT
        self.court.data.xfrc_applied[self.body,3:] = self.forces[3:]*ROT


class CapacityArena(LiveArena):
    def __init__(self, *, near_mode, **settings):
        self.near_mode = near_mode
        self.receiver = None
        super().__init__(**settings)
        self.receiver = SyntheticReceiver(self)

    def reset(self,*args,**kwargs):
        result=super().reset(*args,**kwargs)
        if self.receiver is not None:
            self.receiver.reset(seed=self.seed)
        return result

    def _model_document(self):
        document=super()._model_document()
        world=document.find('worldbody')
        base=world.find("body[@name='learner_base']")
        # This is an opponent-only feasibility control, with the pupil's collision blade replaced.
        parent=next(x for x in world.iter('body') if x.find("geom[@name='learner_racket']") is not None)
        blade=parent.find("geom[@name='learner_racket']")
        parent.remove(blade)
        if parent.find('inertial') is None:
            xml.SubElement(parent,'inertial',pos='0 0 0',mass='.13',diaginertia='.0002 .0002 .0004')
        for g in (*base.iter('geom'), *parent.iter('geom')):
            g.attrib.update(rgba='0 0 0 0',contype='0',conaffinity='0')
            g.attrib.pop('material',None)
        far=world.find(f"body[@name='{CARRIAGE}']")
        stringify=lambda p:' '.join(map(str,p))
        p=np.fromstring(far.get('pos'),sep=' ')*ROT
        quat=rotated(np.fromstring(far.get('quat'),sep=' '))
        mirror=xml.SubElement(world,'body',name=CARRIAGE_NEAR,pos=stringify(p),quat=stringify(quat),gravcomp='1')
        xml.SubElement(mirror,'freejoint',name=CARRIAGE_NEAR)
        xml.SubElement(mirror,'inertial',pos='0 0 0',mass=str(MASS),diaginertia=stringify([INERTIA]*3))
        blade.attrib.update(pos='0 0 0',quat='1 0 0 0')
        mirror.append(blade)
        return document

    def _aerodynamics(self):
        if self.receiver is not None:
            self.receiver.advance()
        super()._aerodynamics()


def point(a, shot, card, cap):
    a.launch(shot,card=card)
    a.partner.reset(seed=card['seed'])
    a.receiver.reset(seed=card['seed'])
    near, far = np.zeros(a.action_size), np.zeros(5)
    steps = 0
    while True:
        _, _, done, truncated, _ = a.exchange(near,far)
        steps += 1
        if done or truncated or a.legal_returns+a.rival_returns-1>=cap:
            break
    out = dict(shot=shot, opponent_seed=card['seed'], score=audit(a.events,opening_side=1),
        placements=a.point_placements(), events=a.events, actions=[near.tolist()]*steps,
        near_plans=a.receiver.reports, control_steps=steps)
    mark(a,out,cap)
    return out


def run(feeds, output, mode, protocol, cap, *, witness=False):
    defaults = dict(strike_ready='lateral',width=192,height=144,clearance_guard=False,
                    fast_search=True,max_steps=MAX_CONTROL_STEPS,rate_scale=.8,velocity_guard=True,
                    motor_envelope=True,guard_table=True,net_rescue=True)
    court = CapacityArena(protocol=protocol,cards=[],near_mode=mode,**defaults)
    plan = _read(PLAN)
    shots = measured_launches(feeds)
    if _hash(feeds)!=plan['feed_sha256']:
        raise ValueError('Measured feeds changed')
    cards=[plan['held_replies'][i] for i in (0,4,8,12)]
    if witness:
        cards=cards[:1]
    rows=[]
    target=output/('opponent-capacity-witness.json.gz' if witness else f'opponent-capacity-{mode}-{protocol}.json.gz')
    started=process_time()
    try:
        for n,c in enumerate(cards):
            row=point(court,shots[c['id']],c,cap)
            rows.append(row)
            print(mode,protocol,c['id'],row['score']['returns'],row['score']['learner'],row['stop'],flush=True)
            _write(target,dict(near_mode=mode,protocol=protocol,cap=cap,cards=cards[:n+1],rows=rows,
                               wide_bounds=WIDE,central_target=CENTRAL,cpu_s=process_time()-started))
    finally:
        court.close()


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('feeds',type=Path)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--near',choices=('central','wide'),default='central')
    p.add_argument('--protocol',choices=('random','tt36','fixed'),default='random')
    p.add_argument('--cap',type=int,default=LONG_CAP)
    p.add_argument('--witness',action='store_true',help='run only the first planned control point')
    args=p.parse_args()
    run(args.feeds,args.output,args.near,args.protocol,args.cap,witness=args.witness)
