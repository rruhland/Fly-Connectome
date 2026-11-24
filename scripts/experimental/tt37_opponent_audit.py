"""Check deterministic live replies, unassisted flight, native contacts and CAD clearance."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt37_opponent as live
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_arm_clearance import ArmClearance
from .tt29_endurance import audit
from .tt29_partner import CARRIAGE
from .tt29_rally_physics import mujoco
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_opponent_benchmark import ARTIFACTS, OUT
from .tt37_opponent_capacity import CARRIAGE_NEAR, CapacityArena, point as capacity_point


class CourtClearance:
    """Full link CAD and explicit moving blade/mount distances to playing fixtures."""

    def __init__(self, model, *, articulated, two_carriages=False):
        self.model=model
        self.tolerance=0.
        self.arm=ArmClearance(model,tolerance=0.) if articulated else None
        names=[model.geom(i).name for i in range(model.ngeom)]
        self.surfaces=[i for i,n in enumerate(names) if n in ('table','table_right','net')]
        self.moving=[i for i,n in enumerate(names) if n in ('rival_racket','tt29_partner_racket_mount')
                     or (two_carriages and n=='learner_racket')]

    def distances(self, shadow, *, horizon):
        mujoco.mj_kinematics(self.model,shadow)
        if self.arm is not None:
            yield from self.arm.distances(shadow,horizon=horizon,update=False)
        for geom in self.moving:
            for surface in self.surfaces:
                distance=mujoco.mj_geomDistance(self.model,shadow,geom,surface,horizon,None)
                if distance<horizon:
                    yield self.model.geom(geom).name,self.model.geom(surface).name,distance


def guarded(arena, clearance):
    arena.sample_clearance(False)
    arena._cad_reader=clearance,mujoco.MjData(clearance.model)


def contact_observer(arena, contacts):
    original=arena._racket_contact

    def note(side, location, incoming):
        own=arena._racket_ids[side]
        ball=arena._geom_ids['ball']
        matching=[c for c in arena.data.contact if c.efc_address>=0 and set(c.geom)=={own,ball}]
        if not matching:
            raise ValueError('Referee racket event did not come from the native solver')
        contact=min(matching,key=lambda c:np.linalg.norm(c.pos-location))
        normal=contact.frame[:3]*(1 if int(contact.geom[0])==own else -1)
        face=float(normal@arena.data.geom_xmat[own].reshape(3,3)[:,2])
        body={'player':arena.model.body(arena.model.geom(own).bodyid[0]).name, 'side':side}
        contacts.append(dict(time_s=float(arena.data.time), signed_face_alignment=face, location=list(location), **body))
        return original(side,location,incoming)
    return patch.object(arena,'_racket_contact',note)


def baseline(path):
    data=_read(path)
    protocol=data['protocol']
    rows=data['rows']
    if len(rows)!=len(data['cards']) or 'cpu_s' not in data:
        raise ValueError('Frozen-pupil run is incomplete')
    longest=max(range(len(rows)),key=lambda i:rows[i]['score']['returns'])
    misses=[i for i,r in enumerate(rows) if any(not u['answered'] for u in r['physical']['later_replies'])]
    split=next(i for i,p in enumerate(data['pools']) if p=='held_replies')
    IDs=list(dict.fromkeys([longest,*(misses[:1]),split]))
    audited=[]
    with centered(data['ready']):
        a=live.arena(protocol=protocol)
        shadow=live.arena(protocol=protocol,detail='cad')
        clearance=CourtClearance(shadow.model,articulated=True)
        try:
            for i in IDs:
                c,r=data['cards'][i],rows[i]
                a.launch(r['shot'],card=c)
                a.partner.reset(seed=c['seed'])
                contacts=[]
                guarded(a,clearance)
                with contact_observer(a,contacts):
                    for rate in r['actions']:
                        a.exchange(rate,FAR_COMMAND)
                expected=tuple(digest_json(obj) for obj in (r['events'],r['placements'],r['score']))
                now=tuple(digest_json(obj) for obj in (a.events,a.point_placements(),audit(a.events,opening_side=1)))
                verify_clearance(a)
                if expected!=now:
                    raise ValueError(f'Action replay diverged for {c["id"]} {protocol}')
                if len(contacts)!=sum(e['kind']=='contact' for e in a.events):
                    raise ValueError('Native contact count differs')
                if any((p['side']==-1) != (p['player']==CARRIAGE) for p in contacts):
                    raise ValueError('Hit did not come from the assigned physical body')
                audited.append(dict(id=c['id'],key=c['key'],phase=data['pools'][i],contacts=contacts,hashes=now,
                                    clearance=a.clearance_result(),simulation_end_s=float(a.data.time)))
        finally:
            a.close()
            shadow.close()
    return dict(protocol=protocol,source_sha256=_hash(path),replays=audited)


def verify_clearance(a):
    if a.clearance_result()['samples']!=round(a.data.time/a.model.opt.timestep):
        raise ValueError('Not all solver states were checked')


def capacity(path):
    data=_read(path)
    index=max(range(len(data['rows'])),key=lambda i:data['rows'][i]['score']['returns'])
    c,r=data['cards'][index],data['rows'][index]
    settings=dict(protocol=data['protocol'],near_mode=data['near_mode'],cards=[],strike_ready='lateral',width=192,height=144,
                  clearance_guard=False,fast_search=True,max_steps=live.MAX_CONTROL_STEPS,rate_scale=.8,
                  velocity_guard=True,motor_envelope=True,guard_table=True,net_rescue=True)
    a=CapacityArena(**settings)
    shadow=CapacityArena(**settings,detail='cad')
    contacts=[]
    try:
        clearance=CourtClearance(shadow.model,articulated=False,two_carriages=True)
        before=a.launch

        def launch(*args,**kwargs):
            first=before(*args,**kwargs)
            guarded(a,clearance)
            return first

        with patch.object(a,'launch',launch), contact_observer(a,contacts):
            replay=capacity_point(a,r['shot'],c,data['cap'])
        if any(digest_json(r[k])!=digest_json(replay[k]) for k in ('events','placements','score')):
            raise ValueError('Two-carriage holdout replay diverged')
        verify_clearance(a)
        if len(contacts)!=sum(e['kind']=='contact' for e in a.events) or any(
                p['player']!=(CARRIAGE_NEAR if p['side']==1 else CARRIAGE) for p in contacts):
            raise ValueError('Capacity contact was not from its marked carriage')
        return dict(id=c['id'],source_sha256=_hash(path),near=data['near_mode'],protocol=data['protocol'],
                    returns=replay['score']['returns'],simulation_end_s=replay['simulation_end_s'],
                    clearance=a.clearance_result(),contacts=contacts,events_sha256=digest_json(replay['events']))
    finally:
        a.close()
        shadow.close()


def archived_control():
    proof=ARTIFACTS/'tt29'/'coupled-camera200-audit.json'
    tape=ARTIFACTS/'tt29'/'envelope-extended.json.gz'
    faces=ARTIFACTS/'tt31'/'previous-cooperative-faces.json'
    audit=_read(proof)
    if audit['source_sha256']!=_hash(tape) or not audit['exact_events'] or not audit['exact_score']:
        raise ValueError('Earlier cooperative proof does not match the tape')
    return dict(tape_sha256=_hash(tape),replay_sha256=_hash(proof),faces_sha256=_hash(faces),
                release=audit['shot_id'],score=audit['score'],clearance=audit['clearance'],
                events_sha256=audit['events_sha256'],event_count=audit['event_count'],
                first_returns=_read(faces)['summary'])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('output',type=Path,default=OUT,nargs='?')
    p.add_argument('--capacity',type=Path)
    args=p.parse_args()
    if args.capacity:
        result=capacity(args.capacity)
        _write(args.output/'opponent-capacity-audit.json',result)
    else:
        files=sorted(args.output.glob('opponent-baseline-*-long.json.gz'))
        _write(args.output/'opponent-audit.json',dict(cooperative=archived_control(),arms=[baseline(r) for r in files]))
