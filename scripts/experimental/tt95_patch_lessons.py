"""Acquire full blade patch velocity from physically rewarded camera/encoder experience."""

import argparse
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from . import tt39_terminal_examples as old
from . import tt88_inverse_body as inv
from . import tt90_installation_learning as tt90
from .tt4_transfer_trial import _hash,_read,_write
from .tt19_view_full_game import digest_json

ROOT=Path('artifacts/tt95_patch_velocity')
PLAN=ROOT/'protocol.json'
PRACTICE=ROOT/'natural-practice25.json.gz'
MOUNTED=ROOT/'mounted-labels.json.gz'
STOCK=ROOT/'stock-labels.json.gz'
MODEL=ROOT/'lessons.npz'
N=25
DT=inv.DT


def freeze():
    ROOT.mkdir(exist_ok=True,parents=True)
    contract=tt90.contract(tt90.FEEDS)
    if not PRACTICE.exists():
        x=_read(tt90.path('practice'))
        if len(x['rows'])<N or x['cards'][:N]!=contract['learn'][:N]:
            raise ValueError('Wait for the exact first 25 natural mounted practice points')
        data=dict(source=str(tt90.path('practice')),source_prefix_rows=N,protocol=x['protocol'],body=x['body'],
                  cards=x['cards'][:N],rows=x['rows'][:N])
        _write(PRACTICE,data)
    exp=old.OUT/'empirical-endpoints.json'
    d=_read(PRACTICE)
    banned={c['id'] for c in contract['assess']+contract['hard_pool']} | set(contract['excluded'])
    if {c['id'] for c in d['cards']}&banned:
        raise ValueError('Mounted lessons overlap immutable assessment releases')
    protocol=dict(schema='tt95-patch-velocity-1',mounted_practice=_hash(PRACTICE),stock_receipt=_hash(exp),
                  stock_sensor_file=_hash(old.OBSERVATIONS),mounting=_hash(tt90.system.BUILD),
                  training=dict(reserve_forbidden=sorted(banned),mounted=d['cards']),
                  premise='Truth identifies the instant and exact surface patch of actual credited returns only. '
                          'The target velocity is derived from the preceding observed joint differences, not a '
                          'teacher action or true ball state exposed at play. Camera contexts are actual paid histories.')
    if not PLAN.exists():
        _write(PLAN,protocol)
    if _read(PLAN)!=protocol:
        raise ValueError('Changed frozen patch-profile acquisition')
    return protocol


def tangent(normal,r):
    return r-normal*(r@normal)


def coords(n,r):
    return np.array([inv.coordinates(n,(1,0))@r,inv.coordinates(n,(0,1))@r])


def projected(x,dt):
    x=np.asarray(x,dtype=float)
    d=np.clip(dt,0,.085)
    return np.r_[x[:3]+d*x[3:6]+np.array([0.,0.,-4.905*d*d]),x[3:6]+[0.,0.,-9.81*d]]


def encoder_at(times,readings,t):
    """Use only encoder samples that have physically occurred before the label."""
    grid=np.asarray(times)
    q=np.asarray(readings,dtype=float)[:,:5]
    i=int(np.searchsorted(grid,t+1e-9,side='right'))-1
    if i<1 or grid[i]-grid[i-1]>.026:
        return None
    d0=grid[i]-grid[i-1]
    omega=(q[i]-q[i-1])/d0
    qdot=omega.copy()
    if i>=2 and grid[i-1]-grid[i-2]<.025:
        d1=grid[i-1]-grid[i-2]
        old=(q[i-1]-q[i-2])/d1
        dd=np.clip((omega-old)/(.5*(d0+d1)),-65.,65.)
        # Taper extrapolation around actuator reversal and physical hard stops.
        qdot=np.clip(omega+.58*(.5*d0+t-grid[i])*dd,-5.9,5.9)
    return q[i]+(t-grid[i])*.5*(omega+qdot),qdot


def label(ik,q,w,location,optical_at,contact=None):
    p,R,J,W=ik.frame(q,True)
    n=R[:,2].copy()
    facing=1. if n[1]>=0 else -1.
    n*=facing
    r=tangent(n,np.asarray(location)-p)
    v=J@w+np.cross(W@w,r)
    patch=p+r
    rel=patch-np.asarray(optical_at[:3])
    data=dict(optical=np.asarray(optical_at).tolist(),patch_relative=rel.tolist(),normal=n.tolist(),
              signed_face=facing,patch_coordinates=coords(n,r).tolist(),
              velocity=v.tolist(),patch_radius_m=float(np.linalg.norm(r)),joint=q.tolist())
    if contact is not None:
        P,R0=np.asarray(contact['p']),np.asarray(contact['R'])
        T=np.asarray(contact['twist'])
        rr=tangent(R0[:,2],np.asarray(location)-P)
        measured=T[3:]+np.cross(T[:3],rr)
        data.update(native_velocity=measured.tolist(),velocity_error_mps=float(np.linalg.norm(measured-v)),
                    native_radius_m=float(np.linalg.norm(rr)),native_contact_data=contact)
    return data


def native_jobs():
    p=freeze()
    c0=_read(PRACTICE)
    body,ready,_=tt90.material()
    valid=dict(protocol=_hash(PLAN),practice=_hash(PRACTICE))
    report=_read(MOUNTED) if MOUNTED.exists() else dict(**valid, rows=[], complete=False)
    if any(report.get(k)!=v for k,v in valid.items()):
        raise ValueError('Native teaching inputs changed')
    with tt90.school.enabled(cameras=False),tt90.system.installed(body['v'],body['mount']),tt90.BASE.centered(ready):
        a=tt90.BASE.arena(width=64,height=48)
        ik=inv.tt72.FullPose(a)
        try:
            for c,row in list(zip(c0['cards'],c0['rows'],strict=True))[len(report['rows']):]:
                paid=dict((t['tag'][-1]-1,t['frames']) for t in row['optical_earned'])
                awards={o['leg']:o for o in row['placements']['delivered'] if o['near_landed'] and o['near_contact']}
                rewards={}
                enc=[]; times=[]
                if awards:
                    a.launch(row['shot'],card=c); a.partner.reset(seed=c['seed'])
                    original=a._racket_contact
                    def contact(side,location,incoming):
                        if side==1:
                            b=inv.tt72.blade(a)
                            rewards[round(b['time']*480)]=(b,np.asarray(location).tolist())
                        return original(side,location,incoming)
                    with patch.object(a,'_racket_contact',contact):
                        for u in row['actions']:
                            times.append(float(a.data.time))
                            enc.append(a.joints.copy())
                            a.exchange(u,tt90.BASE.FAR_COMMAND)
                    if digest_json(a.events)!=digest_json(row['events']):
                        raise ValueError(f'Mounted camera/encoder provenance changed on {c["id"]}')
                labels=[]
                for leg,o in awards.items():
                    t=o['near_contact']['time']
                    obs=[f for f in paid.get(leg,[]) if f[0]*DT <= t+1e-9 and t-f[0]*DT<.071]
                    estimates=encoder_at(times,enc,t)
                    if not obs or estimates is None:
                        continue
                    f=obs[-1]
                    if not np.allclose(f[2],enc[f[0]],rtol=0.,atol=1e-8):
                        raise ValueError('Public measured joint packet mismatched exact physically issued tape')
                    optical_at=projected(f[1],t-f[0]*DT)
                    state,loc=rewards[round(t*480)]
                    if not np.allclose(loc,o['near_contact']['location'],rtol=0.,atol=1e-9):
                        raise ValueError('Credited scored patch does not share the physical contact')
                    data=label(ik,*estimates,loc,optical_at,state)
                    data.update(id=c['id'],leg=leg,tick=f[0],score_time=t,inbound_pv=f[1],
                                receive_height=None if not o['receive_line'] else float(tt90.optics.projected_height(o['receive_line'])),
                                legal=True,source='mounted-natural')
                    if data['patch_radius_m']<=.079 and abs(np.asarray(data['normal'])@data['patch_relative'])<.085 and np.linalg.norm(data['patch_relative'])<.14:
                        labels.append(data)
                report['rows'].append(dict(id=c['id'],scored=len(awards),labels=labels,event_sha256=digest_json(row['events'])))
                report['complete']=len(report['rows'])==N
                _write(MOUNTED,report)
                print('mounted study lessons',c['id'],len(awards),len(labels),flush=True)
        finally:
            a.close()


def stock_jobs():
    freeze()
    bank=old.OBSERVATIONS
    raw=_read(bank.with_suffix('.json'))
    receipt=_read(old.OUT/'empirical-endpoints.json')
    with np.load(bank) as f:
        state,mark=f['states'],f['marks']
    point,identity=old.groups(raw)
    src={}
    valid=[]
    t=mark[:,1]
    i0=np.flatnonzero((state[:,4]<-.9)&(t<0.)&(t>=-DT)&(abs(state[:,0])<.235)&(state[:,2]<1.10))
    matches={}
    for i in i0[i0>0]:
        if point[i]!=point[i-1] or mark[i,0]-mark[i-1,0]!=1:
            continue
        r=raw['practice'][point[i]]
        when=float(mark[i,0]*DT-t[i])
        matches.setdefault((r['source'],r['source_row'],round(when*480)),i)
    d=dict(protocol=_hash(PLAN),sources={},rows=[],complete=False)
    ready=inv.tt72.receipt()[0][0.]
    with inv.mpc.centered(ready):
        arena=inv.mpc.arena(width=64,height=48)
        ik=inv.tt72.FullPose(arena)
        try:
            for row in receipt['train']:
                path=bank.parent/row['source']
                if path not in src:
                    if raw['sources'][row['source']]!=_hash(path):
                        raise ValueError('Earned stock source changed')
                    d['sources'][str(path)]=_hash(path)
                    src[path]=_read(path)
                when=row['impact_time_s']
                key=(row['source'],row['source_row'],round(when*480))
                j=matches[key]
                physical=src[path]['rows'][row['source_row']]
                events=[x for x in physical['events'] if x['kind']=='contact' and x['side']==1 and abs(x['time']-when)<1e-7]
                if len(events)!=1 or identity[j]!=row['id']:
                    raise ValueError('Stock earned label did not match an exact contact')
                before=[i for i in range(max(0,j-3),j+1) if point[i]==point[j]]
                f=encoder_at(mark[before,0]*DT,state[before,6:11],when)
                if f is None:
                    continue
                opt=projected(state[j,:6],when-mark[j,0]*DT)
                x=label(ik,*f,events[0]['location'],opt)
                x.update(id=row['id'],leg=None,source='stock-paid',score_time=when)
                if x['patch_radius_m']<=.085 and np.linalg.norm(x['patch_relative'])<.14:
                    valid.append(x)
            d['rows']=valid
            d['complete']=True
            _write(STOCK,d)
            print('stock lessons',len(valid),flush=True)
        finally:
            arena.close()


def fitted():
    p=freeze()
    a,b=_read(STOCK),_read(MOUNTED)
    if not all(r['complete'] and r['protocol']==_hash(PLAN) for r in (a,b)):
        raise ValueError('Finish the two physical acquisitions first')
    data=a['rows']+[x for r in b['rows'] for x in r['labels']]
    keys=('optical','patch_relative','normal','signed_face','patch_coordinates','velocity','joint','id')
    np.savez_compressed(MODEL,**{k:np.asarray([x[k] for x in data]) for k in keys},
                        body=np.asarray([int(x['source']=='mounted-natural') for x in data]))
    deviations=np.asarray([x['velocity_error_mps'] for r in b['rows'] for x in r['labels']])
    directions={str(i):[] for i in (0,1)}
    for v in data:
        n=np.asarray(v['normal'])
        w=np.asarray(v['velocity'])
        directions[str(int(v['source']=='mounted-natural'))].append(float(np.linalg.norm(tangent(n,w))))
    _write(ROOT/'lesson-receipt.json',dict(protocol=_hash(PLAN),model=_hash(MODEL),entries=len(data),
             stock=_hash(STOCK),mounted=_hash(MOUNTED),mounted_games=N,counts={k:len(v) for k,v in directions.items()},
             tangent_median_mps={k:float(np.median(v)) for k,v in directions.items()},
             mounted_encoder_native_median_mps=float(np.median(deviations)),full_vector=True))


if __name__=='__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('stage',choices=('freeze','stock','mounted','fit'))
    {'freeze':freeze,'stock':stock_jobs,'mounted':native_jobs,'fit':fitted}[cli.parse_args().stage]()
