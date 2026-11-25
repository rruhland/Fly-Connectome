"""Acquire direct motor references from physically rewarded neighboring TT37 deliveries."""
import argparse
from copy import deepcopy
from pathlib import Path

import numpy as np

from . import tt56_central_training as clinic
from . import tt57_motor_school as school
from . import tt58_motor_neighborhood_plan as plan
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt39_carry_joints import Carried
from .tt44_paired_credit import State

mpc = school.mpc
course = school.course
ACTOR_CARRY = school.season.CARRY
ANCHORS = plan.ROOT / 'witnesses.json.gz'
PRACTICE = plan.ROOT / 'practice.json.gz'
HELD = plan.ROOT / 'held.json.gz'
ACQ = ('anchor', 'retarget', 'bank', 'counter')
BALL_SCALE = np.array([.053,.049,.37,.62,.78])
Q_SCALE = np.array([.35,.55,.57,.67,.66])
FINGERPRINT = ('far_contact','near_bounce')
GATE = -.485
# Source of the first 11 recorded practice rounds. The successor changes
# only the accounting of a physically interrupted, therefore unmatched offer.
STRICT_ABORT_RECEIPT = '46b223b12ab3b806d263ab631affef10844fbaf82dfb33814d0e3755344c4f97'


def fingerprint(offer):
    return digest_json(tuple(offer[k] for k in FINGERPRINT)) if offer else None


def check_offer(got, wanted):
    if fingerprint(got.get('offer')) != wanted['exact_offer']:
        raise ValueError('Motor arm changed the previously frozen physically delivered far hit or bounce')


def snapshot(a, actor, eye, tick, q, x, b):
    return dict(time=tick, first=(q,x,b), native=State(a), eye=deepcopy(eye),
                body=clinic.cams.body_snapshot(actor.engine))


def start(a, camera, rig, actor, old, card, leg, cut, change=None, exact_tick=None):
    """Existing physical prefix until a common causal late gate or the documented demo fork."""
    a.launch(old['shot'],card=card)
    a.partner.reset(seed=card['seed'])
    axes=a._arm_ids[1][2]
    actor.begin()
    eye=mpc.Vision(rig,a.control_dt,raw=True)
    eye.eye=mpc.LearnedOptics(eye.eye,course.OPTIC)
    earliest=(exact_tick-48 if exact_tick is not None else max(0,cut-36))
    for tick,u in enumerate(old['actions'][:cut+134]):
        if tick==cut:
            expected=[e for e in old['events'] if e['time']<=a.data.time+1e-8]
            if digest_json(a.events)!=digest_json(expected) or len(a.partner.requests)!=leg:
                raise ValueError('Earlier achieved physical stroke changed')
            if change is not None:
                a.card=plan.card_for(card,leg,change)
        q=a.joints.copy()
        x=mpc.observed(eye,tick,camera.capture() if tick%2==0 and tick>=earliest else None,q)
        b=bool(x is not None and x[4]<-.9 and x[1]<-.09 and (eye.eye.latest.bounced or x[5]>.58))
        if exact_tick is not None and tick==exact_tick:
            return snapshot(a,actor,eye,tick,q,x,b)
        if (exact_tick is None and tick>=cut and x is not None and x[4]<-.95
                and -.82<x[1]<GATE and x[2]>.774):
            return snapshot(a,actor,eye,tick,q,x,b)
        actor.engine.observe(tick,eye.eye,q,a.data.ctrl[axes].copy())
        _,_,done,cap,_=a.exchange(u,mpc.FAR_COMMAND)
        if done or cap:
            break
    return None


def point(track, depth):
    if not len(track) or track[-1,2] >depth+.009 or track[0,2]<depth-.028:
        return None
    return track[np.argmin(np.abs(track[:,2]-depth))]


def flight(path, x, bounced, spin):
    return path.path(np.asarray(x[:3]),np.asarray(x[3:]),spin,1,bounced=bounced,horizon=.61)


def predict(x, bounced, flight_model):
    # Use the same coarse pre-impact RGB projection when writing and reading.
    forecast=flight(flight_model,x,bounced,np.zeros(3))
    if not len(forecast) and x[2]>.780 and x[1]<-.49:
        forecast=flight(flight_model,x,not bounced,np.zeros(3))
    return forecast


class MotorBank:
    def __init__(self, entries):
        self.entries=entries

    def candidates(self, x, q, ref, projected, mode):
        if projected is None:
            return []
        out=[]
        for r in self.entries:
            # The onset band is indexed by observed ball y, not archived ticks or release.
            gx=np.asarray(r['p'])
            if not -.158 < x[1]-gx[1] < .27:
                continue
            dist=np.linalg.norm((projected[1:7][[0,2,3,4,5]]-r['projected'])/BALL_SCALE)
            dist+=.52*np.linalg.norm((q[:5]-np.asarray(r['q'])[:5])/Q_SCALE)
            dist+=.24*np.linalg.norm(((ref-q)[:5]-(np.asarray(r['ref'])-r['q'])[:5])/.16)
            if dist>3.1 or np.max(np.abs(q[:5]-np.asarray(r['q'])[:5]))>.64:
                continue
            # During the handoff prefer the nearest recorded earliest phase.
            cost=dist+2.2*abs(x[1]-gx[1])
            out.append((cost,r,dist))
        out.sort(key=lambda i:i[0])
        if mode=='counter' and len(out)>1 and out[1][2]<out[0][2]+.46:
            out=[out[1],out[0],*out[2:]]
        return out if out and x[1]-out[0][1]['p'][1] < .027 else []


def synthesize(row, q, ref, live, old, geom, mode):
    """Resample references from common postbounce depth, then adapt their position."""
    target, src=np.asarray(live),np.asarray(old)
    delta=(src[0]-target[0])/mpc.DT
    if not -8.5<delta<8.5:
        return None
    ratio=np.clip(abs(target[5]/src[5]),.77,1.29)
    ix=np.arange(1,school.H+1)
    at=src[0]/mpc.DT+(ix-target[0]/mpc.DT)*ratio if mode!='anchor' else ix+delta
    origin=np.asarray(row['ref'])
    each=np.vstack((origin,np.asarray(row['refs'])))
    goals=np.column_stack([np.interp(at,np.arange(len(each)),e) for e in each.T])
    if np.max(abs(goals[0,:5]-ref[:5]))>.63:
        return None
    if mode=='anchor':
        return goals
    # A small, decaying alignment term retains reference/encoder continuity.
    dq0=np.clip((ref-origin)*.54+(q-np.asarray(row['q']))*.21,-.17,.17)
    goals+=np.maximum(0,1-ix[:,None]/11)*dq0
    _,_,J,N=geom.pose(np.asarray(row['impact_q'])[:5],jacobian=True)
    A=np.vstack((J,.11*N))
    err=target[1:4]-src[1:4]
    # Draws can shift speed and position separately. Keep most normal while
    # reproducing the earned forward spacing and lateral/vertical impact patch.
    err=np.clip(err,-.055,.055); err[1]*=.20
    ratio_gain={'retarget':.95,'bank':.73,'counter':1.24}[mode]
    dj=A.T@np.linalg.solve(A@A.T+.043**2*np.eye(6),np.r_[err,np.zeros(3)])
    dj=np.clip(ratio_gain*dj,-.19,.19)
    ramp=np.clip(ix/(.72*target[0]/mpc.DT),0,1)
    goals[:,:5]+=ramp[:,None]*dj
    return goals


class Receiver(clinic.Viewer):
    def __init__(self,raw):
        super().__init__(mpc.BANK,raw,variant='elastic',budget=28,
                         coverage=clinic.fresh.shot.Coverage(_read(clinic.fresh.RISK)))
        self.bank=None;self.mode='stock';self.shield=False

    def begin(self):
        super().begin()
        self.commit=None;self.tick0=None;self.selected=None
        self.planner_steps=[]

    def step(self,tick,eye,x,q,reference,base):
        if self.commit is not None:
            i=tick-self.tick0
            return self.commit[min(i,len(self.commit)-1)]
        u=super().step(tick,eye,x,q,reference,base)
        if self.bank is None or x is None or x[4]>=-.9 or not -1.03<x[1]<-.50 or x[2]<.77:
            return u
        if self.shield and self.best is not None and self.notes and 0<=tick-self.tick<self.lease and self.notes[-1].get('robust',0)>0:
            return u
        path=self.engine.twin.partner.flight
        forecast=predict(x,self.engine.bounced,path)
        ranked=self.bank.candidates(x,q,reference,point(forecast,-1.43),self.mode)
        for _,entry,d in ranked:
            if x[1]-entry['p'][1] >=.027:
                continue
            at=point(forecast,entry['depth'])
            if at is None or at[0]<.09:
                continue
            goals=synthesize(entry,q,reference,at,entry['impact_prediction'],self.geometry,self.mode)
            if goals is None:
                continue
            self.commit=goals; self.tick0=tick
            self.selected=dict(owner=entry['origin'],distance=float(d),tick=tick,mode=self.mode,
                               x=x.tolist(),predicted_impact_y=float(at[2]),
                               predicted_eta=float(at[0]),q=q.tolist(),ref=reference.tolist())
            return goals[0]
        return u


def demonstration(actor,witness,start_tick,contact,origin):
    seg=[r for r in witness['log'] if start_tick<=r['tick']<start_tick+school.H]
    if len(seg)!=school.H or seg[0]['x'] is None or contact is None:
        return None
    it=contact['time']/mpc.DT
    seen=[r for r in witness['log'] if it-4.0<=r['tick']<it and r['x'] is not None and r['x'][4]<-.95]
    if not seen:
        return None
    final=seen[-1]
    x=np.asarray(final['x']); dt=(it-final['tick'])*mpc.DT
    # Exact success supplies a reward/credit time, but phase position and
    # joint contact targets are from the pre-hit stereo and encoder samples.
    depth=float(np.clip(x[1]+x[4]*dt,-1.68,-1.24))
    first=seg[0]; p=np.asarray(first['x'])
    model=actor.engine.twin.partner.flight
    forecast=predict(p,first['bounce'],model)
    view=point(forecast,-1.43); modeled=point(forecast,depth)
    if view is None or modeled is None:
        return None
    return dict(origin=origin,p=p.tolist(),q=first['q'],ref=first['ref'],
                refs=[r['next_ref'] for r in seg],depth=depth,projected=view[1:7][[0,2,3,4,5]].tolist(),
                impact_prediction=modeled.tolist(),impact_q=final['q'],last_pre_hit_dt=dt,
                observed_delay=(it-start_tick)*mpc.DT)


def demo_map():
    data=_read(school.OUT)
    rows={}
    for r in data['rows']:
        w=next((a for a in r.get('arms',[]) if a.get('issued_relative') and a['answered']),None)
        if w:
            rows.setdefault(r['case']['id'],w)
    return rows


def trial(a,res,actor,g,draw,card,case,home,mode='stock',bank=None,*,eval_only=False):
    actor.bank=None if bank is None else MotorBank(bank)
    actor.mode=mode;actor.shield=eval_only
    z=clinic.play(a,actor,res,g,dict(draw,target=draw['goal']),mode,card,case,home)
    z['selected']=actor.selected
    if z['landed'] and z['answered'] and not z['offer']['near_contact']:
        raise ValueError('Return was not earned from a physical arm contact')
    if fingerprint(z['offer'])!=draw['exact_offer']:
        if mode=='stock' or z['selected'] is None:
            check_offer(z,draw)
        z.update(unmatched=True, actual_landed=z['landed'], actual_answered=z['answered'],
                 landed=False,answered=False)
    return z


def run(stage):
    protocol=_read(plan.PLAN);scout=_read(plan.OUT)
    if (not scout['complete'] or scout['protocol_sha256']!=_hash(plan.PLAN)
            or protocol['motor_school_sha256']!=_hash(school.OUT)):
        raise ValueError('Frozen physical scout or witness changed')
    base=dict(protocol_sha256=_hash(plan.PLAN),scout_sha256=_hash(plan.OUT),
              actor_source_sha256=_hash(Path(__file__)),carry_sha256=_hash(ACTOR_CARRY),
              optics_sha256=_hash(course.OPTIC),mode=stage,rows=[],entries=[],complete=False)
    out={'witness':ANCHORS,'practice':PRACTICE,'held':HELD}[stage]
    result=_read(out) if out.exists() else base
    if result['complete']:
        return
    if (stage=='practice' and result.get('actor_source_sha256')==STRICT_ABORT_RECEIPT
            and len(result['rows'])==11):
        result.update(actor_source_sha256=base['actor_source_sha256'],
                      unchanged_first_eleven_sha256=_hash(out),
                      previous_actor_source_sha256=STRICT_ABORT_RECEIPT)
    if any(result.get(k)!=v for k,v in base.items() if k not in ('rows','entries','complete')):
        raise ValueError('Incomplete attempt used a different frozen plan, actor, or bank')
    anchors=[] if stage=='witness' else _read(ANCHORS)['entries']
    if stage!='witness' and len(anchors)!=len(protocol['source']):
        raise ValueError('Could not reconstruct all required camera-body anchor testimonies')
    if stage=='held':
        training=_read(PRACTICE)
        if not training['complete']:
            raise ValueError('Practice must finish before opening held receiver arms')
        earned=training['entries']
        have={r['origin']['id'] for r in earned}
        if len(have)<2:
            result.update(complete=True,status='Fewer than two distinct physically rewarded practice neighborhoods. Held receivers unopened.',
                          trained_sha256=_hash(PRACTICE));_write(out,result);return
        result['trained_sha256']=_hash(PRACTICE)
    else:
        earned=result['entries']
    demos=demo_map()
    with np.load(mpc.BANK) as f:
        raw=mpc.DeepField(f,'local')
    for item,chosen in zip(protocol['source'],scout['rows'],strict=True):
        case=item['case']; seed_id=item['id']
        if chosen['id']!=seed_id:
            raise ValueError('Frozen scouts no longer match source')
        todo=[None] if stage=='witness' else chosen['assigned'][stage]
        todo=[g for g in todo if not any(r['id']==seed_id and r['seed']==(g['seed'] if g else None) for r in result['rows'])]
        if not todo:
            continue
        src=Path(case['source']); doc=_read(src); old=doc['rows'][case['i']]
        card=case['card']; home=np.asarray(case['ready']); leg=case['leg']; cut=item['anchor']
        if _hash(src)!=item['source_sha256'] or card!=doc['cards'][case['i']]:
            raise ValueError('Archived original attained body became different')
        with mpc.centered(home),school.enabled():
            a=mpc.arena(); cad=mpc.arena(detail='cad',width=64,height=48)
            clear=clinic.CourtClearance(cad.model,articulated=True)
            res=mpc.resources(a); camera,rig,*_=res
            from unittest.mock import patch
            with patch.object(mpc,'Fragments',lambda file,engine:Carried(file,engine,practice=[course.PRACTICE],held=set(plan.ANCHORS))):
                actor=Receiver(raw)
            course.resume(actor,ACTOR_CARRY.with_suffix('').with_suffix('.npz'),_read(ACTOR_CARRY))
            try:
                for draw in todo:
                    g=start(a,camera,rig,actor,old,card,leg,cut,None if draw is None else draw['seed'],
                            item['view_phase'] if stage=='witness' else None)
                    row=dict(id=seed_id,seed=None if draw is None else draw['seed'],scout=draw,arms={})
                    if g is None:
                        row['status']='No causal RGB handoff';result['rows'].append(row);_write(out,result);continue
                    row['gate']=g['time'];row['first_x']=None if g['first'][1] is None else g['first'][1].tolist()
                    if stage=='witness':
                        w=demos[seed_id]
                        if digest_json(w['controls'])!=item['correct_tape_sha256']:
                            raise ValueError('Documented actual school motion changed')
                        z=clinic.play(a,actor,res,g,dict(target=item['base_target']),stage,card,case,home,commands=w['controls'])
                        obs=demonstration(actor,z,g['time'],z['offer']['near_contact'],dict(id=seed_id,seed=None,stage=stage))
                        if not z['answered'] or not z['landed'] or not clinic.same(z['offer'],w['delivered']) or obs is None:
                            raise ValueError(f'Actual RGB/encoder anchor capture failed {seed_id}: {z["landed"]} {z["answered"]} template {obs is not None}')
                        result['entries'].append(obs);row['arms']['witness']=z
                    else:
                        z=trial(a,res,actor,g,draw,card,case,home);row['arms']['stock']=z
                        run_modes=ACQ if stage=='practice' else ('fixed','bank')
                        if stage=='practice' and z['landed']:
                            run_modes=()
                        for name in run_modes:
                            bank=anchors if name in ('anchor','retarget','fixed') else [*anchors,*earned]
                            mode='bank' if name=='fixed' else name
                            got=trial(a,res,actor,g,draw,card,case,home,mode,bank,eval_only=stage=='held')
                            row['arms'][name]=got
                            if not got['landed'] or not got['answered'] or got['selected'] is None:
                                continue
                            cad_result=clinic.proof(a,home,card,draw['seed'],case,cut,old,g['time'],got,clear)
                            got['strict_cad']=cad_result
                            if cad_result['penetration_samples']:
                                continue
                            if stage=='practice':
                                now=got['selected']['tick']
                                obs=demonstration(actor,got,now,got['offer']['near_contact'],dict(id=seed_id,seed=draw['seed'],stage=stage))
                                if obs is not None:
                                    result['entries'].append(obs);row['acquired']=name
                                    break
                    result['rows'].append(row)
                    _write(out,result)
                    print(stage,seed_id,row['seed'],row['gate'],[(k,bool(v['landed']),bool(v['answered']),
                          v.get('selected',{}).get('owner') if v.get('selected') else None)
                          for k,v in row['arms'].items()],row.get('acquired'),flush=True)
            finally:
                actor.close()
                camera.close()
                cad.close()
                a.close()
    result['complete']=True
    _write(out,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('witness','practice','held'))
    run(p.parse_args().mode)
