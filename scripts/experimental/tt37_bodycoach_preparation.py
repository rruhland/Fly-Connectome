"""Test CAD-checked connected recovery poses found from an independent witness."""

import argparse
from pathlib import Path
from time import process_time

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt29_rally_physics import measured_launches
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND
from .tt37_bodycoach import SpatialTeacher
from .tt37_bodycoach_windows import MOTOR, Reach
from .tt37_opponent import arena
from .tt37_opponent_benchmark import BASE, OUT, PLAN, mark


WITNESS=OUT/'bodycoach-windows.json.gz'
HUBS=OUT/'bodycoach-hubs.json'


def derive(path, output):
    data=_read(path)
    ready=np.asarray(data['ready_raw'])
    alt=np.asarray(data['ready_center'])
    paths=[]
    pool=[ready[:5],alt[:5]]
    for r in data['rows']:
        rr=[k for k in r['records'] if k['legal_proxy']]
        if rr:
            q=np.array([k['joint'] for k in rr])
            deadline=np.array([k['time_from_far_s'] for k in rr])-.145
            weight=1.65 if abs(r['x_m'])<.075 else 1.
            paths.append((q,deadline,weight))
            pool.extend(k['joint'] for k in rr if k['ball_xyz'][2]>.96)
    # Add intermediate candidates rather than assuming the endpoint of a known strike
    # is itself the right place to prepare. All accepted poses are CAD checked below.
    rng=np.random.default_rng(379801)
    vertices=np.asarray(pool)
    choices=vertices[rng.integers(len(vertices),size=(180,2))]
    pool=np.r_[vertices,np.mean(choices,axis=1)]
    unique=[]
    for q in pool:
        if all(np.max(abs(q-v))>.18 for v in unique):
            unique.append(q)
    pool=np.array(unique)
    optimistic=np.zeros(len(pool))
    for q,dead,weight in paths:
        arrival=np.min(np.max(abs(pool[:,None,:]-q[None,:,:]),axis=2)/MOTOR-dead,axis=1)
        optimistic+=weight*(arrival<=0.)
    ranks=np.argsort(-optimistic)
    with centered(ready):
        geometry=arena(detail='cad')
    try:
        rig=Reach(geometry,ready,alt)
        states=[]
        for i in ranks:
            hub=pool[i]
            arrive=rig.ik.pose(hub)[0]
            if not (-1.65<arrive[1]<-1.12 and .87<arrive[2]<1.27) or not rig.clear(hub):
                continue
            if any(np.max(abs(np.subtract(hub,h['joint'])))<.27 for h in states):
                continue
            score=0.
            coverage=0
            for q,dead,weight in paths:
                dt=np.max(abs(q-hub),axis=1)/MOTOR
                available=np.flatnonzero(dt<=dead)
                if any(rig.route(hub,q[i],steps=8) for i in available[np.argsort(dt[available])[:3]]):
                    score+=weight
                    coverage+=1
            states.append(dict(joint=hub.tolist(),blade_center_m=arrive.tolist(),measured_arc_coverage=coverage,
                               weighted_coverage=float(score),optimistic=float(optimistic[i])))
            if len(states)>=20:
                break
        _write(output,dict(states=sorted(states,key=lambda s:-s['weighted_coverage']),ready=ready.tolist(),
                           source_sha256=_hash(path),selectable_legal_arcs=len(paths),
                           criterion='Optimize sampled collision-clear transitions and arrival budgets on synthetic witness only.'))
    finally:
        geometry.close()


class Preparation(SpatialTeacher):
    def __init__(self,a,ready,rig,hubs):
        super().__init__(a,ready,expand=True,refine=True)
        self.rig=rig
        self.hubs=hubs
        self.recoveries=[]
        self.last=0
        self.hub=None

    def command(self):
        a=self.a
        if a._last_hitter!=1 or not a.contacts or (self.tape is not None and a.steps-self.start<len(self.tape)):
            return super().command()
        if a.contacts!=self.last:
            self.last=a.contacts
            q=a.joints[:5].copy()
            ctrl=a.data.ctrl[self.axes][:5].copy()
            options=[]
            for entry in self.hubs:
                hub=np.array(entry['joint'])
                dt=float(np.max(abs(hub-q))/(MOTOR*.86))
                # Small interval, bounded changes and two checked views of the same pose.
                if dt>.43 or not self.rig.route(q,hub) or not self.rig.route(ctrl,hub):
                    continue
                score=entry['weighted_coverage']-14.*dt
                options.append((score,hub,dt,entry))
            if options:
                rank,hub,dt,entry=max(options,key=lambda p:p[0])
                self.hub=hub
                self.recoveries.append(dict(tick=a.steps,start=q.tolist(),expected_s=dt,**entry))
            else:
                self.hub=None
                self.recoveries.append(dict(tick=a.steps,start=q.tolist(),joint=None))
        if self.hub is None:
            return np.zeros(a.action_size)
        stride=a._joint_increments[self.axes[:5]]*(a.control_dt/a.model.opt.timestep)
        command=np.clip((self.hub-a.data.ctrl[self.axes[:5]])/stride,-.62,.62)
        return np.r_[command,0.]


def run(feeds,hubs_path,output,condition):
    manifest=_read(PLAN)
    cards=[c|dict(reply_seed=377931+i*103,key=f'prep-held-{i}-{c["id"]}')
           for i,c in enumerate(manifest['held_replies'][::4])]
    shots=measured_launches(feeds)
    ready=_read(BASE)['ready']
    hubs=_read(hubs_path)['states']
    started=process_time()
    cap=12
    result=dict(cards=cards,rows=[],variant=dict(grasp='straight',group='prep-independent', cap=cap),
                ready=ready,condition=condition,opening_side=1,oracle=True,
                seeds=[c['reply_seed'] for c in cards], preparation_sha256=_hash(hubs_path))
    cpu_before=0.
    if output.exists():
        previous=_read(output)
        if any(previous.get(key)!=result[key] for key in ('cards','condition','ready','preparation_sha256')):
            raise ValueError('Existing preparation run does not match the frozen protocol')
        if 'cpu_s' in previous and len(previous['rows'])==len(cards):
            return
        result['rows']=previous['rows']
        cpu_before=previous.get('partial_cpu_s',0.)
    with centered(ready):
        a=arena(robust_contact=True,clearance_margin=.003)
        geometry=arena(detail='cad',robust_contact=True,clearance_margin=.003)
    try:
        rig=Reach(geometry,ready,ready)
        a.launches=len(result['rows'])
        gap_in_cpu_record=bool(result['rows'] and not cpu_before)
        for c in cards[len(result['rows']):]:
            shot=shots[c['id']]
            with centered(ready):
                a.launch(shot,card=c)
            a.partner.reset(seed=c['seed'])
            actor=Preparation(a,ready,rig,hubs) if condition=='prepared' else SpatialTeacher(a,ready,expand=True,refine=True)
            actions=[]
            while True:
                step=actor.command()
                _,_,done,truncated,_=a.exchange(step,FAR_COMMAND)
                actions.append(step.tolist())
                if done or truncated or a.legal_returns+a.rival_returns-1>=cap:
                    break
            row=dict(shot=shot,opponent_seed=c['seed'],actions=actions,events=a.events,placements=a.point_placements(),
                     score=audit(a.events,opening_side=1),coach=actor.log,recoveries=getattr(actor,'recoveries',[]))
            mark(a,row,cap)
            result['rows'].append(row)
            print(condition,c['id'],row['score']['learner'],[x['observed_lane'] for x in row['placements']['delivered']],
                  [p['joint'] is not None for p in row['recoveries']],flush=True)
            _write(output,dict(result,partial_cpu_s=cpu_before+process_time()-started))
    finally:
        a.close()
        geometry.close()
    _write(output,dict(result,cpu_s=cpu_before+process_time()-started,
                       cpu_scope='resumed points only' if gap_in_cpu_record else 'all points'))


def summarize(output):
    refs={k:OUT/f'bodycoach-preparation-{k}.json.gz' for k in ('baseline','prepared')}
    a,b=(_read(refs[k]) for k in ('baseline','prepared'))
    if 'cpu_s' not in a or 'cpu_s' not in b or a['cards']!=b['cards']:
        raise ValueError('Preparation assays are incomplete or different')
    records=[]
    for c,r,s in zip(a['cards'],a['rows'],b['rows'],strict=True):
        rx=r['placements']['delivered']
        sx=s['placements']['delivered']
        keys=('far_contact','near_bounce','receive_line')
        first=bool(rx and sx and all(digest_json(rx[0][k])==digest_json(sx[0][k]) for k in keys))
        later=0
        for i,j in zip(rx[1:],sx[1:]):
            later+=all(digest_json(i[k])==digest_json(j[k]) for k in keys)
        asked_a=r['placements']['requested']
        asked_b=s['placements']['requested']
        matched=sum(x['target']==y['target'] for x,y in zip(asked_a,asked_b))
        first_action_change=next((i for i,(x,y) in enumerate(zip(r['actions'],s['actions'])) if x!=y),None)
        hit=next((e['time'] for e in s['events'] if e['kind']=='contact' and e['side']==1),None)
        if first_action_change is not None and (hit is None or first_action_change/60<hit):
            raise ValueError('The preparation changed a command before the first strike')
        records.append(dict(card=c,first_offer_exact=first,later_identical_offers=later,common_aims=matched,
            baseline_legal=r['score']['learner'],prepared_legal=s['score']['learner'],
            baseline_later=sum(x['near_landed'] for x in rx[1:]),prepared_later=sum(x['near_landed'] for x in sx[1:]),
            baseline_later_touches=sum(x['near_contact'] is not None for x in rx[1:]),
            prepared_later_touches=sum(x['near_contact'] is not None for x in sx[1:]),
            selector=s['recoveries'],first_changed_action=first_action_change))
    _write(output,dict(rows=records,source_sha256={k:_hash(v) for k,v in refs.items()}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=('derive','prepared','baseline','summary'))
    p.add_argument('--feeds',type=Path)
    p.add_argument('--witness',type=Path,default=WITNESS)
    p.add_argument('--hubs',type=Path,default=HUBS)
    p.add_argument('--output',type=Path,default=None)
    a=p.parse_args()
    if a.phase=='derive':
        derive(a.witness,a.output or a.hubs)
    elif a.phase=='summary':
        summarize(a.output or OUT/'bodycoach-preparation-summary.json')
    elif a.feeds is None:
        p.error('--feeds is required for play')
    else:
        run(a.feeds,a.hubs,a.output or OUT/f'bodycoach-preparation-{a.phase}.json.gz',a.phase)
