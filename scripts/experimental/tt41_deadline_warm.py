"""One neutral outbound preparation, then matched late coach or archived continuation."""

import argparse
from bisect import bisect_right
from pathlib import Path

import numpy as np

from . import tt41_deadline as deadline
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, resources
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded, verify_clearance
from .tt39_opportunity_map import BANDS, EDGES, projected_height
from .tt40_long_low import indexed, physical_finish, same_trace
from .tt40_low_feasibility import roster

DEST=deadline.DEST
MIX=.42
WARM_AXES=(1,2,4)
MAX_PREP_TICKS=60


class Outbound:
    """Previous two stereo displacements must both carry away from the near body."""
    def __init__(self):
        self.points=[]

    def update(self,tick,p,dt):
        if p is None:
            self.points=[]
            return False
        self.points.append((tick,float(p[1])))
        self.points=self.points[-3:]
        if len(self.points)<3:
            return False
        ts,y=np.array(self.points).T
        return bool(np.all(np.diff(ts)==2) and y[-1]>-.65 and np.all(np.diff(y)>.025)
                    and (y[-1]-y[0])/((ts[-1]-ts[0])*dt)>.9)


def warm_ref(ready,proofs):
    signed=[r['wins'][0]['chosen']['variant']['approach'] for name in ['long-low-flipped','long-low-alternate-strict']
            for r in proofs[name]['rows'] if r['wins']]
    # Sign is a common motor mode (not a direction or a low-arrival prediction).
    # Preserve archive pan, tilt and grip; moderate the shared shoulders/wrist.
    common=np.median(np.asarray(signed),axis=0)
    q=np.asarray(ready).copy()
    cols=list(WARM_AXES)
    q[cols]=(1-MIX)*q[cols]+MIX*common[cols]
    return q


def reference_tape(a,old,c):
    axes=a._arm_ids[1][2]
    a.sample_clearance(False)
    deadline.replay_prefix(a,old,c,0)
    ref=[]
    dt=a.control_dt
    stride=a._joint_increments[axes]*(dt/a.model.opt.timestep)
    ref.append(a.data.ctrl[axes].copy())
    for u in old['actions']:
        a.exchange(u,FAR_COMMAND)
        ref.append(a.data.ctrl[axes].copy())
    if digest_json(a.events)!=digest_json(old['events']):
        raise ValueError('Archive reference tape did not replay exactly')
    return np.array(ref),stride,a._joint_low[axes],a._joint_high[axes]


def controlled(a,clear,view,rig,old,c,start,search_end,refs,stride,lo,hi,ready,warm,*,ordinary=False):
    """Original commands until outbound RGB; causal RGB alone controls the warm interval."""
    if ordinary:
        a.sample_clearance(False)
        deadline.replay_prefix(a,old,c,start)
        guarded(a,clear)
    else:
        deadline.guarded_prefix(a,clear,old,c,start)
    audit_start=float(a.data.time) if ordinary else 0.
    dt=a.control_dt
    ax=a._arm_ids[1][2]
    commands=list(old['actions'][:start])
    out=Outbound()
    turn=deadline.TurnGate()
    active=None
    inbound=None
    why=None
    frames=[]
    for k in range(start,search_end):
        if k%2==0 and inbound is None:
            point=deadline.color_pos(view,rig)
            frames.append([k,None if point is None else point.tolist()])
            if out.update(k,point,dt) and active is None:
                active=k
            if active is not None and turn.update(k,point,dt):
                inbound=k
                why='rgb_turn'
        if active is not None and k-active>=MAX_PREP_TICKS and inbound is None:
            inbound=k
            why='warm_timeout'
        if inbound is not None and not ordinary:
            break
        if warm and active is not None:
            goal=refs[k+1].copy()
            if inbound is None:
                cols=list(WARM_AXES)
                w=float(deadline.ease((k+1-active)/18.))
                goal[cols]=(1.-w)*goal[cols]+w*ready[cols]
            q=a.data.ctrl[ax].copy()
            u=(deadline.bound(goal[None,None,:],q,np.asarray(commands[-1]),stride,lo,hi)[0,0]-q)/stride
            u=np.clip(u,-deadline.RATE,deadline.RATE)
            u=u.tolist()
        else:
            u=old['actions'][k]
        commands.append(u)
        _,_,end,cap,_=a.exchange(u,FAR_COMMAND)
        if end or cap:
            break
    ruling=a.clearance_result()
    if ruling['samples']!=round((a.data.time-audit_start)/a.model.opt.timestep):
        raise ValueError('Not all relevant solver steps were sampled')
    return dict(outbound_tick=active,rgb_tick=inbound,rgb_reason=why,frames=frames,actions=commands,
                clearance=ruling,audit_start_s=audit_start,end_s=float(a.data.time))


def ordinary_rows(source):
    pools={name:{} for name in BANDS[1:]}
    for idx,(c,r) in enumerate(zip(source['cards'],source['rows'],strict=True)):
        if c['id'] in deadline.ORDER:
            continue
        for k,v in enumerate(r['placements']['delivered'][1:],1):
            if not v.get('receive_line'):
                continue
            h=projected_height(v['receive_line'])
            name=BANDS[bisect_right(EDGES,h)]
            if name in pools:
                pools[name].setdefault(c['id'],(idx,k,h))
    # Draw at most three per declared height band, one target per release across all
    # bands. Scored return and CAD outcomes were not queried by case assignment.
    rng=np.random.default_rng(411003)
    result=[]
    used=set()
    for band,pool in pools.items():
        ids=sorted(pool)
        added=0
        for i in rng.permutation(len(ids)):
            id=ids[i]
            if id not in used:
                used.add(id)
                result.append((band,*pool[id]))
                added+=1
                if added==3:
                    break
    return result


def received(a,k):
    placements=a.point_placements()['delivered']
    if len(placements)<=k:
        return dict(offered=False,legal=False,answered=False,contact=False)
    v=placements[k]
    return dict(offered=bool(v['near_bounce']),legal=bool(v['near_landed']),
                contact=bool(v['near_contact']),
                answered=bool(v['near_landed'] and len(placements)>k+1 and placements[k+1]['near_bounce']),
                far=v['far_contact'],bounce=v['near_bounce'])


def run(output,stock,batch,rounds,checks,paired_only=False,low_only=False):
    s=_read(deadline.SOURCE)
    base=_read(stock) if not paired_only else None
    if base and (not base.get('complete') or base['source_sha256']!=_hash(deadline.SOURCE)):
        raise ValueError('Stock deadline run is missing or from a different source')
    proofs={p:_read(deadline.OUT/(p+'.json.gz')) for p in deadline.PROOFS}
    target=warm_ref(s['ready'],proofs)
    receipt=dict(schema='tt41-deadline-warm-v1',role='one RGB-triggered neutral stance; exact-state low suffix coach; fixed archived ordinary continuation',
                 source_sha256=_hash(deadline.SOURCE),stock_sha256=None if paired_only else _hash(stock),
                 ready=s['ready'],warm_axes=WARM_AXES,warm_mix=MIX,warm_q=target.tolist(),
                 rate_bound=deadline.RATE,rate_change_per_tick=deadline.SLEW,rows=[],ordinary=[])
    named={r['id']:r for r in roster(s)}
    _write(output,receipt)
    with centered(s['ready']):
        a=deadline.arena(protocol='random')
        cad=deadline.arena(protocol='random',detail='cad')
        view,rig,*_=resources(a)
        clear=CourtClearance(cad.model,articulated=True)
        engine=deadline.Rollout(a,1)
        dt=a.control_dt
        try:
            for band,idx,k,h in ordinary_rows(s) if not low_only else ():
                old=s['rows'][idx]; c=s['cards'][idx]; v=old['placements']['delivered'][k]
                start,near,_=indexed(old['events'],v,dt)
                specs=reference_tape(a,old,c)
                results={}
                for warm in (False,True):
                    # Observe one target opportunity and enough ensuing physical play
                    # for an answer, continuing the reactive opponent in both arms.
                    end=min(len(old['actions']),near+108)
                    pre=controlled(a,clear,view,rig,old,c,start,end,
                                   *specs,target,warm,ordinary=True)
                    pre.update(received(a,k),commands_sha256=digest_json(pre['actions']),
                               final_events=a.events.copy())
                    results['warm' if warm else 'stock']=pre
                pure=results['stock']
                observed=[e for e in old['events'] if e['time']<=pure['end_s']+1e-8]
                if digest_json(pure['final_events'])!=digest_json(observed):
                    raise ValueError('Ordinary stock control did not replay exactly')
                w=results['warm']
                fixed=same_trace(pure.get('far'),w.get('far')) and same_trace(pure.get('bounce'),w.get('bounce'))
                receipt['ordinary'].append(dict(id=c['id'],leg=k,band=band,projected_height_m=h,
                                                incoming_unchanged=fixed,**results))
                print('ordinary',c['id'],k,band,'incoming same',fixed,'stock',pure['legal'],pure['answered'],
                      'warm',w['legal'],w['answered'],'rgb',w['outbound_tick'],w['rgb_tick'],flush=True)
                _write(output,receipt)
            for b in base.get('rows',[]) if base else []:
                if any(t['name']=='rgb' and t['wins'] for t in b['stages']):
                    continue
                record=named[b['id']]
                idx,k=record['index'],b['leg']
                c=s['cards'][idx]; old=s['rows'][idx]
                v=old['placements']['delivered'][k]
                start,near,_=indexed(old['events'],v,dt)
                priors,recipes=deadline.clean_proofs(a,k,old,c,proofs)
                specs=reference_tape(a,old,c)
                prep=controlled(a,clear,view,rig,old,c,start,near,*specs,target,True)
                tick=prep['rgb_tick']
                pre=prep['actions']
                note=dict(id=c['id'],leg=k,index=idx,card=c,fork_tick=tick,start_tick=start,warm=prep,
                          prefix_sha256=digest_json(a.events),wins=[],physical_attempts=[])
                receipt['rows'].append(note)
                if tick is None or prep['rgb_reason']!='rgb_turn' or prep['clearance']['penetration_samples']:
                    print('warm rejected before search',c['id'],tick,prep['rgb_reason'],prep['clearance'],flush=True)
                    _write(output,receipt)
                    continue
                original=[e for e in old['events'] if e['time']<=a.data.time+1e-8]
                if not same_trace(a.events,original):
                    print('warm changed flight or referee before incoming gate',c['id'],flush=True)
                    note['changed_prefix_events']=True
                    _write(output,receipt)
                    continue
                found,trace,geo,cpu=deadline.search(a,engine,priors,recipes,tick,np.asarray(pre[-1]),
                                                    batch,rounds,[411003,c['id'],tick])
                note.update(shadow=trace,geometry=geo,cpu_s=cpu)
                print('warm prepared RGB coach',c['id'],tick,trace,'private unique',len(found),flush=True)
                mod=dict(old,actions=pre)
                for i,choice in enumerate(found[:checks]):
                    contacts=[]
                    with contact_observer(a,contacts):
                        deadline.guarded_prefix(a,clear,mod,c,tick)
                        r=physical_finish(a,old,c,k,pre,choice)
                        verify_clearance(a)
                    rul=a.clearance_result()
                    ok=r['answered'] and rul['penetration_samples']==0 and received(a,k)['answered']
                    note['physical_attempts'].append(dict(i=i,legal=r['legal'],answered=r['answered'],
                                                          clean_answer=ok,clearance=rul))
                    print('warm physical',c['id'],r['legal'],r['answered'],ok,rul,flush=True)
                    if ok:
                        note['wins'].append(dict(r,clearance=rul,contacts=contacts,fork_tick=tick))
                        break
                _write(output,receipt)
        finally:
            view.close()
            a.close()
            cad.close()
    receipt['complete']=True
    _write(output,receipt)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=DEST/'deadline-warm.json.gz')
    p.add_argument('--stock',type=Path,default=DEST/'deadline-stock.json.gz')
    p.add_argument('--paired-only',action='store_true')
    p.add_argument('--low-only',action='store_true')
    p.add_argument('--batch',type=int,default=192)
    p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--checks',type=int,default=5)
    o=p.parse_args()
    run(o.output,o.stock,o.batch,o.rounds,o.checks,o.paired_only,o.low_only)
