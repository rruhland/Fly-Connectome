"""Mine actual scored motions on the translated stock body, independent of held games."""

import argparse
from pathlib import Path
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt63_lateral_clamp as fixture
from . import tt63_fixture_qualification as physical
from . import tt68_clamp_transfer as registered
from . import tt55_midline_margin as contact
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent import draw
from .tt37_opponent_audit import guarded, verify_clearance
from .tt40_long_low import same_trace
from .tt44_paired_credit import State
from .tt54_training_speedups import enabled

ROOT=Path('artifacts/tt69_clamp_school')
PLAN=ROOT/'protocol.json'
OUT=ROOT/'practice.json.gz'
DX=registered.DX
mpc=registered.mpc
course=registered.course
FEEDS=Path('/tmp/tt25-rallies.json')
ZERO=[0.]*6


def plan():
    d=_read(fixture.PLAN); bench=_read(registered.PLAN); source=_read(course.SOURCE)
    blocked={r['id'] for name in ('development','held') for r in d[name]}
    blocked.update(r['id'] for r in bench['cards'])
    candidates={r['id']:r for r in reversed(source['practice']) if r['profile']=='reach' and r['id'] not in blocked}
    rng=np.random.default_rng(690104)
    values=[candidates[k] for k in rng.permutation(list(candidates))]
    fresh=[dict(c,seed=int(rng.integers(1,2**30)),reply_seed=int(rng.integers(1,2**30))) for c in values[:34]]
    train=fresh[:22]; held=fresh[22:]
    receipt=dict(ready=bench['ready'],dx=DX,ordered_practice=train,held=held,
        sources={str(p):_hash(p) for p in (fixture.PLAN,fixture.GEOMETRY,fixture.DEVELOPMENT,registered.PLAN,course.SOURCE)},
        feeds_sha256=_hash(FEEDS),limits=dict(opening=22,continuation=10,first_pass=10,batch=96,rounds=2,physical=2),
        rule='Measured release IDs independent of TT63 dev and held and all 32 TT68 registered games. Practice only uses native oracle before attempting actual direct commands. First 22 independent ordinary random opening streams plus matched continuation after actual scored teacher strokes across at most ten completed distinct release IDs. Twelve other broad random measured release IDs remain held for an independently RGB-tested policy. No target shaping, rule change, or teacher state for the acting pupil.')
    ROOT.mkdir(exist_ok=True,parents=True); _write(PLAN,receipt)
    print('practice',[(c['id'],c['reply_seed']) for c in train], 'held',[c['id'] for c in held],flush=True)


def checked():
    p=_read(PLAN)
    if any(_hash(Path(f))!=s for f,s in p['sources'].items()) or _hash(FEEDS)!=p['feeds_sha256']:
        raise ValueError('Changed TT69 source or measured starts')
    return p


def arm_rate(a,goal):
    ids=a._arm_ids[1][2]; stride=a._joint_increments[ids]*(a.control_dt/a.model.opt.timestep)
    return np.clip((np.asarray(goal)-a.data.ctrl[ids])/stride,-.70,.70).tolist()


def launch(a,c,shot,prefix,leg,ready):
    a.sample_clearance(False); a.launch(shot,card=c); a.partner.reset(seed=c['seed'])
    controls=[]
    for u in prefix:
        _,_,done,cap,_=a.exchange(u,mpc.FAR_COMMAND); controls.append(u)
        if done or cap:return None,controls
    for _ in range(150):
        motion=arm_rate(a,ready) if leg else ZERO
        _,_,done,cap,info=a.exchange(motion,mpc.FAR_COMMAND); controls.append(motion)
        if done or cap:break
        if any(e['kind']=='separation' and e.get('collider')=='rival_racket' for e in info['events']) and a.rival_contacts==leg+1:
            return State(a),controls
    return None,controls


def trial(a,snap,c,start,leg,motions,guard=None):
    snap.restore(a); action=list(start); contacts=[]; init=float(a.data.time)
    if guard is not None:guarded(a,guard)
    else:a.sample_clearance(False)
    try:
        with contact.physical_contacts(a,contacts):
            for u in motions:
                _,_,done,cap,_=a.exchange(u,mpc.FAR_COMMAND); action.append(u)
                if done or cap or a.rival_returns>=leg+2:break
        pp=a.point_placements(); off=pp['delivered'][leg] if len(pp['delivered'])>leg else None
        if any(v['draw']!='random' or v['target']!=list(draw(c,j,'random',None)[0]) for j,v in enumerate(pp['requested'])):
            raise ValueError('Changed real broad partner law')
        is_legal=bool(off and off['near_landed']); answered=is_legal and len(pp['delivered'])>leg+1
        obs=next((e for e in contacts if e['side']==1),None)
        body_clear=a.clearance_result() if guard is not None else None
        if guard is not None and body_clear['samples']!=round((a.data.time-init)/a.model.opt.timestep):
            raise ValueError('Incomplete shifted physical suffix clearance')
        return dict(controls=action,events=a.events.copy(),placements=pp,offer=off,landed=is_legal,answered=answered,
                    contact=obs,clearance=body_clear)
    finally:a.sample_clearance(False)


def verify(a,c,shot,tape,guard):
    a.launch(shot,card=c); a.partner.reset(seed=c['seed']); guarded(a,guard)
    try:
        for u in tape['controls']:
            if max(map(abs,u))>.8+1e-10:raise ValueError('Exceeded signed physical control rate')
            a.exchange(u,mpc.FAR_COMMAND)
        if digest_json(a.events)!=digest_json(tape['events']) or digest_json(a.point_placements())!=digest_json(tape['placements']):
            raise ValueError('Full independent physical coach tape changed')
        verify_clearance(a); return a.clearance_result()
    finally:a.sample_clearance(False)


def teach(a,guard,c,shot,leg,prefix,ready):
    tm=process_time(); snap,start=launch(a,c,shot,prefix,leg,ready)
    r=dict(card=c,shot=shot,leg=leg,fork=len(start),mode='ordinary-opening' if leg==0 else 'ordinary-earned-continuation',attempts=[])
    if snap is None:return dict(r,status='no valid far separation',cpu_s=process_time()-tm)
    stock=trial(a,snap,c,start,leg,[ZERO]*154)
    target=stock['offer']
    if target is None or target['near_bounce'] is None:return dict(r,status='no naturally delivered ball',cpu_s=process_time()-tm,stock=stock)
    r['stock']=stock
    snap.restore(a); coach=physical.low.Probe(a); roots=coach.basis()
    with patch.object(physical,'STAGE_TRIALS',96),patch.object(physical,'ROUNDS',2):
        a.clearance_margin=.0006
        try:choices,stats=physical.search(coach,[690109,c['seed'],leg],roots)
        finally:a.clearance_margin=0.
    r['native']=stats
    if stock['answered']:
        choices=[(stock['controls'][len(start):],dict(origin='settled ordinary body')),*choices]
    for rate,meta in choices[:2]:
        actual=trial(a,snap,c,start,leg,rate+[ZERO]*154,guard)
        arrived=actual['offer']; same=bool(arrived and all(same_trace(target[k],arrived[k]) for k in ('far_contact','near_bounce')))
        actual.update(same=same,forecast=meta)
        if same and actual['answered'] and actual['contact'] and not actual['clearance']['penetration_samples']:
            actual['strict_full']=verify(a,c,shot,actual,guard)
        r['attempts'].append(actual)
        if actual.get('strict_full',{}).get('penetration_samples')==0:
            break
    return dict(r,status='measured',cpu_s=process_time()-tm)


def paid(row):
    return next((r for r in row['attempts'] if r.get('strict_full',{}).get('penetration_samples')==0),None)


def dev_tapes():
    source={}; ready=_read(registered.PLAN)['ready']; result=[]
    for r in _read(fixture.DEVELOPMENT)['rows']:
        if r['dx']!=DX:continue
        spec=r['case']; f=spec['source']
        if f not in source:source[f]=_read(Path(f))
        c,row,_=physical.get(source[f],spec); act=next((e for e in r['motion']['physical'] if physical.received(e)),None)
        if act is None:continue
        result.append(dict(card=c,shot=row['shot'],leg=0,ready=ready,controls=act['commands'],events=act['events'],
                           placements=act['placements'],contact=act['contact'],source='TT63 lateral development'))
    return result


def continuation_sources(record):
    rows=dev_tapes()
    for r in record['rows']:
        if r['leg']!=0:continue
        a=paid(r)
        if a:
            rows.append(dict(card=r['card'],shot=r['shot'],controls=a['controls'],events=a['events'],
                            placements=a['placements'],contact=a['contact'],leg=0,source='TT69 first paid'))
    return rows[:10]


def acquire(limit,continuations):
    plan=checked(); rec=dict(protocol_sha256=_hash(PLAN),rows=[],complete=False)
    if OUT.exists():
        old=_read(OUT)
        if old['protocol_sha256']!=rec['protocol_sha256']:raise ValueError('Cannot resume different roster')
        rec=old
    shots=mpc.measured_launches(FEEDS)
    ready=np.asarray(plan['ready']); done={(r['card']['id'],r['leg']) for r in rec['rows']}
    jobs=[(c,shots[c['id']],0,[]) for c in plan['ordered_practice'][:limit]]
    if continuations:
        for d in continuation_sources(rec):
            own=d['placements']['delivered'][0]['near_contact']['time']
            cut=int(np.ceil((own+.123)/mpc.DT))
            jobs.append((d['card'],d['shot'],1,d['controls'][:cut]))
    with fixture.slid(DX),mpc.centered(ready),enabled():
        a=mpc.arena(width=64,height=48); cad=mpc.arena(width=64,height=48,detail='cad')
        guard=physical.GripClearance(cad.model,stock=True)
        try:
            for c,shot,leg,old in jobs:
                if (c['id'],leg) in done:continue
                r=teach(a,guard,c,shot,leg,old,ready); v=paid(r)
                rec['rows'].append(r); done.add((c['id'],leg))
                rec['complete']=limit==len(plan['ordered_practice']) and continuations and len(done)>=len(jobs)
                _write(OUT,rec)
                print('scored school',len(done),c['id'],leg,r['status'],round(r['cpu_s'],2),
                      r.get('native'),None if v is None else v['contact']['radial_m'],flush=True)
        finally:cad.close();a.close()


if __name__=='__main__':
    cmd=argparse.ArgumentParser(description=__doc__);cmd.add_argument('mode',choices=('plan','run'))
    cmd.add_argument('--n',type=int,default=10);cmd.add_argument('--next',action='store_true')
    args=cmd.parse_args(); plan() if args.mode=='plan' else acquire(args.n,args.next)
