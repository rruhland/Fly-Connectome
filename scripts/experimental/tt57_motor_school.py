"""Derivative-free motor school at attained ordinary RGB-controlled bodies."""

import argparse
from collections import Counter
from pathlib import Path
from time import process_time

import numpy as np
from scipy.spatial import cKDTree

from . import tt38_joint_mpc as mpc
from . import tt42_learning_course as course
from . import tt46_continuous_season as season
from . import tt56_scored_motor_atlas as atlas
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import Rollout
from .tt37_opponent_audit import CourtClearance
from .tt39_opportunity_map import projected_height
from .tt44_paired_credit import State, prefix
from .tt48_contact_patch import suffix
from .tt54_training_speedups import enabled

ROOT = Path('artifacts/tt57_motor_school')
PLAN = ROOT / 'protocol.json'
OUT = ROOT / 'practice.json.gz'
AUX = Path('artifacts/tt52_practice_dose')
SOURCES = {
    'practice': [season.ROOT/'learning-96.json.gz', AUX/'live-from-12.json.gz', AUX/'live-from-48-96.json.gz'],
    'held': [season.ROOT/'after96.json.gz', season.ROOT/'after128.json.gz'],
}
H = 26
T = 77
TRIALS = (128, 128, 96)
MUTATION = np.array([.145, .105, .124, .132, .17])
PHASE = np.arange(1, H+1)
HOLD = np.zeros(6)
AIM = mpc.AIM.copy()


def course_rows(data, held_ids, split):
    for i,(c, r) in enumerate(zip(data['cards'], data['rows'])):
        if (c['id'] in held_ids) != (split == 'held') or c['profile'] != 'reach':
            continue
        if not r['placements']['delivered'] or any(x['draw']!='random' for x in r['placements']['requested']):
            continue
        o = r['placements']['delivered'][-1]
        if o['near_landed'] or not all(o.get(k) for k in ('near_contact','far_contact','near_bounce','receive_line')):
            continue
        x = float(o['receive_line']['xyz'][0]); z = float(projected_height(o['receive_line']))
        lane = 'left' if x <= -.10 else 'right' if x >= .10 else 'middle'
        low, high = (.997, 1.054) if lane == 'middle' else (.979, 1.084)
        down = o['near_contact']['racket'][2] - o['near_contact']['location'][2]
        if not low <= z <= high or down < .038:
            continue
        observed = [n for n in r['searches'] if n.get('p') and -.99 < n['p'][1] < -.51
                    and o['far_contact']['step'] + 3 < n['tick'] < o['near_contact']['step'] - 9]
        if not observed:
            continue
        yield dict(id=c['id'], i=i, key=c['key'], stream=c['reply_seed'], leg=o['leg'], tick=observed[0]['tick'],
                   lane=lane, height=z, x=x, blade_below_m=down, late_offer_sha256=digest_json(o['near_bounce']))


def balanced(rows, *, count, per_id, rng):
    shuffled = [rows[j] for j in rng.permutation(len(rows))]
    groups = [[r for r in shuffled if r['lane']==k] for k in ('left','middle','right','middle')]
    selected, ids, released, offered = [], Counter(), set(), set()
    while len(selected) < count:
        visit = False
        for g in groups:
            available = [r for r in g if ids[r['id']]<per_id and
                         (r['id'],r['stream']) not in released and r['late_offer_sha256'] not in offered]
            if not available:
                continue
            # Spread distinct physical openings, then bound the expensive long prefixes.
            r=min(available,key=lambda v:(ids[v['id']],(v['tick']+799)//800))
            released.add((r['id'],r['stream'])); offered.add(r['late_offer_sha256']); ids[r['id']]+=1
            selected.append(r); visit = True
            if len(selected)>=count:break
        if not visit:
            break
    return selected


def plan():
    held_ids={c['id'] for c in _read(season.PLAN)['assessment']}
    risk_paths={str(p):_hash(p) for a in SOURCES.values() for p in a}
    with np.load(atlas.MODEL) as f:
        if held_ids & set(f['release'].tolist()):
            raise ValueError('Held opening in the scored donor atlas')
    by={k:[dict(r,source=str(p)) for p in docs for r in course_rows(_read(p),held_ids,k)]
        for k,docs in SOURCES.items()}
    rng=np.random.default_rng(570491)
    train=balanced(by['practice'],count=40,per_id=3,rng=rng)
    held=balanced(by['held'],count=12,per_id=2,rng=rng)
    if len(train)<30 or len(held)<8 or set(r['id'] for r in train)&set(r['id'] for r in held):
        raise ValueError('Insufficient release-disjoint matched forks')
    ROOT.mkdir(exist_ok=True,parents=True)
    record=dict(schema='tt57-attained-motor-school-1',sources=risk_paths,atlas_sha256=_hash(atlas.MODEL),
                optics_sha256=_hash(course.OPTIC),stock_schedule_sha256=_hash(season.PLAN),
                counts_by_lane={k:dict(Counter(r['lane'] for r in rr)) for k,rr in [('available practice',by['practice']),('practice',train),('held',held)]},
                limits=dict(stage_trials=TRIALS,frames=H,suffix_frames=T,physical_shortlist=5,pilot_cases=8),
                practice=train,held=held,
                scope='Exact ordinary restored bodies. Planning positions and native contact feedback are training/evaluation only. No scoring or physical searches on held releases for choosing a deployable movement. The only actor-facing learned rows are measured RGB/encoder features and physically replayed direct controls.')
    _write(PLAN,record)
    print({k:(len(record[k]),len({r['id'] for r in record[k]}),record['counts_by_lane'][k])
           for k in ('practice','held')},flush=True)


def context(x,q,ref,last,bounced):
    return np.r_[np.asarray(x)/[.095,.135,.07,.51,.85,.83],q[:5]/[.45,.35,.44,.45,.55],
                 (ref-q)[:5]/.15,last[:5]/2.4,float(bounced)]


def observed_prefix(a,cam,rig,card,row,tick,axes):
    history=prefix(a,card,row,tick,cam,axes,window=48)
    eye=mpc.Vision(rig,a.control_dt,raw=True)
    eye.eye=mpc.LearnedOptics(eye.eye,course.OPTIC)
    old=None
    for j,img,q,r in history[-48:]:
        x=mpc.observed(eye,j,img,q)
        if x is not None and -1.18 < x[1] < -.2 and x[4] < -.9:
            old=j,x.copy(),r.copy()
    q=a.joints.copy(); ref=a.data.ctrl[axes].copy()
    x=mpc.observed(eye,tick,cam.capture() if tick%2==0 else None,q)
    prev=history[-1][2]
    encoder=context(x,q,ref,(q-prev)/a.control_dt,eye.eye.latest.bounced) if x is not None and eye.eye.latest is not None else None
    return eye,x,q,ref,old,encoder,history


class Donors:
    def __init__(self):
        with np.load(atlas.MODEL) as f:
            good=f['reward']; self.keys=f['keys'][good]; self.moves=f['moves'][good]; self.ids=f['release'][good]
        self.knn=cKDTree(self.keys)

    def paths(self, query, x, avoid):
        distances, inds=self.knn.query(query,k=768)
        used=set(); output=[]; distances_out=[]
        for d,i in zip(distances,inds,strict=True):
            release=int(self.ids[i]); motion=self.moves[i]
            if release==avoid or release in used or any(np.max(abs(v[1][3:min(21,H):4]-motion[3:21:4]))<.035 for v in output):
                continue
            used.add(release)
            # Range limited phase alignment, computed only from the two optical states.
            ox=self.keys[i,:6]*np.r_[atlas.P_SCALE,atlas.V_SCALE]
            eta=lambda z: np.clip((z[1]+1.55)/max(1.6,-z[4]),.065,.47)
            ratio=np.clip(eta(ox)/eta(x),.68,1.42)
            motion=warp(motion,ratio,0.)
            output.append((release,motion)); distances_out.append(float(d))
            if len(output)>=15:
                break
        return output,distances_out


def warp(path, stretch, shift):
    ticks=(PHASE*stretch+shift).clip(0., len(path))
    return np.stack([np.interp(ticks,np.arange(len(path)+1),np.r_[0.,q]) for q in path.T],axis=1)


def mutation(rng,magnitude):
    # Smooth control paths with independent early and late components, not
    # prescribed near-contact joint signs. The stock rate limiter still applies.
    p=PHASE/H
    a,b=rng.normal(size=(2,5))*MUTATION[None,:]*magnitude
    flow=np.minimum(1.,2.9*p)[:,None]
    linear=flow*(a+(2.*p-1.)[:,None]*b)
    return np.c_[np.clip(linear,-.38,.38),np.zeros(H)]


def uniform_root(ref):
    return np.zeros((H,len(ref)))


def initial(stock, donor, rng):
    roots=[stock]+[m for _,m in donor]+[np.zeros_like(stock)]
    pool=list(roots); family=['stock-path']+['scored-donor']*len(donor)+['held-reference']
    while len(pool)<TRIALS[0]:
        j,k=rng.integers(len(roots),size=2)
        a=warp(roots[j],rng.uniform(.76,1.27),rng.uniform(-2.,2.))
        b=warp(roots[k],rng.uniform(.78,1.25),rng.uniform(-2.,2.))
        mix=rng.uniform(.18,.82)
        if len(pool)%3==0:
            gate=np.clip((PHASE-rng.uniform(5,18))/5.+.5,0.,1.)[:,None]
            mix=gate
        elif len(pool)%3==1:
            mix=rng.uniform(.12,.88,size=(1,6))
        c=(1.-mix)*a+mix*b
        pool.append(c+mutation(rng,rng.uniform(.42,1.08)))
        family.append('stock-crossover' if j==0 or k==0 else 'donor-crossover')
    return np.array(pool),family


def score(y,f):
    loss, legal=mpc.cost(y,f)
    # A stock legal score remains better than any modeled miss. Native
    # loss already measures net, launch, bounce, pace and blade edge.
    q=np.where(legal,loss+12.*np.maximum(y[:,12]-.056,0.),loss)
    return q,legal


def next_pool(a, quality, y, f, batch, rng, stage):
    legal=np.isfinite(y[:,0]); separated=np.flatnonzero(f[:,1]>.5)
    best=list(map(int,np.argsort(quality)[:12]))
    for col in (2,3,4,8):
        if len(separated):
            order=separated[np.argsort(f[separated,col])]
            best.extend((int(order[0]),int(order[-1])))
    best=list(dict.fromkeys(best))
    goals=[a[j] for j in best]; labels=['elite']*len(goals)
    while len(goals)<batch:
        j=int(rng.choice(best[:13])); k=int(rng.choice(best))
        if len(goals)%4!=0:
            k=j
        b=warp(a[j],rng.uniform(.9,1.10),rng.uniform(-1.25,1.25))
        if j!=k:
            z=rng.uniform(.12,.45,(1,6)); b=(1-z)*b+z*a[k]
        b=b+mutation(rng,rng.uniform(.28,.8)*(1.04-.18*stage)*(1. if not legal[j] else .54))
        goals.append(b); labels.append('refined-physical-path')
    return np.array(goals),labels


def native(engine, paths, ref):
    goals=ref+np.pad(paths,((0,0),(0,T-H),(0,0)),mode='edge')
    fb=np.zeros((len(paths),9))
    commanded,y=engine.trajectories(goals,AIM,(H+1)*mpc.DT,feedback=fb)
    q,legal=score(y,fb)
    return commanded,y,fb,q,legal


def physical(a,saved,card,row,spec,u,guard=None):
    try:
        return suffix(a,saved,card,row,dict(spec,fork=spec['tick']),u,clearance=guard)
    except ValueError as e:
        if str(e)!='The branch did not receive the same physical delivery':
            raise
        return dict(landed=False,answered=False,unmatched=True)


def build(a,cam,rig,engine,guard,stride,spec,doc,bank):
    start=process_time(); i=spec['i']; tick=spec['tick']; card,row=doc['cards'][i],doc['rows'][i]; axes=engine.axes
    a.sample_clearance(False)
    eye,x,q,ref,previous,features,_=observed_prefix(a,cam,rig,card,row,tick,axes)
    saved=State(a)
    old=np.asarray(row['actions'][tick:]); case=dict(spec,card=card,ready=doc['ready'])
    stock=physical(a,saved,card,row,spec,old)
    if stock['landed'] or stock.get('unmatched') or stock['event_sha256']!=digest_json(row['events']):
        raise ValueError('Original full physical failure failed to replay exactly')
    saved.restore(a)
    if x is None or features is None or x[4]>=-.9 or not -1.03<x[1]<-.47:
        return dict(case=case,stock=stock,visibility_rejected=True,cpu_s=process_time()-start)
    key=atlas.key(tick,x[:3],bool(eye.eye.latest.bounced),ref,previous)
    donor,distances=bank.paths(key,x,card['id'])
    ids=[d[0] for d in donor]
    rng=np.random.default_rng([570493,card['seed'],tick])
    path=atlas.reference_path(ref,old[:H],stride,a._joint_low[axes],a._joint_high[axes])[1:]
    baseline=mpc.tail(path,H)-ref
    paths,labels=initial(baseline,donor,rng)
    candidates=[]; log=[]; diagnostic=[]
    for round_id, batch in enumerate(TRIALS):
        rates,y,fb,quality,legal=native(engine,paths,ref)
        log.append(dict(trials=len(paths),contacts=int(np.sum(fb[:,1]>.5)),legal=int(legal.sum()),
                        legal_inside_62mm=int(np.sum(legal&(y[:,12]<.062))),
                        native_original_legal=bool(legal[0]) if round_id==0 else None))
        ranked=np.flatnonzero(legal)[np.argsort(quality[legal])]
        for j in ranked[:26]:
            candidates.append((float(quality[j]),paths[j].copy(),rates[j].copy(),y[j].copy(),fb[j].copy(),labels[j],round_id))
        if not len(ranked):
            j=int(np.argmin(quality)); diagnostic.append((float(quality[j]),paths[j].copy(),rates[j].copy(),None,fb[j].copy(),labels[j],round_id))
        if round_id+1<len(TRIALS):
            paths,labels=next_pool(paths,quality,y,fb,TRIALS[round_id+1],rng,round_id)
    choices=[]
    for e in sorted(candidates,key=lambda r:r[0]):
        if all(np.sqrt(np.mean((r[1]-e[1])**2))>.016 for r in choices):
            choices.append(e)
        if len(choices)>=5:
            break
    if not choices and diagnostic:
        choices=[min(diagnostic,key=lambda r:r[0])]
    arms=[]
    for loss,desired,u,y,ff,kind,round_id in choices:
        # Immediately take the precisely screened rate tape, stop moving
        # after the actual short stroke, and let the ordinary opponent answer.
        u=np.vstack((u[:H],np.zeros((T+26-H,6))))
        v=physical(a,saved,card,row,spec,u)
        v.update(native_legal=y is not None,native_cost=loss,native_feedback=ff.tolist(),native_origin=kind,
                 native_values=None if y is None else y.tolist(),round=round_id)
        # Preserve physically scored and safely executed movements. Follow-
        # through safety is certified on the exact issued tape, not on goals.
        if v['landed']:
            audited=physical(a,saved,card,row,spec,u,guard)
            if audited['event_sha256']!=v['event_sha256']:
                raise ValueError('CAD vs ordinary replay diverged')
            v['clearance']=audited['clearance']
            if not v['clearance']['penetration_samples']:
                v['issued_relative']=atlas.reference_path(ref,u[:H],stride,a._joint_low[axes],a._joint_high[axes])[1:].tolist()
                v['issued_relative']=(np.asarray(v['issued_relative'])-ref).tolist()
        arms.append(v)
        if v.get('issued_relative') and v['answered']:
            break
    a.sample_clearance(False)
    return dict(case=case,inputs=dict(x=x.tolist(),q=q.tolist(),ref=ref.tolist(),bounced=bool(eye.eye.latest.bounced),
                                     features=features.tolist(),atlas_key=key.tolist()),
                stock=stock,donor_release=ids,donor_distance=distances,native=log,arms=arms,cpu_s=process_time()-start)


def run(limit, p, out):
    design=_read(p)
    for name,checksum in design['sources'].items():
        if _hash(Path(name))!=checksum:
            raise ValueError(f'Changed motor school source {name}')
    if _hash(atlas.MODEL)!=design['atlas_sha256'] or _hash(course.OPTIC)!=design['optics_sha256']:
        raise ValueError('Changed optical observer or atlas')
    limit=min(limit,len(design['practice']))
    receipt=dict(protocol_sha256=_hash(p),operator_sha256=_hash(Path(__file__)),rows=[],complete=False)
    if out.exists():
        old=_read(out)
        if any(old[k]!=receipt[k] for k in ('protocol_sha256','operator_sha256')):
            raise ValueError('Changed branch receipt')
        receipt=old
    bank=Donors(); data={}; out.parent.mkdir(exist_ok=True,parents=True)
    with enabled():
        cad=mpc.arena(width=64,height=48,detail='cad')
        try:
            guard=CourtClearance(cad.model,articulated=True)
            for spec in design['practice'][len(receipt['rows']):limit]:
                if spec['source'] not in data:
                    data[spec['source']]=_read(Path(spec['source']))
                doc=data[spec['source']]
                with mpc.centered(doc['ready']):
                    a=mpc.arena(); cam,rig,stride,_,_=mpc.resources(a)
                    try:
                        r=build(a,cam,rig,Rollout(a,1),guard,stride,spec,doc,bank)
                    finally:
                        cam.close(); a.close()
                receipt['rows'].append(r)
                receipt['complete']=len(receipt['rows'])==len(design['practice'])
                _write(out,receipt)
                print(len(receipt['rows']),spec['id'],spec['lane'],round(spec['height'],3),round(r['cpu_s'],2),
                      [(x['contacts'],x['legal'],x['legal_inside_62mm']) for x in r.get('native',[])],
                      [(t['native_legal'],t['landed'],t['answered'],bool(t.get('issued_relative'))) for t in r.get('arms',[])],flush=True)
        finally:
            cad.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=('plan','run'))
    parser.add_argument('--n',type=int,default=8)
    parser.add_argument('--plan',type=Path,default=PLAN)
    parser.add_argument('--out',type=Path,default=OUT)
    args=parser.parse_args()
    plan() if args.mode=='plan' else run(args.n,args.plan,args.out)
