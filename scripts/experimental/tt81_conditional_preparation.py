"""Learn cautious optical preparation on the real +12 cm body before authorizing a strike."""

import argparse
from pathlib import Path

import numpy as np

from . import tt69_clamp_school as school
from . import tt69_clamp_policy as taught
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import BallFlight

ROOT=Path('artifacts/tt81_conditional_preparation')
PLAN=ROOT/'protocol.json'
ROWS=ROOT/'calibration.json.gz'
MODEL=ROOT/'learner.json'
OUT=ROOT/'held.json.gz'
GAMES=ROOT/'full.json.gz'
BASE=taught.mpc
SCALE=np.array([.13,.075,.63,.85,.88,.45])
LOWS=(.935,1.000)
ROUTINE=(1.026,1.150)
STRENGTH=.68
LIMIT=np.array([.48,.34,.43,.42,.54])
RAY_DEPTH=-1.45
LOW_THRESHOLD=1.005
CAMERAS=9


def plan():
    d=_read(taught.QUAL); ready=d['ready']; shots=BASE.measured_launches(school.FEEDS)
    # Registered whole-point roots remain untrained. This is a different
    # explicitly held random stream from all past near-height tests.
    cards=d['whole']; rng=np.random.default_rng(810087); seen=[]; selected=[]; groups={}
    with taught.fixture.slid(school.DX),BASE.centered(ready),school.enabled():
        a=BASE.arena(width=64,height=48)
        try:
            for card in cards:
                accepted=set()
                for _ in range(42):
                    c=dict(card,seed=int(rng.integers(1,2**30)),reply_seed=int(rng.integers(1,2**30)))
                    saved,prefix=school.launch(a,c,shots[c['id']],[],0,ready)
                    r=None if saved is None else school.trial(a,saved,c,prefix,0,[school.ZERO]*113)
                    o=None if r is None else r['offer']
                    h=None if not o or not o['receive_line'] else taught.projected_height(o['receive_line'])
                    x=None if h is None else o['receive_line']['xyz'][0]
                    mode=None
                    if x is not None:
                        if LOWS[0]<=h<LOWS[1] and -.28<x<.19:mode='low'
                        elif ROUTINE[0]<=h<ROUTINE[1] and abs(x)<.36:mode='routine'
                    row=dict(card=c,h=h,x=x,grade=mode,actual_offer=None if h is None else
                             {k:o[k] for k in ('far_contact','near_bounce','receive_line')})
                    seen.append(row)
                    if mode and mode not in accepted:
                        selected.append(row);accepted.add(mode)
                    if len(accepted)==2:break
                groups[str(card['id'])]=sorted(accepted)
        finally:a.close()
    if sum(r['grade']=='low' for r in selected)<3 or sum(r['grade']=='routine' for r in selected)<4:
        print('short natural coverage',groups,flush=True)
    r0=[c for c in _read(school.registered.PLAN)['cards'] if c['id']!=10544]
    whole=[dict(c,seed=int(rng.integers(1,2**30)),reply_seed=int(rng.integers(1,2**30))) for c in r0[:16]]
    ROOT.mkdir(exist_ok=True,parents=True)
    _write(PLAN,dict(school_sha256=_hash(school.OUT),raw_field_sha256=_hash(taught.MODEL),old_qual_sha256=_hash(taught.QUAL),
        selector_seed=810087,held=selected,headless_scans=seen,whole=whole,ready=ready,
        law='Every selected receive is from an unaltered broad random far draw under native physics. Natural first-offer low vs routine grades selected before pupil from neutral-control actual event tapes. Later whole roots are independently reseeded regular games, never conditioned on delivery outcome. No held ID enters the calibration or pose bank. The existing TT68 release 10544 is diagnostic only.',
        gate='At least one new physically answered same-ball sub-1 return and zero lost comparator hits on the matched fresh set before opening the unfiltered live 16-point course. Both arms retain standard live credit.'))
    print('new natural held',len(seen),[(r['card']['id'],r['grade'],round(r['h'],3)) for r in selected],flush=True)


def cue(path,x):
    q=taught.point_at(path,RAY_DEPTH)
    if q is None:return None
    return np.r_[taught.marker(q),x[1]]


def read_case(a,cam,rig,flight,r):
    stock=r.get('stock')
    if stock is None or not stock['offer'] or not stock['offer']['receive_line']:return []
    o=stock['offer']; far=o['far_contact']['step']; near=o['near_bounce']['step']; c=r['card']; tid=c['id']
    h=float(taught.projected_height(o['receive_line'])); x_true=float(o['receive_line']['xyz'][0])
    a.launch(r['shot'],card=c); a.partner.reset(seed=c['seed'])
    eye=BASE.Vision(rig,a.control_dt,raw=True); eye.eye=BASE.LearnedOptics(eye.eye,taught.course.OPTIC)
    rows=[]
    for t,u in enumerate(stock['controls'][:near+5]):
        show=t%2==0 and t>=far-16
        x=BASE.observed(eye,t,cam.capture() if show else None,a.joints)
        if show and x is not None and far<=t and x[4]<-.90 and -.31<x[1]<.80:
            features=cue(taught.p_arc(flight,eye.eye,t,x),x)
            if features is not None:rows.append(dict(id=tid,leg=r['leg'],features=features.tolist(),true_h=h,true_x=x_true))
        _,_,done,cap,_=a.exchange(u,BASE.FAR_COMMAND)
        if done or cap:break
    if digest_json(a.events)!=digest_json(stock['events'][:len(a.events)]):
        raise ValueError('Exercise optical evidence altered a physical event prefix')
    # Fixed camera dose per actual launch, so longer visible cases do not
    # receive arbitrarily more statistical weight in the height calibrator.
    return [rows[k] for k in np.unique(np.linspace(0,len(rows)-1,min(CAMERAS,len(rows))).astype(int))] if rows else []


def acquire():
    d=_read(PLAN); src=_read(school.OUT); ready=d['ready']
    if d['school_sha256']!=_hash(school.OUT):raise ValueError('Changed native training response')
    r=dict(protocol=_hash(PLAN),features=[],sources=[],complete=False)
    if ROWS.exists():
        o=_read(ROWS)
        if o['protocol']!=r['protocol']:raise ValueError('Cannot resume changed calibration protocol')
        r=o
    with taught.fixture.slid(school.DX),BASE.centered(ready),school.enabled():
        a=BASE.arena(); cam, rig,*_=BASE.resources(a); flight=BallFlight(a)
        try:
            for case in src['rows'][len(r['sources']):]:
                added=read_case(a,cam,rig,flight,case)
                r['sources'].append(dict(id=case['card']['id'],leg=case['leg'],kept=len(added),real_return=bool(school.paid(case))))
                r['features'].extend(added);r['complete']=len(r['sources'])==len(src['rows']);_write(ROWS,r)
                print('calibration',case['card']['id'],case['leg'],len(added),flush=True)
        finally:cam.close();a.close()


def ridge(x,y):
    X=np.c_[np.ones(len(x)),x]
    regular=np.r_[1.4,np.full(x.shape[1],6.4)]
    return np.linalg.solve(X.T@X+np.diag(regular),X.T@y)


def fit():
    q=_read(PLAN); d=_read(ROWS); m=_read(taught.MODEL)
    if not d['complete'] or d['protocol']!=_hash(PLAN) or q['raw_field_sha256']!=_hash(taught.MODEL):
        raise ValueError('Missing or changed preparation practice')
    groups=np.array([r['id'] for r in d['features']]); h=np.array([r['true_h'] for r in d['features']]); x=np.array([r['features'] for r in d['features']])
    center=x.mean(axis=0); f=(x-center)/SCALE; y=h-x[:,1]; plain=[]; residual=[]
    ids=set(groups.tolist())
    excluded={v['card']['id'] for v in q['held']}|{v['id'] for v in q['whole']}|{10544}
    if ids&excluded:
        raise ValueError('Disjoint held or diagnostic release leaked')
    for release in sorted(ids):
        own=groups==release; w=ridge(f[~own],y[~own]); z=x[own,1]+np.c_[np.ones(own.sum()),f[own]]@w
        plain.extend((x[own,1]-h[own]).tolist());residual.extend((z-h[own]).tolist())
    σ=float(np.clip(np.sqrt(np.mean(np.asarray(residual)**2)),.019,.064)); weights=ridge(f,y)
    # Collect references exactly from scored public lessons. Only a moderate
    # prefix correction is ever issued; recorded swing/contact is never sent.
    world=[]
    for s in m['rows']:
        if s['radius']>.061:continue
        if s['id'] in excluded:raise ValueError('Same-body movement from excluded release')
        columns=np.asarray(s['goal']).T
        pre=np.array([np.interp(.64,s['y'],v) for v in columns])
        landed=np.asarray(s['contact_goal'])
        world.append(dict(id=s['id'],leg=s['leg'],key=s['key'],pre=pre.tolist(),landing=landed.tolist(),radius=s['radius']))
    r=dict(protocol=_hash(PLAN),calibration=_hash(ROWS),field=_hash(taught.MODEL),keys=world,
           center=center.tolist(),scale=SCALE.tolist(),weights=weights.tolist(),spread=σ,
           samples=len(x),release_ids=sorted(ids),optical_base_rmse_m=float(np.sqrt(np.mean(np.asarray(plain)**2))),
           corrected_crossfit_rmse_m=float(np.sqrt(np.mean(np.asarray(residual)**2))),strength=STRENGTH,actuator_limit=LIMIT.tolist())
    _write(MODEL,r)
    print('fit',len(world),len(x),len(ids),'uncalibrated vs crossfit',r['optical_base_rmse_m'],r['corrected_crossfit_rmse_m'],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('plan','capture','fit'));a=p.parse_args()
    {'plan':plan,'capture':acquire,'fit':fit}[a.mode]()
