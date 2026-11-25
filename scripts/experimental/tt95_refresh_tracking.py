"""Refresh an issued patch-motion window against the newer public incoming arc."""

from pathlib import Path

import numpy as np

from . import tt95_public_window as window
from . import tt95_patch_lessons as lab
from . import tt88_public_inverse as gate
from . import tt90_installation_learning as tt90
from .tt37_optical_clinic import OpticalClinic
from .tt37_opponent_audit import guarded
from .tt4_transfer_trial import _hash,_read,_write
from .tt19_view_full_game import digest_json

ROOT=lab.ROOT
PLAN=ROOT/'refresh.protocol.json'
OUT=ROOT/'refresh-matched.json.gz'


def refs(rates,reference,plant,lo,hi):
    cmds=np.vstack([rates, np.zeros((gate.H,6))])[:gate.H]
    targets=[]
    q=reference.copy()
    for u in cmds:
        q=np.clip(q+plant*u,lo,hi)
        targets.append(q-reference)
    return np.asarray(targets)


def allowed(candidate):
    return candidate[1]['legal'] and candidate[1]['robust']>=1


def play(a,cam,rig,engine,planner,clear,c,row,anchor,fixed):
    tape=tt90.system.q.tape
    tape.setup(a,c,row,anchor)
    guarded(a,clear)
    ready=a.data.ctrl[a._arm_ids[1][2]].copy()
    ref=ready.copy()
    step,lo,hi=tt90.library.limits(a)
    eye=tt90.BASE.Vision(rig,a.control_dt,raw=True)
    eye.eye=tt90.BASE.LearnedOptics(eye.eye,tt90.COURSE.OPTIC)
    moved=[];hits=[];choices=[];active=None
    archive=fixed['commands']
    spent=next(n['tick'] for n in fixed['notes'] if n['issued'])
    refresh_windows=0
    try:
        with tape.impact.physical_contacts(a,hits):
            for tick in range(len(archive)+120):
                q=a.joints.copy()
                x=tt90.BASE.observed(eye,tick,cam.capture() if tick%2==0 else None,q)
                engine.observe(tick,eye.eye,q,ref)
                later=x is not None and x[4]<-1.1 and -1.40<x[1]<-.36
                if refresh_windows<3 and tick-spent>=5 and later and engine.sync():
                    refresh_windows+=1
                    spent=tick
                    new,proof=planner.roots(engine,ref)
                    old=np.asarray(archive[tick:],dtype=float) if active is None else active[1][tick-active[0]:]
                    current=refs(old.reshape(-1,6),ref,step,lo,hi)
                    new=np.stack([current,*new])
                    proof=[dict(method='kept_previous'),*proof]
                    ranked,stamp=(gate.population.few.rank_paths(engine,new,ref,proof) if proof else
                                  ([],dict(candidates=0)))
                    proposed=[x for x in ranked if allowed(x) and x[1]['method']!='kept_previous']
                    keeping=next((x for x in ranked if x[1]['method']=='kept_previous'),None)
                    picked=min(proposed,key=lambda x:x[1]['score']) if proposed else None
                    if picked and (keeping is None or not allowed(keeping)
                                   or picked[1]['score']+.28<keeping[1]['score']):
                        active=(tick,*picked)
                    else:
                        picked=None
                    choices.append(dict(tick=tick,refresh=True,x=x.tolist(),signals=stamp,
                                        retained=None if keeping is None else keeping[1],
                                        replacement=None if picked is None else picked[1]))
                if active is None:
                    u=archive[tick] if tick<len(archive) else [0.]*6
                else:
                    age=tick-active[0]
                    u=active[1][age].tolist() if age<len(active[1]) else [0.]*6
                moved.append(u)
                ref=np.clip(ref+step*u,lo,hi)
                _,_,done,cap,_=a.exchange(u,tt90.BASE.FAR_COMMAND)
                if done or cap or a.rival_returns>=2:
                    break
        r=tape.trace_result(a,anchor,hits)
        return dict(**r,choices=choices,commands=moved,clearance=a.clearance_result())
    finally:
        a.sample_clearance(False)


def run():
    one=_read(window.OUT);native=_read(tt90.system.OUT)
    if not one['complete']:
        raise ValueError('Complete the initial public window comparison')
    source={r['case']['id']:r for r in native['rows'] if r['body']==1}
    exclude=set(source)
    docket=dict(schema='tt95-receding-window-1',base=_hash(window.OUT),cases=[r['case']['id'] for r in one['rows']],
                rule='No changes to training examples or first contact-window proposal. Before the near strike, '
                     'replay the certified first public command exactly and recheck its still-issued motor '
                     'suffix and new proposed windows against the same updated '
                     'public camera belief; keep the original control unless a newly legalized or lower-cost '
                     'candidate robustly improves it. Up to three body receding updates per incoming. '
                     'This is a mechanism diagnostic on two previously seen public miss traces, not held generalization.')
    if not PLAN.exists():
        _write(PLAN,docket)
    if _read(PLAN)!=docket:
        raise ValueError('Changed the receding comparison receipt')
    out=_read(OUT) if OUT.exists() else dict(protocol=_hash(PLAN),rows=[],complete=False)
    if out['protocol']!=_hash(PLAN):
        raise ValueError('Changed the receding search protocol')
    if out['complete']:
        return
    m,ready,_=tt90.material()
    q,tape=tt90.system.q,tt90.system.q.tape
    with tt90.school.enabled(),tt90.system.installed(m['v'],m['mount']),tt90.BASE.centered(ready):
        a=tt90.BASE.arena()
        cad=tt90.BASE.arena(width=64,height=48,detail='cad')
        guide=q.GripClearance(cad.model,stock=True)
        cam,rig,*_=tt90.BASE.resources(a)
        try:
            for key in docket['cases'][len(out['rows']):]:
                case=source[key]['case']
                c,row,anchor=q.get(_read(Path(case['source'])),case)
                raw=source[key]['trial']['stock_commands']
                base=next(x['results'][1] for x in one['rows'] if x['case']['id']==key)
                engine=OpticalClinic(body=True)
                try:
                    inv=window.Proposal(engine.twin,'window',exclude)
                    n=play(a,cam,rig,engine,inv,guide,c,row,anchor,base)
                finally:
                    engine.twin.close()
                if not n['same'] or not lab.inv.tt72.first_ball(n,raw['placements']['delivered'][0]):
                    raise ValueError('Receding update changed the actual matched first delivery')
                first=min((d['tick'] for d in n['choices'] if d['refresh'] and d['replacement']),default=None)
                if first is None:
                    if digest_json(n['events'])!=digest_json(base['events']):
                        raise ValueError('New readout changed the issued result without replacement')
                elif n['commands'][:first]!=base['commands'][:first]:
                    raise ValueError('Before the first accepted revision, original commands did not match')
                if q.received(n):
                    trial=tape.matched_motion(a,c,row,anchor,n['commands'],guide)
                    if not q.received(trial) or digest_json(n['events'])!=digest_json(trial['events']):
                        raise ValueError('Returned motion did not repeat')
                    n['repeat_sha256']=digest_json(trial['events'])
                out['rows'].append(dict(case=case,first_refresh=first,fixed_landed=base['landed'],receding=n))
                out['complete']=len(out['rows'])==len(docket['cases'])
                _write(OUT,out)
                print('TT95 refreshed',key,first,n['landed'],n['answered'],n['depart'],flush=True)
        finally:
            cam.close();cad.close();a.close()


if __name__=='__main__':
    run()
