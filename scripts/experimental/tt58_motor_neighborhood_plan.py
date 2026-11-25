"""Freeze independently drawn, physically distinct neighboring replies before any receiving trial."""
import argparse
from pathlib import Path

import numpy as np

from . import tt57_motor_school as school
from . import tt56_central_training as exact
from .tt4_transfer_trial import _hash,_read,_write
from .tt19_view_full_game import digest_json
from .tt44_paired_credit import State

ROOT=Path('artifacts/tt58_motor_neighborhood')
PLAN=ROOT/'protocol.json'
OUT=ROOT/'scouts.json.gz'
ANCHORS=(9060,8595,12866)
POOLS=(('practice',14,4),('held',14,3))
SCALE=np.array([.045,.030,.34,.58])


def card_for(c,leg,seed):
    return exact.changed(c,leg,seed)


def phase_offer(o):
    f=o['receive_line'];return exact.deep(np.r_[f['xyz'],f['velocity']])


def freeze():
    d=_read(school.OUT); by={}
    for j in d['rows']:
        if any(a.get('issued_relative') and a['answered'] for a in j.get('arms',[])):
            by.setdefault(j['case']['id'],j)
    held={r['id'] for r in _read(school.PLAN)['held']}
    if not d['complete'] or held&set(ANCHORS):
        raise ValueError('New neighborhood roster cannot touch the motor school qualification set')
    rng=np.random.default_rng(580640)
    result=[]
    for i in ANCHORS:
        r=by[i];c=r['case'];src=Path(c['source']);o=_read(src);old=o['rows'][c['i']]
        leg=c['leg']; actual=old['placements']['delivered'][leg]; g=old['placements']['requested'][leg]
        win=next(a for a in r['arms'] if a.get('issued_relative') and a['answered'])
        if win['clearance']['penetration_samples']:
            raise ValueError('Original source scored action did not have strict physical clearance')
        requests={}
        for name,n,k in POOLS:
            cases=[]
            while len(cases)<n:
                seed=int(rng.integers(1,2**30));target=exact.draw(card_for(c['card'],leg,seed),leg,'random',None)[0]
                dx,dy,dz,dv=np.array(target)-g['target']
                if abs(dx)>.105 or abs(dz)>.042 or abs(dv)>.43:continue
                cases.append(dict(seed=seed,goal=list(target)))
            requests[name]=cases
        result.append(dict(id=i,case=c,source_sha256=_hash(src),gate_y=r['inputs']['x'][1],orig_input=r['inputs'],
             correct_tape_sha256=digest_json(win['controls']),view_phase=int(c['tick']),base_target=g['target'],
             anchor=int(np.floor(g['time_s']/school.mpc.DT+1e-7)),
             original=phase_offer(actual).tolist(),requested=requests))
    ROOT.mkdir(exist_ok=True,parents=True)
    _write(PLAN,dict(motor_school_sha256=_hash(school.OUT),school_protocol_sha256=_hash(school.PLAN),source=result,
         nominated_forbid=sorted(held),roles=POOLS,
         acquisition=dict(max_attempts=4, success='actual legal and answered, CAD-clean same offered ball',
             readout='nearest attained ball/encoder context, phase normalized against donor, with bounded position retargets as fallback practice proposals'),
         role='New local practice and independently seeded nearby assessment on three source-approved grounds. The actual outgoing stroke and earlier physical stage are intact. Admit broad existing target draws by a fixed generous request rectangle for efficiency, then measure real physical inbound line before receiving decisions. Prioritize moderate physically measured direction differences. Record the entire pool and freeze train/held assignments before any receiving results. No use of the motor school held releases or full games.'))


def select(options,anchor,k):
    if not options:return []
    base=np.array(anchor); good=[]
    for a in options:
        f=a['observed']
        if f is None:continue
        d=np.array(f)-base;n=np.linalg.norm(d/SCALE)
        if .24<=n<=2.1 and abs(d[0])<=.069 and abs(d[1])<=.043:
            good.append((n,a,d/SCALE))
    if not good:return []
    # Choose a meaningful nearest held variation, then span different signed
    # directions. Exact matches to earlier physical trajectories are removed.
    first=min(good,key=lambda w:abs(w[0]-.65)); selected=[first]; keep=[q for q in good if q is not first]
    while keep and len(selected)<k:
        i=max(range(len(keep)),key=lambda i:min(np.linalg.norm(keep[i][2]-q[2]) for q in selected)-.16*keep[i][0])
        w=keep.pop(i)
        if min(np.linalg.norm(w[2]-x[2]) for x in selected)>=.24:
            selected.append(w)
    return [v[1] for v in selected]


def scout():
    d=_read(PLAN); result=_read(OUT) if OUT.exists() else dict(protocol_sha256=_hash(PLAN),rows=[],complete=False)
    if result['protocol_sha256']!=_hash(PLAN) or d['motor_school_sha256']!=_hash(school.OUT):
        raise ValueError('Changed frozen development material')
    mpc=school.mpc
    for a0 in d['source'][len(result['rows']):]:
        c=a0['case'];src=Path(c['source']);doc=_read(src)
        if _hash(src)!=a0['source_sha256']:
            raise ValueError('Changed archive')
        row=doc['rows'][c['i']];card=c['card'];leg=c['leg'];cut=a0['anchor']
        obs={};assigned={};observed_later=[]
        with mpc.centered(c['ready']),school.enabled(cameras=False):
            arena=mpc.arena(width=64,height=48)
            try:
                arena.launch(row['shot'],card=card);arena.partner.reset(seed=card['seed'])
                for u in row['actions'][:cut]:arena.exchange(u,mpc.FAR_COMMAND)
                if (len(arena.partner.requests)!=leg or digest_json(arena.events)!=digest_json(
                     [e for e in row['events'] if e['time']<=arena.data.time+1e-8])):
                    raise ValueError('Chosen native afterstate does not preserve its achieved prior stroke')
                root=State(arena)
                for name,_,n in POOLS:
                    obs[name]=[]
                    for s in a0['requested'][name]:
                        root.restore(arena);arena.card=card_for(card,leg,s['seed'])
                        for u in row['actions'][cut:cut+130]:
                            _,_,end,cap,_=arena.exchange(u,mpc.FAR_COMMAND)
                            if end or cap or leg+1 in arena.crossings:break
                        p=arena.point_placements(); goal=p['requested'][leg]['target'] if len(p['requested'])>leg else None
                        if goal is not None and goal!=s['goal']:
                            raise ValueError('Live physical response chose a different standard goal')
                        d0=p['delivered'][leg] if len(p['delivered'])>leg else None
                        f=phase_offer(d0).tolist() if d0 and d0['receive_line'] else None
                        fingerprint=digest_json((d0['far_contact'],d0['near_bounce'])) if d0 else None
                        obs[name].append(dict(s,observed=f,exact_offer=fingerprint,ticks=arena.steps-cut))
                    items=select([x for x in obs[name] if x['exact_offer'] not in observed_later],a0['original'],n)
                    assigned[name]=items
                    observed_later.extend(v['exact_offer'] for v in items)
            finally:arena.close()
        result['rows'].append(dict(id=c['id'],full=obs,assigned=assigned))
        result['complete']=len(result['rows'])==len(d['source']);_write(OUT,result)
        print(c['id'],[(k,len(obs[k]),[(v['seed'],v['observed'][:2]) for v in w]) for k,w in assigned.items()],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('plan','scout'));q=p.parse_args()
    freeze() if q.mode=='plan' else scout()
