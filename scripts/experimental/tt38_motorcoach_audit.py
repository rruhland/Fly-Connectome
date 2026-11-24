"""Independent action/CAD replay and exact comparison for the privileged TT38 teacher."""

import argparse
from pathlib import Path
import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt31_ready import centered
from . import tt33_joint as joint
from .tt33_joint import FAR_COMMAND
from .tt37_opponent import arena
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded, verify_clearance
from .tt38_motorcoach import OUT, SOURCE


def near_departure(row, number):
    shots=[x for x in row['score']['shots'] if x['side']==1]
    if len(shots)<number:
        return None
    shot=shots[number-1]
    events=row['events']
    leave=next(x for x in events if x['kind']=='separation' and x.get('collider')=='learner_racket' and x['time']>shot['hit'])
    f=next((x for x in events if x['kind']=='contact' and x['side']==-1 and x['time']>shot['landing']),None)
    answered=next((x for x in row['score']['shots'] if x['side']==-1 and x['hit']>shot['landing']),None)
    return dict(hit=shot['hit'],landing=shot['landing_xyz'],departure_mps=leave['velocity'],
                spin=leave['spin'],far_touch=f is not None,far_answered=answered is not None)


def summarize(output):
    witness=_read(SOURCE)
    champion=max(witness['rows'],key=lambda r:r['score']['learner'])
    report=dict(archived_terminal=near_departure(champion,19),diagnostic=[],fresh=[],extended=[],long_fresh=[])
    for f in sorted(output.glob('motorcoach-suffix-*.json.gz')):
        doc=_read(f)
        row=doc['rows'][0] if doc['rows'] else None
        if row is None:
            report['diagnostic'].append(dict(tape=f.name,fork=doc['fork_tick'],selected=False))
            continue
        prefix=doc['fork_tick']
        if row['actions'][:prefix]!=champion['actions'][:prefix]:
            raise ValueError('Suffix changed the archived motor prefix')
        a=champion['score']['shots'][:36]
        if digest_json(row['score']['shots'][:36])!=digest_json(a):
            raise ValueError('The diagnostic changed one of the first 18 stroke pairs')
        report['diagnostic'].append(dict(tape=f.name,fork=prefix,learner=row['score']['learner'],
            on_court_19=near_departure(row,19),returns=row['score']['returns'],end=row['physical']['reason']))
    paths={k:output/f'motorcoach-{k}.json.gz' for k in ('baseline','dynamic')}
    if all(p.exists() and _read(p).get('complete') for p in paths.values()):
        d,b=(_read(paths[k]) for k in ('dynamic','baseline'))
        if d['cards']!=b['cards']:
            raise ValueError('Whole-point controls were not seeded identically')
        for c,n,o in zip(d['cards'],d['rows'],b['rows'],strict=True):
            keys=('target','far_contact','near_bounce','receive_line')
            a,z=n['placements']['delivered'],o['placements']['delivered']
            first=bool(a and z and all(digest_json(a[0][k])==digest_json(z[0][k]) for k in keys))
            request_prefix=list(zip(n['placements']['requested'],o['placements']['requested']))
            if any(a['target']!=b['target'] for a,b in request_prefix):
                raise ValueError('An independent draw was changed')
            def brief(r):
                return dict(own=r['score']['learner'],total=r['score']['returns'],
                            prior_later=r['physical']['later_replies'],terminal=r['physical']['reason'],
                            dynamic=sum(a.get('executed_dynamic',False) for a in r['records']))
            report['fresh'].append(dict(card=c,first_offer_exact=first,matching_request_prefix=len(request_prefix),
                dynamic=brief(n),baseline=brief(o)))
    for f in sorted(output.glob('motorcoach-extended-*.json.gz')):
        x=_read(f)
        if not x.get('complete'):
            continue
        r=x['rows'][0]
        predecessor=output/x['source']
        old=_read(predecessor)['rows'][0]
        if r['actions'][:x['prefix_ticks']]!=old['actions'] or _hash(predecessor)!=x['source_sha256']:
            raise ValueError('Extended receipt is not an exact action-prefix continuation')
        own=r['score']['learner']
        report['extended'].append(dict(tape=f.name,own=own,total=r['score']['returns'],cap=x['cap'],
            stop=r['stop'],reason=r['physical']['reason'],last_legal=near_departure(r,own),
            legacy_memories_restored=x['legacy_memories_restored'],exact_prefix=True))
    p=output/'motorcoach-fresh-long.json.gz'
    if p.exists() and (x:=_read(p)).get('complete'):
        for c,r in zip(x['cards'],x['rows'],strict=True):
            own=r['score']['learner']
            report['long_fresh'].append(dict(card=c,own=own,total=r['score']['returns'],cap=x['cap'],
                stop=r['stop'],reason=r['physical']['reason'],last_legal=near_departure(r,own) if own else None))
    _write(output/'motorcoach-summary.json',report)


def verify(paths, output):
    receipts=[]
    for f in paths:
        d=_read(f)
        if not d['rows']:
            continue
        chosen=max(range(len(d['rows'])),key=lambda k:d['rows'][k]['score']['learner'])
        c,r=d['cards'][chosen],d['rows'][chosen]
        with centered(d['ready']):
            a=arena()
            shadow=arena(detail='cad')
            try:
                a.launch(r['shot'],card=c)
                a.partner.reset(seed=c['seed'])
                guarded(a,CourtClearance(shadow.model,articulated=True))
                contacts=[]
                with contact_observer(a,contacts):
                    for u in r['actions']:
                        a.exchange(u,FAR_COMMAND)
                if any(digest_json(x)!=digest_json(r[k]) for k,x in (
                    ('events',a.events),('placements',a.point_placements()),('score',audit(a.events,opening_side=1)))):
                    raise ValueError('CAD-guarded replay changed the observed point')
                verify_clearance(a)
                receipts.append(dict(tape=f.name,source_sha256=_hash(f),card=c,near_legal=r['score']['learner'],
                    events_sha256=digest_json(a.events),clearance=a.clearance_result(),native_contacts=contacts))
            finally:
                a.close()
                shadow.close()
    _write(output/'motorcoach-audit.json',dict(replays=receipts))


def demo(path, output, *, metadata_only=False):
    """Record only rewarded strokes at and after the intervention, as practice."""
    d=_read(path)
    r=d['rows'][0]
    c=d['cards'][0]
    fork=d['fork_tick']
    path_npz=output/'motorcoach-demo.npz'
    if metadata_only:
        with np.load(path_npz) as prior:
            arrays=dict(prior)
    else:
        with centered(d['ready']):
            a=arena()
            stereo,rig,*_=joint.resources(a)
            try:
                a.cards=iter([c])
                columns=joint.observe(a,stereo,rig,r|dict(pupil_ticks=[fork-1]),raw=True)
            finally:
                stereo.close()
                a.close()
        names=('optical','q_rad','velocity_rad_s','target_rad','signed_command','time_index')
        arrays={name:np.asarray(x) for name,x in zip(names,columns,strict=True)}
        np.savez_compressed(path_npz,**arrays)
    legs=set(arrays['time_index'][:,1].astype(int))
    landed=[p for p in r['placements']['delivered'] if p['near_landed']]
    _write(output/'motorcoach-demo.json',dict(source=path.name,source_sha256=_hash(path),fork_tick=fork,
        usage='practice only, diagnostic intervention point, exclude from held evaluation',
        samples=len(arrays['optical']),replayed_events_exact=True,
        inputs=['RGB-derived ball x/y/z/vx/vy/vz','joint angle and velocity from encoders'],
        teacher_targets=['actual signed command','actual servo reference'],metadata_only_columns=['time_index'],
        rewarded_strokes=[dict(leg=int(i+1),lane=landed[i]['observed_lane'],samples=int(np.sum(arrays['time_index'][:,1]==i)))
                          for i in sorted(legs)]))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=OUT)
    p.add_argument('--verify',type=Path,nargs='+')
    p.add_argument('--demo',type=Path)
    p.add_argument('--demo-metadata-only',action='store_true')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    if a.verify:
        verify(a.verify,a.output)
    elif a.demo:
        demo(a.demo,a.output,metadata_only=a.demo_metadata_only)
    else:
        summarize(a.output)
