"""Report dense learner-visited training and complete native controls separately."""

from collections import Counter

import numpy as np

from . import tt106_self_course as study
from .tt4_transfer_trial import _hash,_read,_write
from .tt19_view_full_game import digest_json


def lessons(doc):
    counts=Counter(); phases=[]; recovered=[]
    for row in doc['rows']:
        for b in row['lessons']:
            h=b['height']
            line=b['target']['receive_line']
            if line is None or h is None:
                bucket='unmeasured'
            else:
                u=line['xyz'][0]
                wing=u>=.176 or u<=-.237
                band='low' if h<1.005 else 'transition' if h<1.047 else 'routine'
                bucket=band+('-wing' if wing else '-center')
            counts[bucket+'-offers']+=1
            counts[bucket+'-on-policy-paid']+=int(b['on_policy']['landed'])
            counts[bucket+'-teacher-paid']+=int(b['teacher'] is not None)
            phases.append(b['opening_y'])
            if b['teacher']:
                obj=b['teacher']
                recovered.append(dict(id=row['id'],leg=b['leg'],height=h,bucket=bucket,
                                      radius=obj['contact']['radial_m'],earliest_y=b['opening_y'],
                                      preparation_s=obj['contact']['time_s']-b['tick']*study.DT,
                                      family=obj.get('family','independent-full-spline')))
    return dict(complete=doc['complete'],points=len(doc['rows']),counts=dict(counts),teacher_rewards=recovered,
                own_offers=len(phases),median_early_y=float(np.median(phases)) if phases else None,
                course_legal_returns=sum(r['score']['learner'] for r in doc['rows']),
                body_native_trials=sum(a['tested'] for r in doc['rows'] for b in r['lessons'] for a in b['coach']),
                actual_teacher_attempts=sum(len(b['attempts']) for r in doc['rows'] for b in r['lessons']),
                student_early_ticks=sum(r['interventions'] for r in doc['rows']),
                student_candidate_screens=sum(len(r['screens']) for r in doc['rows']),
                screens_issued=sum(s['issued'] for r in doc['rows'] for s in r['screens']),student=doc.get('model'))


def paired(doc):
    total=np.zeros(2,dtype=int); out=[]
    for g in doc['rows']:
        a,b=[g['results'][k] for k in ('frozen','student')]
        s=np.array([a['score']['learner'],b['score']['learner']]); total+=s
        diff=next((j for j,(x,y) in enumerate(zip(a['actions'],b['actions'])) if x!=y),None)
        same_events=digest_json(a['events'])==digest_json(b['events'])
        first=[x['placements']['delivered'][:1] for x in (a,b)]
        matched=(all(first) and all(study.same_trace(first[0][0][z],first[1][0][z]) for z in ('far_contact','near_bounce')))
        out.append(dict(id=g['id'],returns=s.tolist(),first_diff=diff,same_events=same_events,
                        early=b['interventions'],checked=len(b['screens']),issued=sum(z['issued'] for z in b['screens']),
                        exact_first=bool(matched),terminal=[a['terminal'],b['terminal']]))
    return dict(complete=doc['complete'],returns=total.tolist(),rows=out)


def report():
    z=dict(protocol=_hash(study.PLAN))
    for path,fn in ((study.PRACTICE,lessons),(study.ROOT/'pilot-held.json.gz',paired),(study.HELD,paired)):
        if path.exists():
            z[path.name.split('.')[0]]=fn(_read(path))
    _write(study.ROOT/'summary.json',z)
    print(z,flush=True)


if __name__=='__main__':
    report()
