"""Independent exact replay of the canted installation's earned returns."""

from pathlib import Path

from . import tt84_installation_system as sys
from . import tt84_installation_compare as lateral
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt37_opponent import draw


OUT=sys.ROOT/'replay.json'


def verify():
    p=sys.contract(); a=_read(sys.OUT); b=_read(sys.BUILD); side=_read(lateral.OUT)
    if not a['complete'] or not side['complete'] or a['build']!=_hash(sys.BUILD):
        raise ValueError('Complete the exact actual fixture dockets first')
    r=dict(source={str(f):_hash(f) for f in (sys.PLAN,sys.OUT,sys.BUILD,lateral.OUT)},
           groups={},reproduced=[],complete=False)
    conditions=dict(stock=[q for q in a['rows'] if q['body']==0],lateral12=side['rows'],
                    combined=[q for q in a['rows'] if q['body']==1]);
    for label,data in conditions.items():
        r['groups'][label]={}
        for role in ('low','routine'):
            rows=[v for v in data if v['case']['role']==role]
            r['groups'][label][role]=dict(n=len(rows),coached=sum(v['real'] for v in rows),interior=sum(v['broad'] for v in rows),
                                       transferred_rate_tape=sum(v['reference'] for v in rows),
                                       preserved=sum(v['reference'] or v['real'] for v in rows),
                                       matched=sum(v['trial']['stock_commands']['same'] for v in rows),
                                       exact_openings=sum(v['trial']['matched_prefix'] for v in rows))
    selected=b['rows'][b['chosen']]; c0=selected['candidate']; c=selected['commission']['ready']; cache={}
    with sys.fixture.school.enabled(cameras=False),sys.installed(c0['v'],c0['mount']),sys.MP.centered(c):
        live=sys.MP.arena(width=64,height=48); cad=sys.MP.arena(width=64,height=48,detail='cad')
        try:
            clear=sys.q.GripClearance(cad.model,stock=True)
            for s in conditions['combined']:
                case=s['case']; q=s['trial']; f=case['source']; doc=cache[f] if f in cache else cache.setdefault(f,_read(Path(f)))
                card,old,want=sys.q.get(doc,case)
                trials=([('archived-rates',q['stock_commands'])] if s['reference'] else [])
                trials.extend(('coach',v) for v in q['physical'] if v['landed'] and sys.q.received(v))
                for arm,past in trials:
                    now=sys.q.tape.matched_motion(live,card,old,want,past['commands'],clear)
                    if (not sys.q.received(now) or now['clearance']!=past['clearance'] or
                            digest_json(now['events'])!=digest_json(past['events'])):
                        raise ValueError('Legal fixture witness changed on independent replay')
                    score=audit(now['events'],opening_side=1)
                    if score['learner']!=1 or score['opponent']<2:
                        raise ValueError('Returned ball did not get a legal real answer')
                    if any(v['draw']!='random' or v['target']!=list(draw(card,i,'random',None)[0])
                           for i,v in enumerate(now['placements']['requested'])):
                        raise ValueError('Reactive physical opponent requested a nonstandard draw')
                    r['reproduced'].append(dict(id=case['id'],role=case['role'],arm=arm,contact=now['contact'],
                          referee=digest_json(now['events']),solid_samples=now['clearance']['samples'],broad=sys.q.received(now,True)))
                _write(OUT,r)
        finally:live.close();cad.close()
    r['complete']=True;_write(OUT,r)
    print(r['groups'],len(r['reproduced']),sum(v['solid_samples'] for v in r['reproduced']),flush=True)


if __name__=='__main__':verify()
