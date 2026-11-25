"""Rescore and independently replay the lateral fixture's actual winning tapes."""

from pathlib import Path

from . import tt63_lateral_clamp as lateral
from . import tt63_lateral_optics as optics
from . import tt63_fixture_qualification as q
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_endurance import audit
from .tt37_opponent import draw


OUT=lateral.ROOT/'verification.json'


def verify():
    lateral.check(); g=_read(lateral.GEOMETRY); photo=_read(optics.OUT)
    mpc=lateral.BASE
    sheets=(lateral.DEVELOPMENT,lateral.HOLD); dz={r['dx']:r for r in g['rows']}; cache={}
    if not photo['complete'] or _hash(lateral.GEOMETRY)!=photo['geometry']:
        raise ValueError('Need the completed paired photo receipt')
    shots=0; pairs=0
    for r in photo['rows']:
        exact=r['arms'][0]['obs']==r['arms'][1]['obs']
        if exact:
            shots+=r['arms'][0]['tracked'];pairs+=1
    report=dict(source={str(f):_hash(f) for f in (*sheets,optics.OUT,lateral.GEOMETRY,lateral.PLAN)},
                optical_equal_cases=pairs,optical_equal_estimates=shots,stages={},complete=False)
    with lateral.fixture.school.enabled(cameras=False):
        for f in sheets:
            doc=_read(f); parts={}
            if not doc['complete'] or doc['geometry']!=_hash(lateral.GEOMETRY):
                raise ValueError('Actual matched fixture docket incomplete or invalid')
            report['stages'][f.name]=parts
            for row in doc['rows']:
                cfg=row['dx']; sub=parts.setdefault(str(cfg),dict(rows=[],resimulated_samples=0))
                spec=row['case']; path=Path(spec['source']); historical=cache[path] if path in cache else cache.setdefault(path,_read(path))
                c,old,exp=q.get(historical,spec); motion=row['motion']
                x=dict(id=spec['id'],role=spec['role'],height_m=spec['height'],incoming_match=motion['stock_commands']['same'],
                       fixed_stock_command=q.received(motion['stock_commands']),same_body_coach=row['real'],interior=row['interior'],verified=[])
                successes=[('archived-rate',motion['stock_commands'])] if x['fixed_stock_command'] else []
                successes.extend(('coach',s) for s in motion['physical'] if s['landed'])
                # Unexpected unsafe or unanswered landings are independently
                # reproduced too, but never upgraded to a qualifying witness.
                if successes:
                    with lateral.slid(cfg),mpc.centered(dz[cfg]['ready']):
                        a=mpc.arena(width=64,height=48); cad=mpc.arena(width=64,height=48,detail='cad')
                        try:
                            guard=q.GripClearance(cad.model,stock=True)
                            for key,s in successes:
                                a0=q.tape.matched_motion(a,c,old,exp,s['commands'],guard)
                                if digest_json(a0['events'])!=digest_json(s['events']) or a0['clearance']!=s['clearance']:
                                    raise ValueError('Independently replayed fixture tape differs')
                                recount=audit(a0['events'],opening_side=1)
                                if recount['learner']!=1 or a0['answered']!=(recount['opponent']>=2):
                                    raise ValueError('Saved landed/answered labels are not native referee returns')
                                for j,target in enumerate(a0['placements']['requested']):
                                    expects=list(draw(c,j,'random',None)[0])
                                    if target['draw']!='random' or target['target']!=expects:
                                        raise ValueError('Actual replay did not use the continuously random reactive opponent')
                                samples=a0['clearance']['samples']; sub['resimulated_samples']+=samples
                                x['verified'].append(dict(arm=key,contact_radius_m=a0['contact']['radial_m'],
                                     contact_alignment=a0['contact']['face_alignment'],matched=a0['same'],
                                     events=digest_json(a0['events']),samples=samples,qualifies=q.received(a0),
                                     interior=q.received(a0,True)))
                        finally:a.close();cad.close()
                sub['rows'].append(x)
                _write(OUT,report)
                print(f.name,spec['id'],cfg,x['fixed_stock_command'],x['same_body_coach'],x['interior'],len(x['verified']),flush=True)
    report['complete']=True;_write(OUT,report)


if __name__=='__main__':
    verify()
