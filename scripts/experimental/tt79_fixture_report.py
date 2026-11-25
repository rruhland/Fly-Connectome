"""Independently measure the actual patch velocity on the TT79 physical trials."""

from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt79_impulse_fixture as assay
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt37_opponent import draw


OUT=assay.ROOT/'native-contact-audit.json'


def run():
    p=assay.plan(); tapes=_read(assay.OUT); MP=assay.MP; tr=assay.tr
    if not tapes['complete'] or _hash(assay.MAP)!=tapes['kinematics']:
        raise ValueError('Complete whole declared fixture comparison')
    result=dict(physical_sha256=_hash(assay.OUT),physical_operator_sha256=_hash(Path(assay.__file__)),
                protocol_sha256=_hash(assay.PLAN),rows=[],complete=False)
    cache={}
    with assay.fixture.school.enabled(cameras=False):
        for d in tapes['rows']:
            f=d['case']['source']; source=cache[f] if f in cache else cache.setdefault(f,_read(Path(f)))
            c,row,intake=tr.qual.get(source,d['case']); r=dict(name=d['name'],id=c['id'],v=[])
            if d['physical']:
                with assay.mounted(d['name']),MP.centered(d['ready']):
                    a=MP.arena(width=64,height=48); cad=MP.arena(width=64,height=48,detail='cad')
                    try:
                        guard=tr.qual.GripClearance(cad.model,stock='seat' not in assay.CONFIG[d['name']])
                        for v in d['physical']:
                            checks=[]; base=a._racket_contact
                            def observe(side,location,incoming):
                                if side==1:
                                    b=tr.blade(a); T=np.asarray(b['twist']); pos=np.asarray(b['p']); close=np.asarray(location)-pos
                                    linear=T[3:]+np.cross(T[:3],close); n=np.asarray(b['R'])[:,2]
                                    if n@(a.ball-pos)<0.:n=-n
                                    checks.append(dict(time_s=b['time'],patch_mps=linear.tolist(),normal=n.tolist(),
                                                       local_patch_normal_mps=float(n@linear),incoming=np.asarray(incoming).tolist(),
                                                       disk_center=b['p'],disk_angular=T[:3].tolist(),encoder=b['q'],rate=b['qd']))
                                return base(side,location,incoming)
                            with patch.object(a,'_racket_contact',observe):
                                got=tr.tape.matched_motion(a,c,row,intake,v['commands'],guard)
                            if digest_json(got['events'])!=digest_json(v['events']) or got['clearance']!=v['clearance']:
                                raise ValueError('Independently sampled physical tape differs')
                            if any(i['draw']!='random' or i['target']!=list(draw(c,k,'random',None)[0])
                                   for k,i in enumerate(got['placements']['requested'])):
                                raise ValueError('Opponent trajectory law changed on recheck')
                            contact=got['contact']; hit=next((s for s in checks if contact is not None and
                                                             abs(s['time_s']-contact['time_s'])<1e-8),None)
                            if contact is not None and hit is None:
                                raise ValueError('Actual contact missing its twist observation')
                            r['v'].append(dict(actual_patch=hit,contact=contact,depart=got['depart'],scored=got['landed'],
                                               answered=got['answered'],same_incoming=bool(got['same']),
                                               clearance=got['clearance'],soft_answer=assay.imp.actual_soft(got),
                                               replay=digest_json(got['events'])))
                    finally:a.close();cad.close()
            result['rows'].append(r);_write(OUT,result)
            print(d['name'],c['id'],[(v['scored'],v['answered'],None if not v['actual_patch'] else
                  [round(x,3) for x in v['actual_patch']['patch_mps'][1:]]) for v in r['v']],flush=True)
    result['complete']=True;_write(OUT,result)


if __name__=='__main__':
    run()
