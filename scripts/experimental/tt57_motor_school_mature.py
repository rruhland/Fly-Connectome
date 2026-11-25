"""Resolve safe earned practice actions whose ordinary response was still in flight."""

import numpy as np

from . import tt57_motor_school as school
from .tt4_transfer_trial import _hash, _read, _write

OUT=school.ROOT/'matured.json.gz'


def run():
    data={}; doc=_read(school.OUT); digest=_hash(school.OUT)
    if not doc['complete']:
        raise ValueError('Complete the original practice docket before resolving censoring')
    doc.update(original_practice_sha256=digest,extensions=[])
    with school.enabled():
        cad=school.mpc.arena(width=64,height=48,detail='cad')
        try:
            guard=school.CourtClearance(cad.model,articulated=True)
            for r in doc['rows']:
                todo=[(j,v) for j,v in enumerate(r.get('arms',[]))
                      if v.get('issued_relative') and not v['answered'] and v['events'][-1]['kind']!='point']
                if not todo:continue
                spec=r['case']; src=spec['source']; c=spec['card']
                if src not in data:data[src]=_read(school.Path(src))
                stock=data[src]['rows'][spec['i']]
                with school.mpc.centered(spec['ready']):
                    a=school.mpc.arena(width=64,height=48)
                    try:
                        school.prefix(a,c,stock,spec['tick'],None,a._arm_ids[1][2],window=0)
                        saved=school.State(a)
                        for j,v in todo:
                            old_event=v['event_sha256']
                            cmd=np.vstack((v['controls'],np.zeros((68,6))))
                            extended=school.physical(a,saved,c,stock,spec,cmd,guard)
                            if (school.digest_json(v['events'])!=school.digest_json(extended['events'][:len(v['events'])])
                                or not extended['landed']):
                                raise ValueError('Censoring follow-up did not preserve original scored prefix')
                            v.update(extended,initial_censored_event_sha256=old_event)
                            if v['clearance']['penetration_samples']:
                                v.pop('issued_relative')
                            doc['extensions'].append(dict(id=spec['id'],lane=spec['lane'],choice=j,
                                answered=extended['answered'],terminal=extended['events'][-1].get('reason'),
                                events_sha256=extended['event_sha256']))
                            print('matured',doc['extensions'][-1],flush=True)
                    finally:
                        a.close()
        finally:
            cad.close()
    _write(OUT,doc)


if __name__=='__main__':run()
