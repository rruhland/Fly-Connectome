"""Hold out independent same-law aims after bootstrapping an actual forecourt family."""
import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt54_midline_course as school
from . import tt54_midline_refine as teacher
from . import tt51_low_reverse_curriculum as optical
from .tt4_transfer_trial import _hash, _read
from .tt19_view_full_game import digest_json
from .tt37_opponent_audit import CourtClearance
from .tt39_carry_joints import Carried
from .tt44_paired_credit import State

ROOT=school.ROOT
OUT=ROOT/'family-transfer.json.gz'


def run(out):
    row,live,ready,old=school.teacher();p=_read(school.PLAN);c=row['card'];seeded=_read(school.PACKET)
    taught=_read(teacher.OUT);config=_read(teacher.PLAN)
    if not taught['complete'] or taught['plan_sha256']!=_hash(teacher.PLAN):
        raise ValueError('Physical learning must precede any held use')
    result=_read(out) if out.exists() else dict(physical_training_sha256=_hash(teacher.OUT),
           reflex_protocol_sha256=_hash(school.PLAN),roster_sha256=_hash(teacher.PLAN),
           n_preexisting=len(seeded['templates']),n_physically_rewarded=len(taught['templates']),rows=[],complete=False)
    if result['physical_training_sha256']!=_hash(teacher.OUT) or result['roster_sha256']!=_hash(teacher.PLAN):
        raise ValueError('Training boundary changed after held play')
    if len(taught['templates'])<2:
        result.update(complete=True,status='Not enough distinct physically learned coaches. Held seeds preserved.')
        optical.save(out,result);return
    alone=optical.RGBTaught(seeded['templates']);family=optical.RGBTaught(seeded['templates']+taught['templates'])
    with np.load(mpc.BANK) as bank:
        raw=mpc.DeepField(bank,'local')
    with mpc.centered(ready):
        a=mpc.arena();cad=mpc.arena(detail='cad');clear=CourtClearance(cad.model,articulated=True)
        resources=mpc.resources(a);camera,rig,*_=resources
        with patch.object(mpc,'Fragments',lambda b,e:Carried(b,e,practice=[school.course.PRACTICE],held={c['id']})):
            actor=school.Forecourt(mpc.BANK,raw,variant='elastic',budget=28,
                                   coverage=school.fresh.shot.Coverage(_read(school.fresh.RISK)))
        try:
            a.launch(live['shot'],card=c);a.partner.reset(seed=c['seed']);axes=a._arm_ids[1][2]
            eye=mpc.Vision(rig,a.control_dt,raw=True);eye.eye=mpc.LearnedOptics(eye.eye,school.course.OPTIC)
            actor.begin()
            for t,u in enumerate(live['actions'][:p['anchor']]):
                q=a.joints.copy();mpc.observed(eye,t,camera.capture() if t%2==0 and t>=p['anchor']-34 else None,q)
                actor.engine.observe(t,eye.eye,q,a.data.ctrl[axes].copy())
                a.exchange(u,mpc.FAR_COMMAND)
            if len(a.partner.requests)!=p['leg']:
                raise ValueError('The next opportunity has already been drawn')
            root=State(a);engine=school.body_snapshot(actor.engine)
            for i,h in enumerate(config['ordinary_nearby_independent']):
                if i<len(result['rows']):continue
                got=school.one(a,resources,actor,root,eye,engine,c,live,p,alone,h['seed'],h['target'],
                               'held_normal_nearby',i,clear,family=family)
                for variant in got['variants'].values():
                    if not variant['answered'] or not got['matched']:
                        continue
                    tape=dict(shot=live['shot'],actions=live['actions'][:p['cut']]+variant['controls'],
                              events=variant['events'],placements=variant['placements'])
                    variant['full_cad']=school.independent_verify(ready,c,h['seed'],tape,clear,p['anchor'])
                result['rows'].append(got)
                optical.save(out,result)
                print(i,h['seed'],[(k,v['landed'],v['answered'],sum(s['issued'] for s in v['cadence']))
                                   for k,v in got['variants'].items()],flush=True)
        finally:
            actor.close();camera.close();a.close();cad.close()
    result.update(complete=True,status='Four held TT37 nearby draws on the physically attained receiving body. No new full-game claim.')
    optical.save(out,result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=OUT)
    run(p.parse_args().out)
