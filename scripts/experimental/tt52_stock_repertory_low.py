"""Check the installed stock-window options on matched genuine low arrivals from an attained body."""

import argparse
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np

from . import tt52_stock_repertory as rp
from .tt50_optical_contact_school import inputs
from .tt37_opponent_audit import CourtClearance,guarded
from .tt39_opportunity_map import projected_height


def run(output,limit):
    prior=rp.pf.prior;mpc=rp.mpc
    source=rp._read(prior.ARCHIVE);case,archive=prior.terminal(source)
    archive_plan=rp._read(prior.PLAN)
    _,tapes=inputs()
    # The leftward redraw used its original continuous opponent draw. It was
    # withheld from both selected teaching templates and is not target-tuned.
    tasks=[(0,case['card']['reply_seed'],'hard teaching source'),(1,tapes[1]['stream'],'separate teaching source'),
           (0,515671963,'unused-for-motors native left low')][:limit]
    model=rp.low_field(rp.low_artifact)
    doc=dict(schema='tt52-stock-window-on-attained-body-1',policy_sha256=rp._hash(Path(rp.__file__)),
             low_templates_sha256=prior.digest_json(rp._read(rp.low_artifact)['initial_templates']),rows=[],
             scope='Each attained-body practice prefix comes from an actual same-fixture teacher. Every later command is chosen by the same stock search, with or without learned RGB and encoder options. No full-point qualification.')
    with np.load(mpc.BANK) as z:raw=mpc.DeepField(z,'local')
    boot=source['ready']
    with mpc.centered(boot):
        a=mpc.arena();cad=mpc.arena(detail='cad');guard=CourtClearance(cad.model,articulated=True)
        camera,rig,stride,lo,hi=mpc.resources(a);axes=a._arm_ids[1][2]
        with patch.object(mpc,'Fragments',lambda f,e:prior.Carried(f,e,practice=[prior.course.PRACTICE],held={case['id']})):
            player=rp.Arm(mpc.BANK,raw,mode='stock',finish=rp.pf.Finish(),variant=source['variant'],
                          budget=source['budget'],coverage=prior.fresh.shot.Coverage(rp._read(prior.fresh.RISK)))
        try:
            ancestral={**source,'rows':[None]*len(rp._read(prior.season.CARRY)['rows'])+source['rows']}
            prior.course.resume(player,prior.BANK,ancestral)
            start=archive_plan['anchor']
            history=prior.prefix(a,case['card'],archive,start,camera,axes,window=32)
            anchor=prior.State(a);warm=prior.Observer(a,rig,player,history)
            for which,stream,label in tasks:
                cut=tapes[which]['cut']; anchor.restore(a)
                c=dict(case['card'],reply_seed=stream);a.card=c
                tape=tapes[which]['tape']['actions'];eye=warm.restore(player)
                for t in range(start,cut):
                    q=a.joints.copy();ref=a.data.ctrl[axes].copy()
                    frames=camera.capture() if t%2==0 else None
                    x=mpc.observed(eye,t,frames,q)
                    player.observe_only(t,frames,eye.eye,x,q,ref)
                    a.exchange(tape[t],mpc.FAR_COMMAND)
                saved=prior.State(a);screen=deepcopy(eye)
                e=player.engine; state=mujoco.MjData(e.body_model)
                mujoco.mj_copyData(state,e.body_model,e.body_state)
                values={k:deepcopy(getattr(e,k)) for k in ('previous','rate','tick','reference','bounced')}
                row=dict(stream=stream,cut=cut,role=label,native_prefix=prior.digest_json(a.events),
                         q=a.joints.tolist(),ref=a.data.ctrl[axes].tolist(),variants={})
                initial_degree=np.array(row['ref'])
                for enabled in (False,True):
                    saved.restore(a);guarded(a,guard);player.begin()
                    player.repertory=rp.Joined(None,model) if enabled else None
                    eye=deepcopy(screen)
                    mujoco.mj_copyData(e.body_state,e.body_model,state)
                    for k,v in values.items():setattr(e,k,deepcopy(v))
                    e.tracker=eye.eye;e.latest=eye.eye.latest
                    tape_out=[];optics=[]
                    opened=False
                    for tick in range(cut,cut+150):
                        q=a.joints.copy();ref=a.data.ctrl[axes].copy()
                        frames=camera.capture() if tick%2==0 else None
                        x=mpc.observed(eye,tick,frames,q)
                        # For these physically achieved starting states both
                        # branches retain that preparation until the same public
                        # stock eligibility condition, then choose ordinary goals.
                        opened=opened or (x is not None and x[4]<-.9 and -.49>x[1]>-1.62)
                        if opened:
                            g=player.choose(tick,frames,eye.eye,x,q,ref,player.priors.home)
                        else:
                            player.observe_only(tick,frames,eye.eye,x,q,ref)
                            g=initial_degree
                        u=np.clip((np.clip(g,lo,hi)-ref)/stride,-.8,.8).tolist()
                        tape_out.append(u)
                        if x is not None:optics.append([tick,*x,*q,*ref])
                        _,_,done,capped,_=a.exchange(u,mpc.FAR_COMMAND)
                        if done or capped or a.rival_returns>=44:break
                    ps=a.point_placements();offer=ps['delivered'][case['leg']]
                    if ps['requested'][case['leg']]['target']!=list(rp.draw(c,case['leg'],'random',None)[0]):
                        raise ValueError('Native broad draw changed')
                    row['variants']['stock-options' if enabled else 'prepared-stock']=dict(
                        offer=offer,legal=bool(offer['near_landed']),answered=a.rival_returns>=44,
                        commands=tape_out,sensory=optics,opts=player.repertory_log,stock_notes=player.notes,
                        clearance=a.clearance_result(),event_sha256=prior.digest_json(a.events))
                before,after=row['variants'].values()
                row['same_actual_offer']=all(prior.same_trace(before['offer'][k],after['offer'][k]) for k in ('far_contact','near_bounce'))
                if not row['same_actual_offer']:raise ValueError('Each motor choice must receive the same actual far ball')
                row['height']=projected_height(before['offer']['receive_line'])
                doc['rows'].append(row);rp._write(output,doc)
                print(stream,row['height'],[(k,v['legal'],v['answered'],sum(t['accepted'] for t in v['opts']))
                       for k,v in row['variants'].items()],flush=True)
        finally:player.close();camera.close();a.close();cad.close()
    doc['complete']=True;rp._write(output,doc)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=rp.root/'low.json.gz')
    p.add_argument('--limit',type=int,default=3)
    a=p.parse_args();run(a.out,a.limit)
