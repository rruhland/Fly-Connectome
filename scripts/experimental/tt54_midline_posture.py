"""Earlier equally physical core arm solutions on two central low practice routes."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from . import tt38_joint_mpc as mpc
from . import tt40_long_low as long
from . import tt40_low_feasibility as native
from . import tt54_midline as midline
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import Rollout
from .tt37_opponent import draw
from .tt37_opponent_audit import CourtClearance, contact_observer, guarded

ROOT=midline.ROOT
PLAN=ROOT/'posture-protocol.json'
RESULT=ROOT/'posture.json.gz'
IDS=(4214,4309)
DEPTHS=(-1.08,-1.18,-1.29,-1.39,-1.47,-1.55,-1.63,-1.71,-1.79,-1.87,-1.94)


def plan():
    source=_read(midline.PLAN)
    keep=[r for r in source['rows'] if r['id'] in IDS]
    if [r['id'] for r in keep]!=list(IDS):
        raise ValueError('Changed early preparation roster')
    _write(PLAN,dict(midline_protocol_sha256=_hash(midline.PLAN),cases=keep,depths=DEPTHS,max_routes=4,
        trials_per_route=96,rounds_per_route=2,allow_evaluator_to_depth=-1.97,
        role='The middle two preassigned same-release central-low failures, requiring a previous own scored arm strike to preserve before any preparation difference. Before the next opponent strikes, a privileged diagnostic chooses among both arm-facing axes, several depths and continuous joint routes with local CAD clearance. The actual far stroke, target, bounce, solver, motor envelope, signed rates and clearance criteria do not change. This branch demonstrates finite physical feasibility, not currently available early lookahead or a deployed policy. The native shadow toy is allowed to follow incoming ball flight past its usual -1.82 m cut only for generating candidates.'))


def run(out):
    config=_read(PLAN)
    if config['midline_protocol_sha256']!=_hash(midline.PLAN):
        raise ValueError('The fixed central practice set changed')
    results=_read(out) if out.exists() else dict(plan_sha256=_hash(PLAN),rows=[],complete=False)
    if results['plan_sha256']!=_hash(PLAN):
        raise ValueError('Different earlier posture diagnostic')
    with TemporaryDirectory(prefix='tt54-midline-native-') as tmp:
        cpp=Path(native.__file__).with_name('tt29_rally_rollout.cpp')
        body=cpp.read_text(); old='y < -1.82'
        if body.count(old)!=1:
            raise ValueError('Changed near-native time cutoff definition')
        source=Path(tmp)/'midline_rollout.cpp'
        # Only the isolated candidate evaluator goes deeper. No environment
        # cutoff, speed, motor, ball, opponent or court is changed in replay.
        source.write_text(body.replace(old,f'y < {config["allow_evaluator_to_depth"]}'))
        trial=long.build(source,Path(tmp))
        results['stock_shadow_sha256']=_hash(cpp)
        results['expanded_shadow_sha256']=_hash(source)
        for r in config['cases'][len(results['rows']):]:
            path=Path(r['source']);doc=_read(path)
            if _hash(path)!=r['source_sha256']:
                raise ValueError('Actual archived motor commands changed')
            c,old=doc['cards'][r['index']],doc['rows'][r['index']]
            ready=doc['ready']; leg=r['leg']; offer=old['placements']['delivered'][leg]
            with mpc.centered(ready):
                a=mpc.arena(width=64,height=48)
                cad=mpc.arena(width=64,height=48,detail='cad')
                g=CourtClearance(cad.model,articulated=True)
                try:
                    stage(a,g,c,old,leg,offer,r,ready,trial,config,results,out)
                finally:
                    a.close();cad.close()
    results['complete']=True
    _write(out,results)


def stage(a,g,c,old,leg,offer,r,ready,trial,config,results,out):
    midline.replay_control(a,old,c)
    fork,late,table=long.indexed(old['events'],offer,a.control_dt)
    arch=[e for e in old['events'] if e['time']<=late*a.control_dt+1e-8]
    stamp=native.replay_prefix(a,old,c,fork)
    ref=a.data.ctrl[a._arm_ids[1][2]][:5].copy()
    workspace=long.Workspace(a,np.asarray(ready))
    with patch.object(long,'DEPTHS',tuple(config['depths'])):
        trace=workspace.windows(table,defensive=False)
    possible,bad=workspace.support(trace,ref,(late-fork)*a.control_dt)
    for z in possible:
        z['eta']=float(z['time']+table['time']-late*a.control_dt)
    candidates=long.select(possible,config['max_routes'])
    engine=Rollout(a,1); engine.trial=trial
    note=dict(case=r,card=c,fork_tick=fork,receiving_tick=late,original_prefix_sha256=stamp,
              arm_ref=ref.tolist(),window_s=(late-fork)*a.control_dt,
              supported=[long.plain(q) for q in possible],rejected_count=len(bad),routes=[],win=None)
    print(r['id'],'routes',[(q['face'],float(q['p'][1])) for q in candidates],flush=True)
    for i,q in enumerate(candidates):
        prefix,same,_=long.physically_prepare(a,old,c,fork,late,q,arch,brisk_flip=True)
        reported=dict(route=i,prepared=long.plain(q),phase_prefix_match=same,posture=a.joints.tolist(),model=[],physical=None)
        note['routes'].append(reported)
        if not same:
            reported['mismatch']=long.event_difference(a.events,arch)
            continue
        try:
            a.clearance_margin=.0006
            plans,model=long.late_search(a,engine,q,possible,config['trials_per_route'],
                                          config['rounds_per_route'],[540728,c['seed'],i])
        finally:
            a.clearance_margin=0.
        reported['model']=model
        if not plans:
            continue
        began=float(a.data.time);guared=[]
        guarded(a,g)
        try:
            with contact_observer(a,guared):
                live=long.physical_finish(a,old,c,leg,prefix,plans[0])
            clearance=a.clearance_result()
            if clearance['samples']!=round((a.data.time-began)/a.model.opt.timestep):
                raise ValueError('Missed a physical suffix CAD state')
            if any(value['target']!=list(draw(c,j,'random',None)[0])
                   for j,value in enumerate(live['placements']['requested'])):
                raise ValueError('Unmodified broad target stream diverged')
            live.update(suffix_cad=clearance,suffix_contacts=guared,cut_tick=fork)
        finally:
            a.sample_clearance(False)
        reported['physical']=live
        if live['answered'] and clearance['penetration_samples']==0:
            live['full_cad']=midline.full_verify(ready,c,live,g)
            if live['full_cad']['penetration_samples']==0:
                note['win']=i
                break
    results['rows'].append(note)
    _write(out,results)
    print(r['id'],'physical',[(z['physical']['legal'],z['physical']['answered']) if z['physical'] else None for z in note['routes']],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('plan','run'))
    p.add_argument('--out',type=Path,default=RESULT)
    a=p.parse_args();plan() if a.mode=='plan' else run(a.out)
