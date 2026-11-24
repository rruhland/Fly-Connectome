"""Replay scored-camera states and compare native rollout output bit for bit."""

import argparse
import ctypes
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time, perf_counter
from unittest.mock import patch

import mujoco
import numpy as np

from . import tt33_joint as joint, tt37_motor_clinic as clinic
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import _native_trials
from .tt31_ready import centered
from .tt34_latency import PUPIL
from .tt37_belief import ArrivalHistory
from .tt37_opponent import arena
from .tt37_optical_clinic import OpticalClinic
from .tt37_visual_inertia import LearnedOptics
from .tt38_motorcoach import setup


ROOT=Path(__file__).resolve().parents[2]
NATIVE=Path(__file__).with_name('tt29_rally_rollout.cpp')
ARCHIVE=ROOT/'artifacts/tt37/value-closed.json.gz'


def build(path, folder):
    package=Path(mujoco.__file__).parent
    lib=next(package.glob('libmujoco.so.*'))
    target=Path(folder)/(path.stem+'.so')
    subprocess.run(['c++','-std=c++17','-O3','-ffp-contract=off','-shared','-fPIC','-fopenmp','-I',str(package/'include'),
                    str(path),str(lib),f'-Wl,-rpath,{package}','-o',str(target)],check=True)
    call=ctypes.CDLL(str(target)).trial_strokes_v5
    call.argtypes=_native_trials()[0].argtypes
    call.restype=None
    return call


def invoke(engine, call, goals, eta, observe):
    old=engine.predict.trial
    engine.predict.trial=call
    try:
        n,k=goals.shape[:2]
        if observe:
            feedback=np.zeros((n,9)); trace=np.full((n,k,17),np.nan); approaches=np.full((n,19),np.nan)
        else:
            feedback=trace=approaches=None
        rates,scores=engine.predict.trajectories(goals,clinic.AIM,eta+.12,feedback=feedback,traces=trace,approaches=approaches)
        return [x for x in (rates,scores,feedback,trace,approaches) if x is not None]
    finally:
        engine.predict.trial=old


def assert_bits(a,b):
    for i,(v,w) in enumerate(zip(a,b,strict=True)):
        if v.shape!=w.shape or not np.array_equal(v.view(np.uint64),w.view(np.uint64)):
            delta=np.count_nonzero(v.view(np.uint64)!=w.view(np.uint64))
            raise ValueError(f'Native output {i} changed in {delta} scalars')


def bench(reference, candidate, output, repeats, *, profile=False):
    doc=_read(ARCHIVE)
    rows=doc['rows']; c=doc['cards'][0]; row=rows[0]
    ticks=(row['records'][0]['tick'],row['records'][-1]['tick'])
    home=np.asarray(doc['ready']); s=setup(doc['depth']); joint.RECEIVE_Y=s.depth
    variants=[('short',9,0.,False),('normal',25,0.,True),('full',72,0.,True),('positive_margin',25,.0005,True)]
    model=_read(PUPIL); data=[]
    with TemporaryDirectory(prefix='tt40-native-rollout-') as folder, centered(home):
        original=build(reference,folder)
        optimized=build(candidate,folder)
        a=arena(width=384,height=288); engine=OpticalClinic(body=True)
        render,rig,stride,lo,hi=joint.resources(a)
        try:
            a.launch(row['shot'],card=c); a.partner.reset(seed=c['seed'])
            actor=joint.JointPolicy(model|{'ready':home.tolist()},stride,lo,hi)
            vision=joint.Vision(rig,a.control_dt); vision.eye=LearnedOptics(vision.eye,doc['visual'])
            raw=ArrivalHistory()
            for tick,u in enumerate(row['actions']):
                q=a.joints.copy()
                opt=raw.update(tick*a.control_dt,vision.update(tick,render.capture() if tick%2==0 else None,q))
                engine.observe(tick,vision.eye,q,actor.reference.copy())
                if tick in ticks:
                    entry=next(r for r in row['records'] if r['tick']==tick)
                    if opt is None or not np.allclose(entry['context'],np.r_[opt[1:6],q[:5]],atol=1e-10):
                        raise ValueError('Archived optical input differs at the measured decision')
                    physical_prefix=[e for e in row['events'] if e['time']<=a.data.time+1e-8]
                    if digest_json(a.events)!=digest_json(physical_prefix):
                        raise ValueError('Archived physical context changed')
                    if not engine.sync():
                        raise ValueError('No sensor-initialized model at measured decision')
                    eta=opt[0]; fit=clinic.geometry(s,opt)
                    for label,n,margin,observe in (variants[1:2] if profile else variants):
                        engine.twin.clearance_margin=margin
                        # Preserve an actual old proposal. Add independent deterministic
                        # proposals including misses and clipped actuator commands.
                        options=clinic.generate(n-1)
                        options[0]=np.asarray(entry['action'])
                        goals=clinic.schedules(*fit,eta,options,a.control_dt,home,round((eta+1.05)/a.control_dt))
                        a0=invoke(engine,original,goals,eta,observe)
                        b0=invoke(engine,optimized,goals,eta,observe)
                        assert_bits(a0,b0)
                        logs={k:[] for k in ('reference_cpu','candidate_cpu','reference_wall','candidate_wall')}
                        for j in range(repeats):
                            order=[('reference',original),('candidate',optimized)]
                            if j%2:order.reverse()
                            for name,call in order:
                                t=process_time(); wall=perf_counter()
                                result=invoke(engine,call,goals,eta,observe)
                                logs[name+'_cpu'].append(process_time()-t)
                                logs[name+'_wall'].append(perf_counter()-wall)
                                assert_bits(a0,result)
                        legal=np.isfinite(a0[1][:,0]); counts=np.mean(np.max(abs(a0[0]),axis=-1)>.7999,axis=1)
                        report=dict(tick=tick,eta=float(eta),label=label,margin=margin,n=n,frames=goals.shape[1],
                            explicit_feedback=observe,model_geoms=engine.twin.model.ngeom,pairs=len(engine.predict.first),
                            legal=int(legal.sum()),failed=int((~legal).sum()),maximum_saturation_fraction=float(counts.max()),
                            identical_bits=True,**logs)
                        data.append(report)
                        print(tick,label,report['legal'],report['failed'],
                              round(np.median(logs['reference_cpu']),5),round(np.median(logs['candidate_cpu']),5),flush=True)
                a.exchange(u,clinic.FAR); actor.reference=np.clip(actor.reference+stride*np.asarray(u),lo,hi)
                if tick==max(ticks):break
        finally:
            render.close(); engine.twin.close(); a.close()
    _write(output,dict(reference_sha256=_hash(reference),candidate_sha256=_hash(candidate),archive_sha256=_hash(ARCHIVE),
                        repeat_count=repeats,data=data))


def physical(reference,candidate,output,n):
    from . import tt29_rally_physics as native
    from .tt38_impact_margin import games,compare
    prior=_read(ARCHIVE)
    with TemporaryDirectory(prefix='tt40-rollout-physical-') as folder:
        cases=[]
        raw=[]
        for label,code in [('stock',reference),('bounds',candidate)]:
            call=build(code,folder)
            path=Path(folder)/f'{label}.json.gz'
            with patch.object(native,'_native_trials',lambda: (call,folder)):
                games(ARCHIVE,path,'stock',count=n)
            compare(ARCHIVE,path)
            raw.append(_read(path))
        for c,old,new,witness in zip(raw[0]['cards'],raw[0]['rows'],raw[1]['rows'],prior['rows'][:n],strict=True):
            for k in ('actions','events','placements','score','records'):
                if digest_json(old[k])!=digest_json(new[k]):
                    raise ValueError(f'Physical comparison failed on {c["id"]}, {k}')
            cases.append(dict(id=c['id'],common_own=old['score']['learner'],common_total=old['score']['returns'],
                reference_cpu=old['cpu_s'],candidate_cpu=new['cpu_s'],reference_arena_cpu=old['arena_cpu_s'],
                candidate_arena_cpu=new['arena_cpu_s'],reference_perception_cpu=old['perception_cpu_s'],
                candidate_perception_cpu=new['perception_cpu_s'],moves=len(old['records']),
                actions_sha256=digest_json(old['actions']),events_sha256=digest_json(witness['events'])))
    _write(output,dict(reference_sha256=_hash(reference),candidate_sha256=_hash(candidate),archive_sha256=_hash(ARCHIVE),paired=cases))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reference',type=Path)
    p.add_argument('output',type=Path)
    p.add_argument('--candidate',type=Path,default=NATIVE)
    p.add_argument('--repeats',type=int,default=6)
    p.add_argument('--profile',action='store_true')
    p.add_argument('--physical',type=int,metavar='CASES')
    a=p.parse_args()
    if a.physical:
        physical(a.reference,a.candidate,a.output,a.physical)
    else:
        bench(a.reference,a.candidate,a.output,a.repeats,profile=a.profile)
