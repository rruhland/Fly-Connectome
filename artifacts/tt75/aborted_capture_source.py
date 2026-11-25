"""Frozen plain-ball high-rate sensing on actual held joint-action tapes.

Native p/v/spin are evaluator-only, read after public estimates are formed.
No new controller is authorized by a state-error improvement alone.
"""
from pathlib import Path
from time import monotonic,process_time
import argparse
import numpy as np
import mujoco
from . import tt57_motor_school as school
from .tt37_optical_clinic import OpticalClinic
from .tt29_rally_physics import Rollout
from .tt75_retina import Retina
from .tt75_dense_fit import DenseFlight

ROOT=Path('artifacts/tt75')
PLAN=ROOT/'protocol.json'


def freeze():
    plan=school._read(school.PLAN);routine=school._read(school.ROOT/'routine.protocol.json')
    misses=sorted(plan['held'],key=lambda r:r['tick'])[:3]
    used={r['id'] for r in misses};ordinary=[r for r in sorted(routine['rows'],key=lambda r:r['tick']) if r['id'] not in used][:3]
    if len(misses)!=3 or len(ordinary)!=3:raise ValueError('Need fixed six release-disjoint held shots')
    rows=[dict(r,role='recorded_miss') for r in misses]+[dict(r,role='recorded_legal') for r in ordinary]
    with np.load('artifacts/tt37/hz60-practice.npz') as z:old_training=set(z['groups'][:,0].tolist())
    if old_training&{r['id'] for r in rows}:raise ValueError('Spin practice release in held sensor test')
    settings=school._read(Path('artifacts/tt39/spin-checkpoint.json'))['tuned']['coupled']
    source={p:school._hash(Path(p)) for p in {r['source'] for r in rows}}
    ROOT.mkdir(parents=True,exist_ok=True)
    school._write(PLAN,dict(schema='tt75_plain_retina_held_v1',rows=rows,sources=source,settings=settings,
        sensor_sha256=school._hash(Path(__file__).with_name('tt75_retina.py')),
        fit_sha256=school._hash(Path(__file__).with_name('tt75_dense_fit.py')),
        calibration_sha256=school._hash(ROOT/'sensor-calibration.json'),camera_rate=120,
        scope='Six existing release-disjoint held action tapes, stratified three old misses/three legal controls by shortest prefixes. Original ball appearance and uniform opponent unchanged. Native state only scores estimates and an explicit exact-state diagnostic. No new motor search or held policy promotion.',
        state_modes=['legacy','dense_joint'],recognition_modes=['legacy','dense_joint','exact_state_evaluator'],
        high_noise_m=[.0003,.0005,.0003],capture_seconds_before_fork=1.5,
        max_wall_seconds=900))


def prediction(engine,path,ref):
    _,y,feedback,loss,legal=school.native(engine,np.asarray([path]),ref)
    return dict(legal=bool(legal[0]),separation=bool(feedback[0,1]>.5),loss=float(loss[0]),
        nearest=float(feedback[0,0]),outgoing=feedback[0,2:5].tolist(),radius=float(feedback[0,8]))


def run(output,n):
    if output.exists():raise ValueError('Use a new output')
    p=school._read(PLAN)
    if p['sensor_sha256']!=school._hash(Path(__file__).with_name('tt75_retina.py')) or p['fit_sha256']!=school._hash(Path(__file__).with_name('tt75_dense_fit.py')):
        raise ValueError('Frozen sensing rule changed')
    doc=dict(schema='tt75_held_plain_rgb_state_v1',plan_sha256=school._hash(PLAN),rows=[],complete=False,
        source_sha256=school._hash(Path(__file__)),new_actor_actions=False)
    clock=monotonic();cpu=process_time();paid=0;deadline=clock+p['max_wall_seconds']
    def save(status=None):
        doc.update(actual_control_ticks=paid,wall_seconds=monotonic()-clock,cpu_seconds=process_time()-cpu)
        if status:doc.update(complete=True,status=status)
        school._write(output,doc)
    sources={}
    with school.enabled():
      for case in p['rows'][:n]:
        if monotonic()>=deadline:save('bounded_partial');return
        source=case['source']
        if source not in sources:
            if school._hash(Path(source))!=p['sources'][source]:raise ValueError('Changed issued source tape')
            sources[source]=school._read(Path(source))
        old=sources[source];card=old['cards'][case['i']];tape=old['rows'][case['i']];fork=case['tick']
        with school.mpc.centered(old['ready']):
            a=school.mpc.arena();wide,rig,stride,lo,hi=school.mpc.resources(a)
            retina=Retina(a,rig,wide);dense=DenseFlight(retina.period,p['settings'])
            eye=school.mpc.Vision(rig,a.control_dt,raw=True);eye.eye=school.mpc.LearnedOptics(eye.eye,school.course.OPTIC)
            clinic=OpticalClinic(body=True)
            try:
                a.launch(tape['shot'],card=card);a.partner.reset(seed=card['seed'])
                start=max(0,fork-round(1.5/a.control_dt));next_image=start*a.control_dt
                measures=[];fork_result={};capture_times=[];crop_errors=[]
                original_contacts=a._contacts
                def capture(now):
                    nonlocal next_image
                    point,_,_=retina.capture(now);dense.ingest(now,point);capture_times.append(now)
                    # Ground truth is saved only AFTER the RGB-only estimate.
                    if point is not None:crop_errors.append((now,(point-a.ball).tolist()))
                    next_image+=retina.period
                def contacts(incoming):
                    original_contacts(incoming)
                    now=float(a.data.time)
                    if now>=next_image-1e-8 and now>=start*a.control_dt and not (a.terminated or a.truncated):capture(now)
                a._contacts=contacts
                for tick in range(min(len(tape['actions']),fork+school.H+8)):
                    if monotonic()>=deadline:break
                    now=float(a.data.time);q=a.joints.copy();ref=a.data.ctrl[a._arm_ids[1][2]].copy()
                    if tick==start and not capture_times:capture(now)
                    frames=wide.capture() if tick>=start and tick%2==0 else None
                    x=school.mpc.observed(eye,tick,frames,q);clinic.observe(tick,eye.eye,q,ref)
                    newer=dense.fit(now) if tick>=start else None
                    if tick>=max(start,fork-30):
                        older=None if x is None else dict(pv=x.tolist(),spin=eye.eye.filter.rotation(now).tolist())
                        enhanced=None if newer is None else dict(pv=newer[0][0].tolist(),spin=newer[0][1].tolist(),
                            residual=newer[1].residual,posterior=newer[1].posterior.tolist(),bounce=newer[1].bounce,
                            exposure_age=now-dense.samples[-1][0])
                        truth=dict(pv=np.r_[a.ball,a.velocity].tolist(),spin=a.spin.tolist(),bounced=bool(a._received_bounce))
                        measures.append(dict(tick=tick,time=now,legacy=older,dense=enhanced,evaluator=truth))
                    if tick==fork:
                        commands=np.zeros((school.H,6));future=tape['actions'][fork:fork+school.H];commands[:len(future)]=future
                        path=school.atlas.reference_path(ref,commands,stride,lo,hi)[1:]-ref
                        if clinic.sync():
                            fork_result['legacy']=prediction(clinic.predict,path,ref)
                            if newer is not None:
                                twin=clinic.twin;pv,omega=newer[0]
                                twin.ball[:]=pv[:3];twin.velocity[:]=pv[3:];twin.data.qvel[twin._ball_dof+3:twin._ball_dof+6]=omega
                                twin._received_bounce=newer[1].bounce is not None
                                mujoco.mj_forward(twin.model,twin.data)
                                fork_result['dense_joint']=prediction(clinic.predict,path,ref)
                        fork_result['exact_state_evaluator']=prediction(Rollout(a,1),path,ref)
                    _,_,done,capped,_=a.exchange(tape['actions'][tick],school.mpc.FAR_COMMAND);paid+=1
                    if done or capped:break
                expected=[e for e in tape['events'] if e['time']<=float(a.data.time)+1e-8]
                exact=school.digest_json(a.events)==school.digest_json(expected)
                if not exact:raise ValueError('Additional camera observations changed the actual physics')
                actual=tape['placements']['delivered'][case['leg']]
                row=dict(case=case,measurements=measures,fork_recognition=fork_result,
                    original_returned=bool(actual['near_landed']),exact_recorded_physical_prefix=True,
                    actual_replayed_control_ticks=a.steps,camera_calls=retina.calls,wide_calls=retina.wide_calls,
                    valid_stereo=retina.valid,retina_cpu_seconds=retina.cpu,raw_center_errors_evaluator=crop_errors,
                    pixel_maps=[A.tolist() for A in retina.pixel_maps],last_physical_time=float(a.data.time))
                doc['rows'].append(row);save()
                common=[r for r in measures if r['legacy'] is not None and r['dense'] is not None and r['evaluator']['pv'][4]<-1]
                print(case['id'],case['role'],len(common),'camera',retina.valid,'/',retina.calls,
                    'recognition',{k:v['legal'] for k,v in fork_result.items()},flush=True)
            finally:clinic.twin.close();retina.close();wide.close();a.close()
    save('completed_fixed_held_actions')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('mode',choices=('plan','run'))
    parser.add_argument('--out',type=Path,default=ROOT/'held.json.gz');parser.add_argument('--n',type=int,default=6)
    args=parser.parse_args();freeze() if args.mode=='plan' else run(args.out,args.n)
