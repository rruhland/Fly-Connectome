"""Use signed camera evidence to retain a reach and refine a late paddle adjustment."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time
from unittest.mock import patch

import numpy as np

from . import tt33_joint as joint, tt36_impact as impact, tt36_local as local
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import measured_launches
from .tt35_replies import varied_opponent
from .tt36_inverse import target
from .tt36_staged import body_model, movement
from .tt4_transfer_trial import _hash, _read, _write


IDS = (6253, 6861)
EYES = ('left', 'right')
OUT = impact.OUTPUT
ZERO = np.zeros(5)
WEIGHT = np.array([.8, 1.25, 1.])


class Frames:
    """Keep accepted RGB stereo points, individual sightings and encoder readouts."""

    def __init__(self, rig):
        self.rig = rig
        self.stereo, self.mono, self.blade, self.encoders = [], [], [], []

    def observe(self, frame, time, q):
        p, blade = impact.rgb_point(frame, self.rig)
        if p is not None:
            self.stereo.append((time, p, blade))
            if blade is not None:
                self.blade.append((time, blade))
        else:
            sightings = [impact.native_ball_pixels(frame[name]) for name in EYES]
            for k, seen in enumerate(sightings):
                if seen is not None and not seen.clipped:
                    camera = self.rig.fixed[EYES[k]]
                    direction = np.r_[(seen.center-camera.center)/camera.focal,1.] @ camera.rotation
                    direction /= np.linalg.norm(direction)
                    self.mono.append((time, k, camera.eye, np.eye(3)-np.outer(direction,direction)))
            if any(s is not None for s in sightings):
                b = [impact.blade_sighting(frame[k]) for k in EYES]
                if all(x is not None for x in b):
                    pose = impact.triangulate([x[0]*frame[k].shape[0]/360 for x,k in zip(b,EYES)],self.rig)
                    if pose is not None and -1.75 < pose[1] < -.77 and .75 < pose[2] < 1.40 and abs(pose[0]) < .64:
                        self.blade.append((time, pose))
        self.encoders.append((time,np.asarray(q[:5]).copy()))

    def estimates(self, calibration):
        stereo = [dict(e, estimator='stereo-delta') for e in impact.estimate_impacts(self.stereo,calibration)]
        # This fallback is tested here on held-out stereo-invisible gaps, not counted in the stereo benchmark.
        recovered = recover_gaps(self, calibration)
        return sorted(stereo + [e for e in recovered if all(abs(e['time']-s['time']) > .14 for s in stereo)],
                      key=lambda e:e['time'])


def recover_gaps(frames, model):
    """Fit one continuous 3D bend through a unilateral gap, using each visible camera ray."""
    coeff = np.array(model['acceleration'])
    stereo = frames.stereo
    for i in range(4,len(stereo)-2):
        before, after = stereo[i-1][0], stereo[i][0]
        if not .115 < after-before < .32 or stereo[i-1][1][1] >= -.80:
            continue
        pre = stereo[i-4:i]
        rest = [(t,p) for t,p,_ in stereo[i:i+4] if t-after < .056 and p[2]>.795]
        if len(rest) < 2 or before-pre[0][0]>.071 or stereo[i][1][2] < .80:
            continue
        sight = [(t,k,o,m) for t,k,o,m in frames.mono if before < t < after]
        if len(sight) < 7 or max(sum(s[1]==k for s in sight) for k in (0,1)) < .86 * len(sight):
            continue
        _, start_v = impact.tangent(pre,before,np.zeros(3))
        a = impact.acceleration_basis(start_v) @ coeff
        p_ref, v_ref = impact.tangent(pre,before,a)
        if v_ref[1] > -.65:
            continue
        old = lambda t: p_ref + (t-before)*v_ref + .5*(t-before)**2*a
        pre_error = [np.linalg.norm(p-old(t)) for t,p,_ in pre]
        mono_null = [np.linalg.norm(m@(old(t)-o)) for t,_,o,m in sight]
        stereo_null = [np.linalg.norm(old(t)-p) for t,p in rest]
        if np.median(pre_error)>.016 or np.mean(stereo_null)<.13:
            continue
        possibilities=[]
        for t in np.linspace(before+.025,after-.035,46):
            earlier=[(s,e) for s,e in zip(sight,mono_null,strict=True) if s[0]<t]
            later=[s for s in sight if s[0]>t]
            if len(earlier)<2 or len(later)<4 or np.mean([e for _,e in earlier])>.028:
                continue
            p = old(t)
            places=[(u,b) for u,b in frames.blade if abs(u-t)<.049]
            if not places:
                continue
            u, near = min(places,key=lambda x:abs(x[0]-t))
            gap=np.linalg.norm(p-near)
            if gap>.175 or not (-1.67 < p[1] < -.84 and .835 < p[2] < 1.36):
                continue
            v = v_ref+(t-before)*a
            w=(rest[0][1]-p)/(rest[0][0]-t)
            for _ in range(3):
                aa = impact.acceleration_basis(w)@coeff
                design = [(stamp-t)*m for stamp,_,_,m in later] + [(stamp-t)*np.eye(3) for stamp,_ in rest]
                rhs = [m@(o-p-.5*(stamp-t)**2*aa) for stamp,_,o,m in later]
                rhs += [r-p-.5*(stamp-t)**2*aa for stamp,r in rest]
                w = np.linalg.lstsq(np.concatenate(design),np.concatenate(rhs),rcond=None)[0]
            predict = lambda stamp: p+(stamp-t)*w+.5*(stamp-t)**2*aa
            errors = [np.linalg.norm(m@(predict(stamp)-o)) for stamp,_,o,m in later]
            errors += [np.linalg.norm(predict(stamp)-r) for stamp,r in rest]
            mse = float(np.mean(np.square(errors + [e for _,e in earlier])))
            norm = np.linalg.norm(w-v)
            if norm<1. or np.linalg.norm(w)>10. or np.sqrt(mse)>.024:
                continue
            if np.mean(errors)>.32*np.mean(mono_null+stereo_null):
                continue
            possibilities.append((mse+.002*gap**2,dict(estimator='stereo-anchored-mono',time=float(t),
                available=float(max(s[0] for s in rest)),incoming=v.tolist(),outgoing=w.tolist(),center=p.tolist(),
                delta_v_mps=float(norm),gap_m=float(gap),stereo_post=len(rest),mono_frames=len(sight),
                fit_rmse_m=float(np.sqrt(mse)))))
        if possibilities:
            yield min(possibilities,key=lambda x:x[0])[1]


def rank(e, goal):
    observation = e['first']
    error = np.linalg.norm(WEIGHT*(np.array(observation['outgoing'])-goal)) if observation is not None else float('inf')
    return e['reward'], observation is not None, -error, -np.linalg.norm(e['late_rad'])


def first_alarm(frames, fit, onset):
    alarms = [e for e in frames.estimates(fit) if onset+.17 <= e['time'] <= onset+.95]
    return alarms[0] if alarms else None


def choose_from(states, field):
    centers = np.asarray([e['context'] for e in states])
    actions = np.asarray([e[field] for e in states])

    def choose(x):
        ds = np.linalg.norm((centers-x)/local.SCALE,axis=1)
        i = int(np.argmin(ds))
        return actions[i]*np.exp(-.5*(ds[i]/1.25)**2) if ds[i]<2.35 else ZERO.copy()
    return choose


def unit(v):
    return v/max(1e-9,np.linalg.norm(v))


def propose(history, goal, jac, body_j, index):
    """Small orienting probes and local secants around a camera-confirmed reach."""
    seed = history[0]
    v = np.array(seed['first']['outgoing'])
    drift = WEIGHT*(goal-v)
    inertia = .34*np.eye(5) + 5*body_j.T@body_j
    j = WEIGHT[:,None]*jac
    follow = unit(np.linalg.solve(j.T@j+inertia,j.T@drift))
    roll = np.array([0.,0.,0.,0.,1.])
    rotate = unit(roll-(roll@follow)*follow)
    pitch = np.array([0.,0.,0.,1.,0.])
    lean = unit(pitch-(pitch@follow)*follow-(pitch@rotate)*rotate)
    if index<5:
        fs = [(.25,follow),(-.25,follow),(.31,rotate),(-.31,rotate),(.29,lean)]
        length, direction = fs[index]
        return length * direction
    valid = [e for e in history if e['first'] is not None]
    best = max(valid,key=lambda e:rank(e,goal))
    center = np.array(best['late_rad'])
    vec = np.array(best['first']['outgoing'])
    dx = np.array([e['late_rad'] for e in valid])-center
    dv = np.array([e['first']['outgoing'] for e in valid])-vec
    slope = np.linalg.solve(dx.T@dx+.022*np.eye(5),dx.T@dv+.022*jac.T).T
    j = WEIGHT[:,None]*slope
    step = unit(np.linalg.solve(j.T@j+inertia,j.T@(WEIGHT*(goal-vec))))
    axes = (-lean,step,follow)
    options=[center+sign*r*a for a in axes for r in (.19,.30) for sign in (1.,-1.)]
    options=[p*(min(1.,.43/max(np.linalg.norm(p),1e-9))) for p in options]
    novel = [p for p in options if min(np.linalg.norm(p-e['late_rad']) for e in history)>.082]
    score = lambda p: np.linalg.norm(WEIGHT*(vec+slope@(p-center)-goal)) + .9*np.linalg.norm(body_j@p)
    return min(novel or options,key=score)


def run(feeds, output, only=None):
    begin=process_time()
    plan = _read(OUT/'local-plan.json.gz')
    prior_path = OUT/'inverse-prior.json'
    prior = _read(prior_path)
    previous_path = OUT/'inverse-checkpoint.json'
    previous = [dict(e,late=ZERO.tolist()) for e in _read(previous_path)['memory']]
    model_path = OUT/'impact-model.json'
    calibration = _read(model_path)['models']['60']
    model = _read(impact.BASE)
    launches = measured_launches(feeds)
    body = body_model()
    jac = np.asarray(prior['velocity_jacobian'])
    slots=[dict(s) for s in previous]
    archive = _read(OUT/'inverse-practice.json.gz')
    seed_by_id={sid:cases['new'] for sid,cases in zip(archive['ids'],archive['cases'],strict=True) if sid in IDS}
    result=dict(families=[], held=[], hashes={p.name:_hash(p) for p in (prior_path,previous_path, model_path,OUT/'inverse-practice.json.gz',impact.BASE)},
                initial_source='four inverse attempt tapes per release', new_trial_cap=8)
    with impact.centered(model['ready']):
        arena=impact.replay_arena([])
        renderer,rig,stride,lo,hi=joint.resources(arena)
        try:
            def play(card,seed,state):
                frames=Frames(rig)
                Eye = joint.Vision

                class Witness(Eye):
                    def update(self,tick,image,q):
                        frames.observe(image if image is not None else renderer.capture(),tick*arena.control_dt,q)
                        return super().update(tick,image,q)

                with patch.object(joint,'Vision',Witness), movement(body,choose_from(state,'late')):
                    row=local.trial(arena,renderer,rig,stride,lo,hi,model,launches,card,choose_from(state,'offset'),seed,progress=False)
                onset=row['picks'][0]['tick']*arena.control_dt
                return row,first_alarm(frames,calibration,onset)

            for index,case in enumerate(plan['cases']):
                entry=case['original']
                sid=entry['id']
                if sid not in IDS or (only is not None and only!=sid):
                    continue
                ctx=previous[index]['context']
                goal=target(prior,ctx)
                starts=[]
                body_poses={}
                for old in seed_by_id[sid]:
                    tape=old['row']
                    frames=Frames(rig)
                    with varied_opponent(entry['index'],tape['reply_seed']):
                        arena.launch(tape['shot'],card=entry)
                        arena.partner.reset(seed=entry['seed'])
                        for tick,action in enumerate(tape['actions']):
                            frames.observe(renderer.capture(),tick*arena.control_dt,arena.joints)
                            arena.exchange(action,joint.FAR_COMMAND)
                    if digest_json(tape['events'])!=digest_json(arena.events):
                        raise ValueError(f'Nonmatching replay {sid} attempt {old["attempt"]}')
                    e=first_alarm(frames,calibration,tape['picks'][0]['tick']*arena.control_dt)
                    if e is not None:
                        _,body_poses[old['attempt']]=min(frames.encoders,key=lambda pair:abs(pair[0]-e['time']))
                    starts.append(dict(attempt=old['attempt'],offset=old['offset_rad'],first=e,reward=tape['score']['learner'],late_rad=ZERO.tolist()))
                anchor=max(starts,key=lambda e:rank(e,goal))
                family=dict(id=sid, index=index, goal_mps=goal.tolist(), seeds=starts, episodes=[], retained=None)
                result['families'].append(family)
                print(sid,'visible starts',[(e['attempt'],None if e['first'] is None else (e['first']['estimator'],e['first']['outgoing'])) for e in starts],flush=True)
                if anchor['first'] is None:
                    continue
                slots[index]['offset']=anchor['offset']
                family['retained']=anchor['attempt']
                q=body_poses[anchor['attempt']]
                dq=np.eye(5)*.025
                body_j=np.array([(body(q+d)-body(q-d))/.05 for d in dq]).T
                history=[anchor]
                for i in range(8):
                    delta=propose(history,goal,jac,body_j,i)
                    slots[index]['late']=delta.tolist()
                    seed=368000+100*index+i
                    row,impact_seen=play(entry,seed,slots)
                    e=dict(attempt=i+1,late_rad=delta.tolist(),first=impact_seen,reward=row['score']['learner'],row=row)
                    history.append(e)
                    family['episodes'].append(e)
                    print(sid,i+1,e['reward'],None if impact_seen is None else np.round(impact_seen['outgoing'],3).tolist(),flush=True)
                    _write(output/'impact-refine.json.gz',result)
                best=max(history,key=lambda e:rank(e,goal))
                family['winner']=dict(late_rad=best['late_rad'],attempt=best['attempt'] if best is not anchor else 0, reward=best['reward'])
                slots[index]['late']=best['late_rad']
            retained=[dict(e,late=ZERO.tolist()) for e in slots]
            for family in result['families']:
                index=family['index']
                card=plan['cases'][index]['original']
                for seed in (369170+17*index,369171+17*index):
                    runs={}
                    for label,state in [('prior',previous),('retained',retained),('refined',slots)]:
                        row,saw=play(card,seed,state)
                        runs[label]=dict(row=row,first=saw,reward=row['score']['learner'])
                    result['held'].append(dict(id=card['id'],card=card,reply_seed=seed,runs=runs))
                    print('held',card['id'],seed,[(k,r['reward'],None if r['first'] is None else np.round(r['first']['outgoing'],3).tolist()) for k,r in runs.items()],flush=True)
        finally:
            renderer.close()
            arena.close()
    result['cpu_s']=process_time()-begin
    _write(output/'impact-refine.json.gz',result)
    _write(output/'impact-refine-checkpoint.json',dict(states=slots,tape_sha256=_hash(output/'impact-refine.json.gz')))


def summary(output):
    """After choosing movement, audit native impacts and exact first offers."""
    source = output/'impact-refine.json.gz'
    data = _read(source)
    plan = _read(OUT/'local-plan.json.gz')
    inverse = _read(OUT/'inverse-practice.json.gz')
    prior = {sid:{e['attempt']:e['row'] for e in cases['new']}
             for sid,cases in zip(inverse['ids'],inverse['cases'],strict=True)}

    def brief(trial, row):
        contact = impact.contacts(row)
        true = contact[0] if contact else None
        saw = trial['first']
        terminal = next((e for e in reversed(row['events']) if e['kind'] in ('point','timeout','let')),None)
        return dict(legal=trial['reward'], referee_note=None if terminal is None else terminal.get('reason'),
                    terminal_ball_m=None if terminal is None else terminal.get('ball'), physical_touch=true is not None,
                    native_mps=None if true is None else true['outgoing'],
                    optical_mps=None if saw is None else saw['outgoing'],
                    observer=None if saw is None else saw['estimator'],
                    t_error_s=None if true is None or saw is None else saw['time']-true['time'])

    def offer(row):
        d=row['placements']['delivered'][0]
        return [d[k] for k in ('target','far_contact','near_bounce','receive_line')]

    practice=[]
    for fam in data['families']:
        sid=fam['id']
        baseline=prior[sid][8]
        if any(offer(e['row'])!=offer(baseline) for e in fam['episodes']):
            raise ValueError(f'Practice offers changed for {sid}')
        practice.append(dict(id=sid, matched_offers=len(fam['episodes']), winner=fam.get('winner'),
            old_motion=[dict(attempt=e['attempt'],**brief(e,prior[sid][e['attempt']])) for e in fam['seeds']],
            late=[dict(attempt=e['attempt'],**brief(e,e['row'])) for e in fam['episodes']]))
    held, checks = [], []
    with TemporaryDirectory(prefix='tt36-impact-refine-') as work:
        temp=Path(work)
        for case in data['held']:
            rows=case['runs']
            present=[e['row'] for e in rows.values()]
            if not all(offer(r)==offer(present[0]) for r in present[1:]):
                raise ValueError('The held conditions got different first balls')
            original=rows['retained']['row']['actions']
            final=rows['refined']['row']['actions']
            different=next((i for i,(a,b) in enumerate(zip(original,final)) if a!=b),None)
            held.append(dict(id=case['id'], seed=case['reply_seed'], first_action_change_tick=different,
                             runs={k:brief(e,e['row']) for k,e in rows.items()}))
            # One held seed per family and condition is sufficient for the CAD replay.
            if sum(c['id']==case['id'] for c in held)>1:
                continue
            for mode, run in rows.items():
                p=temp/'point.json.gz'
                v=temp/'audit.json'
                _write(p,dict(cards=[case['card']],rows=[run['row']],ready=plan['ready']))
                with varied_opponent(case['card']['index'],case['reply_seed']):
                    joint.verify(p,v,cad=True)
                checks.append(dict(mode=mode,seed=case['reply_seed'],**_read(v)['points'][0]))
    _write(output/'impact-refine-audit.json',dict(source_sha256=_hash(source),
        plan_sha256=_hash(OUT/'local-plan.json.gz'),practice=practice,held=held,cad=checks))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('feeds',type=Path)
    parser.add_argument('--output',type=Path,default=OUT)
    parser.add_argument('--only',type=int,choices=IDS)
    parser.add_argument('--audit',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    if args.audit:
        summary(args.output)
    else:
        run(args.feeds,args.output,args.only)
