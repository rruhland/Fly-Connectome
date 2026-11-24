"""Find signed paddle deflections from stereo motion without requiring y reversal."""

import argparse
from contextlib import nullcontext
from pathlib import Path
from time import process_time

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_source_rally import blade_sighting
from .tt29_fast_camera import native_ball_pixels
from .tt29_optical_outcomes import stereo_position
from .tt31_curriculum_opponent import replay_arena
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, Vision, resources
from .tt35_error import triangulate
from .tt35_replies import varied_opponent
from .tt36_hit import optical_outcomes
from .tt36_miss import BASE, OUTPUT, ROOT


PRACTICE = (3928, 4363, 12403)
INVERSE_HELD = (6253, 6861, 12373)
TOLERANCE = .067


def examples(split):
    if split == 'practice':
        for sid in PRACTICE:
            source = OUTPUT / f'local-practice-{sid}.json.gz'
            document = _read(source)
            for episode in document['common']:
                yield source, document['card'], episode['row'], f'{sid}-{episode["attempt"]}'
        return
    inverse = OUTPUT / 'inverse-practice.json.gz'
    document = _read(inverse)
    if len(document['cases']) != len(document['ids']) or not set(INVERSE_HELD) <= set(document['ids']):
        raise ValueError('Inverse action tapes are incomplete')
    cases = {c['original']['id']: c['original'] for c in _read(OUTPUT/'local-plan.json.gz')['cases']}
    for sid, record in zip(document['ids'], document['cases'], strict=True):
        if sid in INVERSE_HELD:
            for attempt in record['new']:
                yield inverse, cases[sid], attempt['row'], f'{sid}-{attempt["attempt"]}'
    nominal = ROOT/'tt34'/'body-novel-stock.json.gz'
    source = _read(nominal)
    rows = [(c, r) for c, r in zip(source['cards'],source['rows'],strict=True)
            if c['id'] not in set(PRACTICE+INVERSE_HELD)]
    for c, r in rows[:8]:
        yield nominal, c, r, f'{c["id"]}-nominal'


def rgb_point(images, rig):
    names = ('left','right')
    balls = [native_ball_pixels(images[k]) for k in names]
    if any(x is None or x.clipped for x in balls):
        return None, None
    p = triangulate([v.center for v in balls], rig)
    if p is None or not (abs(p[0]) < 1.10 and -2.55 < p[1] < 1.50 and .54 < p[2] < 1.78):
        return None, None
    if p[1] > -.57:
        return p, None
    blades = [blade_sighting(images[k]) for k in names]
    if any(v is None for v in blades):
        return p, None
    b = triangulate([v[0]*images[k].shape[0]/360 for k,v in zip(names,blades,strict=True)], rig)
    if b is not None and not(-1.75 < b[1] < -.77 and .75 < b[2] < 1.40 and abs(b[0]) < .64):
        b = None
    return p, b


def contacts(row):
    """Evaluator labels, never read by calibration or the estimator."""
    hits = []
    for e in row['events']:
        if e['kind'] == 'contact' and e['side'] == 1:
            hits.append(dict(time=e['time'], incoming=e['incoming'], position=e['location'],
                             outgoing=None, separation_time=None))
        if e['kind'] == 'separation' and e.get('collider') == 'learner_racket' and hits and hits[-1]['outgoing'] is None:
            hits[-1].update(outgoing=e['velocity'], separation_time=e['time'])
    return hits


def capture(output, split):
    ready = _read(BASE)['ready']
    tracks, truth, sources = [], [], {}
    started = process_time()
    with centered(ready):
        arena = replay_arena([])
        renderer, rig, _, _, _ = resources(arena)
        try:
            for source, entry, row, key in examples(split):
                seed = row.get('reply_seed')
                with varied_opponent(entry['index'],seed) if seed is not None else nullcontext():
                    arena.launch(row['shot'],card=entry)
                    arena.partner.reset(seed=entry['seed'])
                    legacy = Vision(rig, arena.control_dt, raw=True)
                    raw, old = [], []
                    for i, action in enumerate(row['actions']):
                        frame = renderer.capture()
                        time = i * arena.control_dt
                        p, blade = rgb_point(frame,rig)
                        if p is not None:
                            raw.append([time,p.tolist(),None if blade is None else blade.tolist()])
                        if i % 2 == 0:
                            state = legacy.update(i,frame,arena.joints)
                            p = stereo_position(legacy.eye)
                            if p is not None:
                                old.append([time,p.tolist(),None if state is None else state.tolist()])
                        arena.exchange(action, FAR_COMMAND)
                    if digest_json(row['events']) != digest_json(arena.events):
                        raise ValueError(f'RGB replay changed game {key}')
                tracks.append(dict(key=key, id=entry['id'], offer_seed=entry['seed'], reply_seed=seed, raw=raw, legacy=old))
                truth.append(dict(key=key, id=entry['id'], contacts=contacts(row), arm_returns=row['score']['learner']))
                sources[source.name] = _hash(source)
                print(split,key,len(raw),len(old),flush=True)
        finally:
            renderer.close()
            arena.close()
    _write(output/f'impact-{split}-rgb.json.gz',dict(rows=tracks,sources=sources,cpu_s=process_time()-started))
    _write(output/f'impact-{split}-oracle.json',dict(rows=truth))


def track_at(raw, hz):
    return [(float(t), np.asarray(p), np.asarray(b) if b is not None else None)
            for t,p,b in raw if hz == 60 or round(t*60) % 2 == 0]


def acceleration_basis(velocity):
    return np.r_[1.,velocity*np.linalg.norm(velocity)]


def fit_acceleration(tracks, hz):
    inputs, outputs = [], []
    for track in tracks:
        for i in range(len(track)-4):
            packet = track[i:i+5]
            t = np.asarray([v[0] for v in packet])
            p = np.asarray([v[1] for v in packet])
            if np.diff(t).max() > 1.5/hz or not np.all((p[:,1] > -1.47) & (p[:,1] < .92) & (p[:,2] > .86)):
                continue
            if any((v[2] is None or np.linalg.norm(v[1]-v[2]) < .27) and v[1][1] < -.79 for v in packet):
                continue
            if np.any((abs(p[:,1]) < .14) & (p[:,2] < 1.01)):
                continue
            dt = t - t[2]
            _, v, a = np.linalg.lstsq(np.c_[np.ones(len(t)),dt,.5*dt**2],p,rcond=None)[0]
            if np.linalg.norm(a) < 36. and np.linalg.norm(v) > .4:
                inputs.append(acceleration_basis(v))
                outputs.append(a)
    if len(inputs) < 20:
        raise ValueError('Insufficient visibly free-flight intervals')
    x, y = np.asarray(inputs), np.asarray(outputs)
    regularization = .30*np.eye(x.shape[1])
    coefficients = np.linalg.solve(x.T@x + regularization,x.T@y)
    for _ in range(3):
        error = np.linalg.norm(x@coefficients-y,axis=1)
        w = np.minimum(1., max(1.2,1.7*np.median(error))/np.maximum(.01,error))
        coefficients = np.linalg.solve(x.T@(w[:,None]*x)+regularization,x.T@(w[:,None]*y))
    return coefficients, len(inputs)


def blade_at(track, time, available):
    before = next(((t,p) for t,_,p in reversed(track) if p is not None and 0 <= time-t <= .085), None)
    after = next(((t,p) for t,_,p in track if p is not None and time < t <= min(time+.085,available)), None)
    if before is not None and after is not None:
        return before[1] + (time-before[0])/(after[0]-before[0])*(after[1]-before[1])
    nearest = before if before is not None else after
    return nearest[1] if nearest is not None and abs(nearest[0]-time) <= .052 else None


def tangent(packet, time, a):
    t = np.array([v[0]-time for v in packet])
    p = np.array([v[1] for v in packet]) - .5 * t[:,None]**2 * a
    centered = t - t.mean()
    slope = centered@p / (centered@centered)
    return p.mean(axis=0) - t.mean()*slope, slope


def candidates(track, coefficients):
    for i in range(3,len(track)-1):
        left = track[i-3:i]
        right = track[i:i+3]
        before, after = left[-1][0], right[0][0]
        if not .012 <= after-before <= .117 or max(np.diff([v[0] for v in left]).max(),np.diff([v[0] for v in right]).max()) > .061:
            continue
        if not (-1.95 < left[-1][1][1] < -.56 and .74 < left[-1][1][2] < 1.44):
            continue
        if len(right)==3 and right[2][0]-right[0][0] > .090:
            right = right[:2]
        midpoint = .5*(before+after)
        _, first_v = tangent(left, midpoint, np.zeros(3))
        a = acceleration_basis(first_v)@coefficients
        p, v = tangent(left,midpoint,a)
        r, w = tangent(right,midpoint,a)
        change = w-v
        delta = np.linalg.norm(change)
        if delta <= .045 or v[1] >= -.56:
            continue
        shift = float(np.clip(-(r-p)@change/(delta**2),before-midpoint,after-midpoint))
        delta_pos = r-p+shift*change
        join = float(np.linalg.norm(delta_pos))
        center = .5*(p+r) + .5*(v+w)*shift + .5*a*shift**2
        if not (-1.78 < center[1] < -.59 and .822 < center[2] < 1.46 and join < .057):
            continue
        time = midpoint+shift
        blade = blade_at(track,time,right[-1][0])
        gap = float(np.linalg.norm(center-blade)) if blade is not None else None
        yield dict(time=time, available=right[-1][0], incoming=(v+a*shift).tolist(), outgoing=(w+a*shift).tolist(),
                   delta_v_mps=float(delta), join_m=join, center=center.tolist(), gap_m=gap,
                   n_before=len(left), n_after=len(right))


def calibrate(output):
    path = output/'impact-practice-rgb.json.gz'
    data = _read(path)
    models = {}
    for hz in (30,60):
        tracks = [track_at(row['raw'],hz) for row in data['rows']]
        coefficients, count = fit_acceleration(tracks,hz)
        null = [p['delta_v_mps'] for samples in tracks for p in candidates(samples,coefficients)
                if (p['gap_m'] is not None and p['gap_m'] > .235) or p['center'][1] > -.78]
        if len(null) < 20:
            raise ValueError('Insufficient matched image-only non-paddle transitions')
        models[str(hz)] = dict(hz=hz, acceleration=coefficients.tolist(), acceleration_intervals=count,
                              null_candidates=len(null), jump_threshold_mps=max(.40,float(np.quantile(null,.975))))
        print(hz, count, len(null),models[str(hz)]['jump_threshold_mps'],flush=True)
    _write(output/'impact-model.json',dict(models=models,rgb_sha256=_hash(path),release_ids=sorted({r['id'] for r in data['rows']})))


def estimate_impacts(track, calibration):
    """Read a chronological (time_s, ball_xyz, blade_xyz or None) RGB track.

    Returns signed velocities at detected events. Each candidate uses frames
    only through its `available` time. Scanning a longer track can replace a
    provisional result with a better-supported candidate for the same event.
    """
    if len(track) < 5:
        return []
    points = [(float(t),np.asarray(p),None if b is None else np.asarray(b)) for t,p,b in track]
    threshold = calibration['jump_threshold_mps']
    rivals = [p for p in candidates(points,np.asarray(calibration['acceleration']))
             if p['gap_m'] is not None and p['gap_m'] <= .166 and p['center'][1] < -.84
             and p['delta_v_mps'] > threshold]
    accepted = []
    for p in sorted(rivals,key=lambda x: x['delta_v_mps']/(1+.60*x['join_m']/.04 + .29*x['gap_m']/.11),reverse=True):
        if all(abs(p['time']-t['time']) >= .185 for t in accepted):
            accepted.append(p)
    return sorted(accepted,key=lambda r:r['time'])


def match(detected, contacts):
    edges = sorted((abs(p['time']-t['time']),i,j) for i,p in enumerate(detected) for j,t in enumerate(contacts)
                   if abs(p['time']-t['time']) <= TOLERANCE)
    used, matched = set(), set()
    pairs=[]
    for dt,i,j in edges:
        if i in used or j in matched:
            continue
        truth = contacts[j]['outgoing']
        estimate=np.asarray(detected[i]['outgoing'])
        change = None if truth is None else estimate-np.asarray(truth)
        pairs.append(dict(alarm=i, contact=j, time_error_s=dt, velocity_error_mps=None if change is None else change.tolist(),
                          direction_ok=None if truth is None else bool((estimate[1] > 0)==(truth[1] > 0))))
        used.add(i)
        matched.add(j)
    return dict(pairs=pairs, false_alarms=[i for i in range(len(detected)) if i not in used],
                missed=[i for i in range(len(contacts)) if i not in matched])


def evaluate(output):
    state = _read(output/'impact-model.json')
    result = {}
    for split in ('practice','held'):
        rgb = _read(output/f'impact-{split}-rgb.json.gz')['rows']
        if split == 'held' and set(state['release_ids']) & {r['id'] for r in rgb}:
            raise ValueError('A held release entered the free-flight fit')
        # Run all estimators without opening the physical labels.
        predicted = {}
        for r in rgb:
            old = optical_outcomes([(t,np.array(p),None if s is None else np.array(s)) for t,p,s in r['legacy']])
            states = {t:np.array(s) for t,_,s in r['legacy'] if s is not None}
            identical_input = optical_outcomes([(t,p,states.get(t)) for t,p,_ in track_at(r['raw'],30)])
            predicted[r['key']] = {'legacy30':[dict(time=e['reversal_time'], outgoing=e['velocity']) for e in old],
                'same_legacy30':[dict(time=e['reversal_time'], outgoing=e['velocity']) for e in identical_input],
                **{f'change{hz}':estimate_impacts(track_at(r['raw'],hz),state['models'][str(hz)]) for hz in (30,60)}}
        truth = _read(output/f'impact-{split}-oracle.json')['rows']
        rows=[]
        aggregate={k:dict(hits=0, touches=0, false=0, rear_hits=0, rear=0, errors=[], rear_errors=[],
                          y_signs=[], negative_episodes=0, false_episodes=0, delay=[])
                   for k in ('legacy30','same_legacy30','change30','change60')}
        common = {name:[] for name in ('change30','change60')}
        by_key = {r['key']:r for r in rgb}
        for record in truth:
            key=record['key']
            hits=record['contacts']
            evaluations={}
            for method, found in predicted[key].items():
                result_row=match(found,hits)
                summary=aggregate[method]
                summary['hits'] += len(result_row['pairs'])
                summary['touches'] += len(hits)
                summary['false'] += len(result_row['false_alarms'])
                if not hits:
                    summary['negative_episodes'] += 1
                    summary['false_episodes'] += bool(found)
                rear = {i for i,touch in enumerate(hits) if touch['outgoing'] is not None and touch['outgoing'][1] <= 0}
                summary['rear'] += len(rear)
                summary['rear_hits'] += sum(r['contact'] in rear for r in result_row['pairs'])
                summary['errors'].extend(r['velocity_error_mps'] for r in result_row['pairs'] if r['velocity_error_mps'] is not None)
                summary['rear_errors'].extend(r['velocity_error_mps'] for r in result_row['pairs']
                                              if r['contact'] in rear and r['velocity_error_mps'] is not None)
                summary['y_signs'].extend(r['direction_ok'] for r in result_row['pairs'] if r['direction_ok'] is not None)
                summary['delay'].extend(found[r['alarm']]['available']-hits[r['contact']]['time']
                                        for r in result_row['pairs'] if 'available' in found[r['alarm']])
                evaluations[method]=dict(found=found,**result_row)
            paired_errors = {m:{p['contact']:np.linalg.norm(p['velocity_error_mps']) for p in d['pairs']
                                if p['velocity_error_mps'] is not None} for m,d in evaluations.items()}
            old_errors = paired_errors['legacy30']
            for mode, records in common.items():
                records.extend((old_errors[i],e) for i,e in paired_errors[mode].items() if i in old_errors)
            capture_support=[]
            for hit in hits:
                t=hit['time']
                raw=by_key[key]['raw']
                capture_support.append(dict(before_100ms=sum(t-.10 <= v[0]<t for v in raw),
                                            after_100ms=sum(t < v[0] <= t+.10 for v in raw)))
            rows.append(dict(record, detector=evaluations, capture_support=capture_support))
        for m, d in aggregate.items():
            errors=np.asarray(d.pop('errors'))
            d['median_velocity_error_mps']=float(np.median(np.linalg.norm(errors,axis=1))) if len(errors) else None
            d['axis_mae_mps']=np.mean(abs(errors),axis=0).tolist() if len(errors) else None
            rear_errors=np.asarray(d.pop('rear_errors'))
            d['nonforward_median_velocity_error_mps']=float(np.median(np.linalg.norm(rear_errors,axis=1))) if len(rear_errors) else None
            delay=d.pop('delay')
            d['median_alarm_delay_s']=float(np.median(delay)) if delay else None
            signs=d.pop('y_signs')
            d['velocity_y_sign_correct']=sum(signs)
            d['velocity_y_sign_count']=len(signs)
            d['episodes']=len(rows)
            print(split,m,d,flush=True)
        result[split]=dict(summary=aggregate, rows=rows, same_contacts={mode:dict(count=len(records),
            median_error_mps=np.median(records,axis=0).tolist() if records else None,
            order=['legacy30',mode]) for mode,records in common.items()})
    _write(output/'impact-results.json.gz',dict(models_sha256=_hash(output/'impact-model.json'),tolerance_s=TOLERANCE,split=result))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('practice','held','fit','evaluate'))
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    args.output.mkdir(exist_ok=True,parents=True)
    if args.phase=='fit': calibrate(args.output)
    elif args.phase=='evaluate': evaluate(args.output)
    else: capture(args.output,args.phase)


if __name__=='__main__':
    main()
