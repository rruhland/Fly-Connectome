"""Change only angular velocity in a private twin on held, actually issued action tapes."""

import argparse
from pathlib import Path
from time import process_time

import mujoco
import numpy as np

from . import tt33_joint as joint
from .tt35_error import passage
from .tt36_impact_refine import Frames
from .tt37_opponent import arena
from .tt37_optical_clinic import OpticalClinic
from .tt37_visual_inertia import LearnedOptics
from .tt38_contact_id import AIM, HELD, IMPACT, OLD, VISION, outcome_cost
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt31_ready import centered
from .tt39_spin import Observer, solve


OUT = Path('artifacts/tt39')
ARMS = ('zero', 'legacy', 'arc', 'coupled', 'oracle-spin-audit')
HORIZON = 25


def roll(engine, stored, bounced, goals, eta, spins):
    t = engine.twin
    start = t._ball_dof+3
    answer = {}
    for mode, angular in spins.items():
        mujoco.mj_copyData(t.data, t.model, stored)
        t._received_bounce = bounced
        t.data.qvel[start:start+3] = angular
        f = np.zeros((1, 9))
        _, result = engine.trajectories(goals[None], AIM, eta+.13, feedback=f)
        answer[mode] = dict(feedback=f[0].tolist(), legal=bool(np.isfinite(result[0, 0])))
    return answer


def ingest(view, now, frame, q, obs):
    view.observe(frame, now, q)
    last = view.stereo[-1] if view.stereo else None
    point = last[1] if last is not None and abs(last[0]-now)<1e-7 else None
    obs.ingest(now, point)


def capture(a, renderer, rig, stride, lo, hi, twin, row, card, ready, cfg, impact):
    start = process_time()
    dt = a.control_dt
    a.launch(row['shot'], card=card)
    a.partner.reset(seed=card['seed'])
    twin.previous = None
    twin.bounced = False
    eye = joint.Vision(rig, dt)
    eye.eye = LearnedOptics(eye.eye, VISION)
    view = Frames(rig)
    readout = Observer(dt, cfg['settings'])
    commands = np.asarray(row['actions'])
    ref = np.asarray(ready)
    goals = []
    for cmd in commands:
        ref = np.clip(ref+stride*cmd, lo, hi)
        goals.append(ref.copy())
    goals = np.asarray(goals)
    ref = np.asarray(ready)
    snapshots = []
    taken = -5.
    t = twin.twin
    angular_index = t._ball_dof+3
    for tick, cmd in enumerate(commands):
        now = dt*tick
        q = a.joints.copy()
        last = eye.eye.latest
        dense = last is not None and now-last.time<.11 and -.1>last.velocity[1] and -.2<last.position[2]<1.8 and last.position[1]<.65
        images = renderer.capture() if tick % 2 == 0 or dense else None
        obs = eye.update(tick, images if tick % 2 == 0 else None, q)
        if images is not None and (dense or (obs is not None and 0.<obs[0]<.30)):
            ingest(view, now, images, q, readout)
        twin.observe(tick, eye.eye, q, ref.copy())
        ready_goal = goals[tick:tick+HORIZON]
        if (obs is not None and .085<obs[0]<.182 and now-taken>.53 and len(ready_goal)>=5
                and twin.latest is not None and twin.latest.rms<2.4 and twin.latest.depth_sigma<.16
                and now-twin.latest.time<.075 and twin.sync()):
            measured = readout.evidence() if readout.samples and now-readout.samples[-1][0]<=dt+.001 else None
            if measured is not None:
                ts, pp, pack, time, bounce = measured
                saved = mujoco.MjData(t.model)
                mujoco.mj_copyData(saved, t.model, t.data)
                legacy = t.data.qvel[angular_index:angular_index+3].copy()
                spins = {'zero': np.zeros(3), 'legacy': legacy}
                res = {}
                for mode in ('arc','coupled'):
                    e = solve(ts, pp, pack, cfg['tuned'][mode], mode, time, bounce)
                    spins[mode] = e.state(now)[1]
                    res[mode] = dict(spin=spins[mode].tolist(), uncertainty=e.posterior.tolist(),
                                     residual=e.residual)
                pre = roll(twin, saved, bool(t._received_bounce), ready_goal, float(obs[0]), spins)
                # The one privileged replay is an explicitly non-deployable angular ablation. Its
                # result never feeds the optical input, fit, snapshot timing or other arms.
                pre.update(roll(twin, saved, bool(t._received_bounce), ready_goal, float(obs[0]),
                                {'oracle-spin-audit': a.spin.copy()}))
                snapshots.append(dict(tick=tick, time=now, eta=float(obs[0]), inferred=res,
                                      angular_baseline=legacy.tolist(), optical_bounce=bounce,
                                      public_pos=saved.qpos[t._ball_joint:t._ball_joint+3].tolist(),
                                      public_vel=saved.qvel[t._ball_dof:t._ball_dof+3].tolist(),
                                      predictions=pre))
                taken = now
        a.exchange(cmd, joint.FAR_COMMAND)
        ref = goals[tick]
    if digest_json(a.events) != digest_json(row['events']):
        raise ValueError('Issued controls no longer reproduce the physical solver events')
    detections = view.estimates(impact)
    frames = [(round(t/dt), p, b) for t,p,b in view.stereo if b is not None]
    for x in snapshots:
        i, now, eta = x['tick'], x['time'], x['eta']
        matches = [d for d in detections if .022<d['time']-now<.345 and abs(d['time']-now-eta)<.145]
        actual = min(matches, key=lambda d:abs(d['time']-now-eta)) if matches else None
        miss = None
        if actual is None:
            p = passage([(j,b,d) for j,d,b in frames if i<j<=i+23], i)
            if p is not None and np.linalg.norm(p['gap_xz_m'])>.13:
                miss = float(np.linalg.norm(p['gap_xz_m']))
        x['measured'] = actual
        x['camera_miss_m'] = miss
        cutoff = min(.34, eta+.13)
        contacts = [e for e in a.events if e.get('kind')=='contact' and e.get('side')==1
                    and now < e['time'] < now+cutoff]
        x['scored_contact'] = bool(contacts)
        x['evaluator_contact_time'] = contacts[0]['time'] if contacts else None
        if actual is not None or miss is not None:
            data = np.array([x['predictions'][m]['feedback'] for m in ARMS])
            x['measured_losses'] = dict(zip(ARMS, outcome_cost(data, actual, miss).tolist(), strict=True))
    return snapshots, process_time()-start


def selected(path):
    return [int(v) for v in np.unique(np.load(path)['groups'][:, 0])]


def run(checkpoint_path, practice, output, limit, resume=False):
    cfg = _read(checkpoint_path)
    impact = _read(IMPACT)['models']['60']
    train = set(selected(practice))
    if cfg['optics']['practice'] != _hash(practice):
        raise ValueError('The checkpoint uses another practice receipt')
    receipt = dict(schema='tt39-spin-contact-1', model_sha256=_hash(checkpoint_path),
                   p_and_v_source_sha256=_hash(VISION), labels_sha256=_hash(IMPACT),
                   score='later RGB departure or clear passage. Independently timed physical contact audit',
                   sources={}, points=[])
    if resume:
        previous = _read(output)
        if any(previous[k] != value for k,value in receipt.items() if k not in ('sources','points')):
            raise ValueError('Cannot resume with changed optical or angular models')
        for source, checksum in previous['sources'].items():
            if checksum != _hash(OLD/f'{source}.json.gz'):
                raise ValueError('Cannot resume a changed input action tape')
        receipt['sources'] = previous['sources']
        receipt['points'] = previous['points']
    seen = {(r['source'],r['index']) for r in receipt['points']}
    for source, start, stop in HELD:
        if len(receipt['points'])>=limit:
            break
        file = OLD/f'{source}.json.gz'
        doc = _read(file)
        receipt['sources'][source] = _hash(file)
        points = list(zip(doc['cards'][start:stop],doc['rows'][start:stop],strict=True))
        if train & {c['id'] for c, _ in points}:
            raise ValueError('Contact evaluation overlaps a development release')
        joint.RECEIVE_Y = doc['depth']
        with centered(doc['ready']):
            a = arena(protocol='random')
            twin = OpticalClinic(body=True)
            s, rig, stride, lo, hi = joint.resources(a)
            try:
                for j, (c,r) in enumerate(points,start):
                    if len(receipt['points'])>=limit:
                        break
                    if (source,j) in seen:
                        continue
                    records,cpu = capture(a,s,rig,stride,lo,hi,twin,r,c,doc['ready'],cfg,impact)
                    receipt['points'].append(dict(source=source, index=j, card=c, rows=records, cpu_s=cpu))
                    _write(output,receipt)
                    print(source,j,len(records),sum('measured_losses' in d for d in records),flush=True)
            finally:
                s.close();a.close();twin.twin.close()
    receipt['complete'] = True
    _write(output,receipt)


def comparison(data):
    rows = [dict(r, release=p['card']['id']) for p in data['points'] for r in p['rows']]
    assessed = [r for r in rows if 'measured_losses' in r]
    common = [r for r in assessed if r['measured'] is not None
              and all(r['predictions'][m]['feedback'][1]>.5 for m in ARMS)]
    result = dict(snapshots=len(rows), camera_scored=len(assessed), common_hits=len(common),
                  optical_departures=sum(r['measured'] is not None for r in assessed),
                  audited_contacts=sum(r['scored_contact'] for r in rows), modes={})
    for m in ARMS:
        hit = np.array([r['predictions'][m]['feedback'][1]>.5 for r in rows], dtype=bool)
        truth = np.array([r['scored_contact'] for r in rows], dtype=bool)
        err = [np.linalg.norm(np.asarray(r['predictions'][m]['feedback'][2:5])-r['measured']['outgoing'])
               for r in common]
        also = [r for r in assessed if r['measured'] is not None and r['predictions'][m]['feedback'][1]>.5]
        allerr = [np.linalg.norm(np.asarray(r['predictions'][m]['feedback'][2:5])-r['measured']['outgoing'])
                  for r in also]
        result['modes'][m] = dict(predicted_contacts=int(hit.sum()), true_positive=int((hit & truth).sum()),
                                  false_positive=int((hit & ~truth).sum()), false_negative=int((~hit & truth).sum()),
                                  optical_departures_matched=len(also),
                                  optical_cost=float(np.mean([r['measured_losses'][m] for r in assessed])) if assessed else None,
                                  velocity_error_common_mps=float(np.mean(err)) if err else None,
                                  velocity_error_measured_contacts_mps=float(np.mean(allerr)) if allerr else None)
    result['paired_velocity_error_change_vs_legacy'] = {}
    if common:
        def deviation(row, arm):
            return np.linalg.norm(np.asarray(row['predictions'][arm]['feedback'][2:5])-row['measured']['outgoing'])

        prior = np.array([deviation(r,'legacy') for r in common])
        groups = np.array([r['release'] for r in common])
        for m in ('zero','arc','coupled','oracle-spin-audit'):
            delta = np.array([deviation(r,m) for r in common])-prior
            hold = [float(np.mean(delta[groups!=g])) for g in np.unique(groups) if np.any(groups!=g)]
            result['paired_velocity_error_change_vs_legacy'][m] = dict(
                better=int((delta<-.01).sum()), worse=int((delta>.01).sum()),
                same=int((abs(delta)<=.01).sum()), mean_mps=float(np.mean(delta)),
                median_mps=float(np.median(delta)), leave_one_release_out_mps=[min(hold),max(hold)] if hold else None)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,default=OUT/'spin-checkpoint.json')
    p.add_argument('--practice',type=Path,default=OLD/'hz60-practice.npz')
    p.add_argument('--output',type=Path,default=OUT/'spin-contact.json.gz')
    p.add_argument('--report',type=Path,default=OUT/'spin-contact-summary.json')
    p.add_argument('--limit',type=int,default=9)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--summarize-only', action='store_true')
    a = p.parse_args()
    if not a.summarize_only:
        run(a.checkpoint,a.practice,a.output,a.limit,a.resume)
    receipt = _read(a.output)
    if not receipt.get('complete'):
        raise ValueError('Contact replay not complete')
    _write(a.report, comparison(receipt))
