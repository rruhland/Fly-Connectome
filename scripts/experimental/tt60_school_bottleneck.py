"""Locate why physically issued school strokes are not recognized on ordinary RGB.

Exact native quantities in this module are evaluator-only. Never import this
diagnostic in an actor or use its receipts as a live source of ball/arm state.
"""

import argparse
from collections import Counter
from itertools import product
from pathlib import Path

import mujoco
import numpy as np

from . import tt57_motor_school as school
from . import tt57_motor_school_mature as maturity
from . import tt57_motor_school_transfer as transfer
from .tt4_transfer_trial import _hash, _read, _write
from .tt29_rally_physics import Rollout
from .tt37_opponent import MAX_CONTROL_STEPS
from .tt37_optical_clinic import OpticalClinic


ROOT = Path('artifacts/tt60_school_bottleneck')
FACTOR = ROOT / 'practice-factors.json.gz'
HISTORY = ROOT / 'ordinary-history.json'
REPORT = ROOT / 'report.json'
ORDINARY = school.season.ROOT / 'learning.json.gz'
FACTORS = ('position_velocity', 'spin', 'bounce', 'body')


def group(row, n):
    served = row['placements']['delivered']
    if n == len(served):
        if len(row['actions']) >= MAX_CONTROL_STEPS:
            return 'censored'
        return 'no_legal_opponent_delivery'
    if len(served) != n + 1 or served[-1]['near_landed']:
        raise ValueError('Scores do not match uninterrupted legal deliveries')
    last = served[-1]
    if last['near_contact'] is None:
        return 'receive_no_contact'
    if last['receive_line'] is None:
        return 'receive_no_line'
    phase = last['receive_line']
    z, x = school.projected_height(phase), phase['xyz'][0]
    lo, hi = (.997, 1.054) if -.10 < x < .10 else (.979, 1.084)
    if z < lo:
        return 'contact_below_school_range'
    if z > hi:
        return 'contact_above_school_range'
    if last['near_contact']['racket'][2] - last['near_contact']['location'][2] < .038:
        return 'school_height_without_low_face'
    looks = [v for v in row['searches'] if v.get('p') and -.99 < v['p'][1] < -.51
             and last['far_contact']['step'] + 3 < v['tick'] < last['near_contact']['step'] - 9]
    return 'school_envelope' if looks else 'school_height_without_search'


def history():
    """Count release-level endings and chain transitions without taking actions."""
    doc = _read(ORDINARY)
    lines, trans = [], Counter()
    windows = {'0': Counter(), '1-4': Counter(), '5-9': Counter(), '10-19': Counter(), '20+': Counter()}
    names = tuple(windows)
    # A capped tape is censored by the game's score limit or native max_steps.
    # This particular historical docket has no cap at or below its maximum.
    if max(r['score']['learner'] for r in doc['rows']) >= doc['cap']:
        raise ValueError('Unexpected score censoring in the frozen history')
    for card, row in zip(doc['cards'], doc['rows'], strict=True):
        if any(x['draw'] != 'random' for x in row['placements']['requested']):
            raise ValueError('Historical standard random draw changed')
        n = row['score']['learner']
        final = group(row, n)
        terminal = row['events'][-1]
        lines.append(dict(id=card['id'], score=n, terminal=terminal.get('reason', terminal['kind']), class_=final))
        for leg in range(n+1):
            k = int(np.searchsorted([1, 5, 10, 20], leg, side='right'))
            stage = 'continue' if leg < n else final
            windows[names[k]][stage] += 1
            trans[stage] += 1
    result = dict(source=str(ORDINARY), sha256=_hash(ORDINARY), counts=dict(trans),
                  at_own_return_count={k:dict(v) for k,v in windows.items()}, rows=lines,
                  scope='All continuously randomized standard training games, never held results. Each completed own stroke advances the chain. The ending class is a physical descriptive census, not proof a class can all be repaired or that transitions are independent.')
    ROOT.mkdir(exist_ok=True, parents=True)
    _write(HISTORY, result)
    print('historical', dict(trans), flush=True)


def goals_for(s, ref, stride, lo, hi):
    cmd = np.zeros((school.H, 6))
    n = min(len(s['controls']), school.H)
    cmd[:n] = s['controls'][:n]
    path = school.atlas.reference_path(ref, cmd, stride, lo, hi)[1:] - ref
    return path, cmd


def substitutions(clinic, live, mode):
    if not clinic.sync():
        raise ValueError('No live public flight observation at the practice fork')
    t = clinic.twin
    use = dict(zip(FACTORS, mode, strict=True))
    if use['position_velocity']:
        t.ball[:] = live.ball
        t.velocity[:] = live.velocity
    if use['spin']:
        t.data.qvel[t._ball_dof + 3:t._ball_dof + 6] = live.spin
    if use['bounce']:
        t._received_bounce = bool(live._received_bounce)
    if use['body']:
        d, native = t.data, live.data
        p, v = clinic.paddle_q, clinic.paddle_v
        d.qpos[clinic.positions] = native.qpos[clinic.positions]
        d.qvel[clinic.dofs] = native.qvel[clinic.dofs]
        d.qpos[p:p+7] = native.qpos[p:p+7]
        d.qvel[v:v+6] = native.qvel[v:v+6]
        d.qacc_warmstart[clinic.dofs] = native.qacc_warmstart[clinic.dofs]
        d.qacc_warmstart[v:v+6] = native.qacc_warmstart[v:v+6]
    mujoco.mj_forward(t.model, t.data)


def verdict(engine, choices, cmd, ref, precedes_contact):
    u, y, feedback, loss, legal = school.native(engine, np.asarray(choices), ref)
    n = len(choices)
    mags = np.abs(u[:, :school.H] - cmd).max(axis=2)
    before = np.arange(school.H)[None, :] < precedes_contact[:, None]
    return [dict(legal=bool(legal[j]), loss=float(loss[j]),
                 separation=bool(feedback[j,1] > .5),
                 nearest_m=float(feedback[j,0]),
                 radius_m=float(feedback[j,8]) if feedback[j,1] > .5 else None,
                 impact_xyz=feedback[j,5:8].tolist() if feedback[j,1] > .5 else None,
                 outgoing_xyz=feedback[j,2:5].tolist() if feedback[j,1] > .5 else None,
                 landing_xy=y[j,3:5].tolist() if legal[j] else None,
                 input_delta_before_measured_contact=float(np.max(np.where(before[j], mags[j], 0.))))
                 for j in range(n)]


def audit(n):
    plan, trained = _read(school.PLAN), _read(maturity.OUT)
    if trained['protocol_sha256'] != _hash(school.PLAN) or trained['original_practice_sha256'] != _hash(school.OUT):
        raise ValueError('Changed motor-school practice source')
    practice = {tuple(r[k] for k in ('id', 'source', 'i', 'tick')) for r in plan['practice']}
    held_ids = {c['id'] for c in _read(school.season.PLAN)['assessment']}
    for r in trained['rows']:
        c = r['case']
        if c['id'] in held_ids or tuple(c[k] for k in ('id', 'source', 'i', 'tick')) not in practice:
            raise ValueError('Factor audit must be practice only')
    receipt = dict(schema='tt60-school-recognition-factor-1',
                   source_sha256=_hash(maturity.OUT), training_protocol=_hash(school.PLAN),
                   optics_sha256=_hash(school.course.OPTIC), operator=_hash(Path(__file__)),
                   factor_order=FACTORS, complete=False, rows=[])
    if FACTOR.exists():
        old = _read(FACTOR)
        if any(old[k] != v for k,v in receipt.items() if k not in ('rows','complete','factor_order')):
            raise ValueError('Changed diagnostic, archive to a new output before continuing')
        receipt = old
    data = {}
    ROOT.mkdir(exist_ok=True, parents=True)
    with school.enabled():
        for r in trained['rows'][len(receipt['rows']):n]:
            c = r['case']; src = c['source']; i = c['i']; tick = c['tick']
            if src not in data:
                if _hash(Path(src)) != plan['sources'][src]:
                    raise ValueError('Changed physical practice tape')
                data[src] = _read(Path(src))
            doc = data[src]
            row = doc['rows'][i]
            with school.mpc.centered(doc['ready']):
                a = school.mpc.arena()
                cam, rig, stride, lo, hi = school.mpc.resources(a)
                clinic = OpticalClinic(body=True)
                try:
                    args = (a, cam, rig, doc['cards'][i], row, tick, a._arm_ids[1][2])
                    eye, x, q, ref, _, features, hist = school.observed_prefix(*args)
                    rec = dict(id=c['id'], tick=tick, lane=c['lane'], height=c['height'], leg=c['leg'])
                    if x is not None and features is not None and r.get('inputs'):
                        if not np.allclose(x, r['inputs']['x'], atol=1e-10, rtol=0):
                            raise ValueError('Frozen RGB prefix changed')
                        transfer.warm_clinic(clinic, rig, a.control_dt, hist, eye, tick, q, ref)
                        t = clinic.twin
                        substitutions(clinic, a, (False,)*len(FACTORS))
                        real_sp = a.spin.copy()
                        optical_sp = t.spin.copy()
                        rec.update(optical=dict(x=x.tolist(), q=q.tolist(), ref=ref.tolist(),
                                                encoder_rate=clinic.rate.tolist(), spin=optical_sp.tolist(),
                                                freshness_s=float(tick*a.control_dt-clinic.latest.time),
                                                bounce=bool(t._received_bounce)),
                                   residual=dict(x=(np.r_[a.ball, a.velocity]-x).tolist(),
                                                 spin=(real_sp-optical_sp).tolist(),
                                                 bounce=bool(a._received_bounce),
                                                 q=(a.data.qpos[clinic.positions]-t.data.qpos[clinic.positions]).tolist(),
                                                 qdot=(a.data.qvel[clinic.dofs]-t.data.qvel[clinic.dofs]).tolist(),
                                                 paddle_p=(a.data.qpos[clinic.paddle_q:clinic.paddle_q+3]-t.data.qpos[clinic.paddle_q:clinic.paddle_q+3]).tolist(),
                                                 paddle_v=(a.data.qvel[clinic.paddle_v:clinic.paddle_v+6]-t.data.qvel[clinic.paddle_v:clinic.paddle_v+6]).tolist()))
                        actual = [r['stock']] + [v for v in r.get('arms', []) if v.get('controls')]
                        paths, cmd = map(np.asarray, zip(*(goals_for(s, ref, stride, lo, hi) for s in actual), strict=True))
                        precedes_contact = np.array([min(school.H, int(np.ceil(s['actual']['when']/a.control_dt)))
                                                     if s.get('actual') else school.H for s in actual])
                        rec['tapes'] = [dict(landed=s['landed'], answered=s['answered'],
                                             safe=bool(s.get('issued_relative')),
                                             event_sha256=s['event_sha256'],
                                             measured_contact_location=s['actual']['location'] if s.get('actual') else None,
                                             measured_contact_tick=int(precedes_contact[j])) for j,s in enumerate(actual)]
                        private = Rollout(a, 1)
                        rec['arms'] = {'exact_native': verdict(private, paths, cmd, ref, precedes_contact)}
                        for mode in product((False, True), repeat=len(FACTORS)):
                            substitutions(clinic, a, mode)
                            label = '+'.join(k for k,v in zip(FACTORS, mode, strict=True) if v) or 'optical'
                            rec['arms'][label] = verdict(clinic.predict, paths, cmd, ref, precedes_contact)
                        rec['original_optical_gate'] = transfer.choose(clinic, paths.tolist(),
                            [dict(tape=j) for j in range(len(paths))],ref)[1]
                    else:
                        rec['unobservable'] = True
                finally:
                    clinic.twin.close(); cam.close(); a.close()
            receipt['rows'].append(rec)
            receipt['complete'] = len(receipt['rows']) == len(trained['rows'])
            _write(FACTOR, receipt)
            selected = ('optical','position_velocity','spin','bounce','body','exact_native')
            print(len(receipt['rows']), c['id'], c['lane'],
                  [[z['landed'] for z in rec.get('tapes',())],
                   [[z['legal'] for z in rec['arms'][k]] for k in selected] if 'arms' in rec else None], flush=True)


def report():
    raw = _read(FACTOR)
    observed = [r for r in raw['rows'] if 'arms' in r]
    modes = tuple(observed[0]['arms'])
    rec = dict(factor_sha256=_hash(FACTOR), historical_sha256=_hash(HISTORY),
               complete=raw['complete'], cases=len(raw['rows']), observable=len(observed), modes={})
    rec['physically_landed'] = sum(a['landed'] for r in observed for a in r['tapes'])
    rec['physically_answered'] = sum(a['landed'] and a['answered'] and a['safe'] for r in observed for a in r['tapes'])
    rec['physically_failed'] = sum(not a['landed'] for r in observed for a in r['tapes'])
    for k in modes:
        pairs = [(a,b) for r in observed for a,b in zip(r['tapes'],r['arms'][k],strict=True)]
        ranked = [min(((a,b) for a,b in zip(r['tapes'],r['arms'][k],strict=True) if b['legal']),
                      key=lambda pair:pair[1]['loss'], default=None) for r in observed]
        rec['modes'][k] = dict(accept_landed=sum(a['landed'] and b['legal'] for a,b in pairs),
                               accept_safe_answered=sum(a['answered'] and a['safe'] and b['legal'] for a,b in pairs),
                               accept_failed=sum(not a['landed'] and b['legal'] for a,b in pairs),
                               separation_landed=sum(a['landed'] and b['separation'] for a,b in pairs),
                               selection_landed=sum(a is not None and a[0]['landed'] for a in ranked),
                               selection_failed=sum(a is not None and not a[0]['landed'] for a in ranked))
    for side in ('x','spin','q','qdot','paddle_p','paddle_v'):
        data = np.asarray([r['residual'][side] for r in observed]); m = np.median(data, axis=0)
        dev = np.quantile(abs(data), [.5,.9], axis=0)
        rec[side] = dict(signed_median=m.tolist(), absolute_quantiles=dev.tolist())
    rec['bounce_confusion'] = dict(Counter(f"{r['optical']['bounce']}/{r['residual']['bounce']}" for r in observed))
    _write(REPORT, rec)
    print(rec, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=('history','audit','report'))
    p.add_argument('--n', type=int, default=40)
    a = p.parse_args()
    {'history':history, 'report':report, 'audit':lambda:audit(a.n)}[a.mode]()
