"""Test cross-fitted public ball-state hypotheses on fixed issued practice paths."""

import argparse
from pathlib import Path

import numpy as np

from . import tt57_motor_school as school
from . import tt57_motor_school_mature as maturity
from . import tt57_motor_school_transfer as transfer
from . import tt60_school_bottleneck as bottleneck
from .tt4_transfer_trial import _hash, _read, _write
from .tt37_optical_clinic import OpticalClinic


MODEL = bottleneck.ROOT / 'optical-gate.npz'
OUT = bottleneck.ROOT / 'cross-fitted-gate.json.gz'
REPORT = bottleneck.ROOT / 'gate-report.json'
LIMITS = (.12, .22, .35, .50, .65, .80)
K = 10


def feature(x, age):
    """Only the current public flight and exposure timing, not release or reply seed."""
    x = np.asarray(x)
    return np.r_[x[0]/.20, x[1]/.19, x[2]/.055, x[3]/.46, x[4]/.64, x[5]/.80, age/.033]


def fit():
    d = _read(bottleneck.FACTOR)
    if not d['complete']:
        raise ValueError('Wait for the complete frozen practice calibration docket')
    rows = [r for r in d['rows'] if 'arms' in r]
    np.savez_compressed(MODEL, features=[feature(r['optical']['x'],r['optical']['freshness_s']) for r in rows],
                        phase=[r['optical']['bounce'] for r in rows],
                        residual=[r['residual']['x'] + r['residual']['spin'] for r in rows],
                        release=[r['id'] for r in rows], practice=_hash(bottleneck.FACTOR))


class Gate:
    """Correlated translational/angular hypotheses fit on ordinary practice RGB only at inference."""
    def __init__(self, path=MODEL):
        with np.load(path) as data:
            self.features = data['features']
            self.residual = data['residual']
            self.release = data['release']
            self.phase = data['phase']

    def neighbors(self, x, age, phase, excluded=None):
        eligible = np.flatnonzero((self.phase == phase) & (self.release != excluded))
        dis = np.linalg.norm(self.features[eligible]-feature(x,age), axis=1)
        order, used = [], set()
        for i in np.argsort(dis,kind='stable'):
            j = eligible[i]; uid = int(self.release[j])
            if uid not in used:
                # Exclusion is an evaluator operation for release cross-fitting,
                # never a required actor input.
                order.append((j,dis[i])); used.add(uid)
            if len(order) == K:
                break
        if len(order) < 4 or order[0][1] > 4.:
            return None
        ix, distance = map(np.asarray, zip(*order,strict=True))
        center = distance[len(distance)//2]
        raw = np.exp(-.5*((distance-distance[0])/max(.85,center*.72))**2)
        weight = raw/raw.sum()
        return self.residual[ix.astype(int)], weight, [int(self.release[int(j)]) for j in ix]

    def decisions(self, clinic, paths, ref, x, age, phase, excluded=None):
        if not clinic.sync():
            return None
        selection = self.neighbors(x, age, phase, excluded)
        if selection is None:
            return None
        errors, w, neighbors = selection
        t = clinic.twin
        p, v, s = t.ball.copy(),t.velocity.copy(),t.spin.copy()
        omega = t.data.qvel[t._ball_dof+3:t._ball_dof+6]
        center = w@errors
        # Retain nominal as a fixed reference. The sampled option uses actual
        # covariance across p/v/spin, not simultaneous same-sign extremes.
        settings = np.vstack((np.zeros(9),center,errors))
        options = []
        try:
            for adjustment in settings:
                t.ball[:] = p+adjustment[:3]
                t.velocity[:] = v+adjustment[3:6]
                omega[:] = s+adjustment[6:]
                _, _, f, cost, legal = school.native(clinic.predict, paths, ref)
                options.append((legal,np.clip(cost,-8.,16.),f[:,1]>.5))
        finally:
            t.ball[:]=p; t.velocity[:]=v; omega[:]=s
        hits,costs,contact = map(np.asarray,zip(*options,strict=True))
        # Mixture weights fixed before the cross-fit. Requiring feasibility
        # under one arbitrarily chosen nominal can veto true modes entirely.
        weights = np.r_[.13,.17,.70*w]
        return dict(probability=(weights @ hits).tolist(),
                    support=hits[2:].sum(axis=0).tolist(),
                    expected_loss=(weights @ costs).tolist(),
                    contact_probability=(weights @ contact).tolist(),
                    nominal=hits[0].tolist(), center=hits[1].tolist(),
                    donor_releases=neighbors, weights=w.tolist())


def fixed_tapes(n):
    if not MODEL.exists():
        fit()
    with np.load(MODEL) as model:
        if str(model['practice']) != _hash(bottleneck.FACTOR):
            raise ValueError('Changed frozen calibration')
    original = _read(maturity.OUT)
    raw = _read(bottleneck.FACTOR)
    if raw['source_sha256'] != _hash(maturity.OUT):
        raise ValueError('Changed measured practices')
    contract = dict(schema='tt60-release-crossfit-gate-1', calibration=_hash(bottleneck.FACTOR),
                    practice=_hash(maturity.OUT), model=_hash(MODEL), operator=_hash(Path(__file__)),
                    limits=list(LIMITS), complete=False, rows=[])
    if OUT.exists():
        old = _read(OUT)
        if any(old[k] != v for k,v in contract.items() if k not in ('rows','complete')):
            raise ValueError('Changed public gate audit')
        contract = old
    model = Gate(); cache={}
    with school.enabled():
        for old, prev in list(zip(original['rows'],raw['rows'],strict=True))[len(contract['rows']):n]:
            c=old['case']; src=c['source']; i=c['i']; tick=c['tick']
            if (c['id'],tick)!=(prev['id'],prev['tick']):
                raise ValueError('Mismatched calibrated action prefix')
            rec=dict(id=c['id'],tick=tick,lane=c['lane'])
            if 'arms' in prev:
                if src not in cache:
                    cache[src]=_read(Path(src))
                doc=cache[src]; row=doc['rows'][i]
                with school.mpc.centered(doc['ready']):
                    a=school.mpc.arena()
                    cam,rig,stride,lo,hi=school.mpc.resources(a)
                    engine=OpticalClinic(body=True)
                    try:
                        eye,x,q,ref,_,_,history=school.observed_prefix(a,cam,rig,doc['cards'][i],row,tick,a._arm_ids[1][2])
                        transfer.warm_clinic(engine,rig,a.control_dt,history,eye,tick,q,ref)
                        matches=[old['stock']]+[v for v in old.get('arms',[]) if v.get('controls')]
                        paths=np.array([bottleneck.goals_for(s,ref,stride,lo,hi)[0] for s in matches])
                        rec.update(tapes=prev['tapes'], old=prev['original_optical_gate'],
                                   new=model.decisions(engine,paths,ref,x,prev['optical']['freshness_s'],
                                                       prev['optical']['bounce'],excluded=c['id']))
                        if rec['new'] and c['id'] in rec['new']['donor_releases']:
                            raise ValueError('Release leakage in public uncertainty hypothesis')
                    finally:
                        engine.twin.close(); cam.close(); a.close()
            contract['rows'].append(rec)
            contract['complete']=len(contract['rows'])==len(raw['rows'])
            _write(OUT,contract)
            print(len(contract['rows']),rec['id'],[x['landed'] for x in rec.get('tapes',[])],
                  [] if rec.get('new') is None else np.round(rec['new']['probability'],2).tolist(),flush=True)


def summary():
    data=_read(OUT); rows=[r for r in data['rows'] if 'tapes' in r]
    doc=dict(gate_sha256=_hash(OUT), complete=data['complete'], observed=len(rows),
             landed=sum(x['landed'] for r in rows for x in r['tapes']),
             failed=sum(not x['landed'] for r in rows for x in r['tapes']),
             answered=sum(x['landed'] and x['answered'] and x['safe'] for r in rows for x in r['tapes']),
             old=dict(selected_landed=0,selected_failed=0), arms={})
    for r in rows:
        if 'selected' in r['old']:
            choice=r['tapes'][r['old']['selected']]
            doc['old']['selected_landed' if choice['landed'] else 'selected_failed']+=1
    for threshold in LIMITS:
        details=dict(accepted_landed=0,accepted_failed=0,selected_landed=0,
                     selected_failed=0,selected_safe_answered=0)
        for r in rows:
            if r['new'] is None:
                continue
            d=r['new']; candidates=[]
            for i,t in enumerate(r['tapes']):
                if d['probability'][i]>=threshold and d['support'][i]>=2:
                    details['accepted_landed' if t['landed'] else 'accepted_failed']+=1
                    candidates.append(i)
            if candidates:
                j=min(candidates,key=lambda j:(-d['probability'][j],d['expected_loss'][j]))
                t=r['tapes'][j]
                details['selected_landed' if t['landed'] else 'selected_failed']+=1
                details['selected_safe_answered']+=bool(t['landed'] and t['answered'] and t['safe'])
        doc['arms'][str(threshold)]=details
    _write(REPORT,doc)
    print(doc,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=('fit','audit','report'))
    p.add_argument('--n',type=int,default=40)
    a=p.parse_args()
    {'fit':fit,'audit':lambda:fixed_tapes(a.n),'report':summary}[a.mode]()
