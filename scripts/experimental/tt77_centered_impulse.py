"""Use centered arm failures to learn a different outgoing impulse on a real low ball."""

import argparse
from pathlib import Path

import numpy as np
from scipy.stats import qmc

from . import tt72_retarget_trajectory as retarget
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import Rollout
from .tt4_transfer_trial import _hash, _read, _write


ROOT = Path('artifacts/tt77_centered_impulse')
PLAN = ROOT / 'protocol.json'
TRAIN = ROOT / 'development.json.gz'
mpc, qual, tape = retarget.mpc, retarget.qual, retarget.tape
ZERO, DT = retarget.ZERO, retarget.DT
# Controller offsets in dissipative stock actuator coordinates. The last
# dimension is a bounded advance of the attained trajectory before contact.
MAG = np.array([.20, .28, .24, .21, .22, .18, .25, .24, .20, .23,
                .12, .19, .14, .17, .15, 3.5])
LARGE = 96
ROUNDS = 6
LIMIT = .058
LONG = .38
SOFT = 4.35


def freeze():
    ROOT.mkdir(parents=True, exist_ok=True)
    prior = retarget.ROOT / 'equivalence.json.gz'
    target = dict(schema='tt77-centered-impulse-1', case_order=[9254],
                  sources={str(p): _hash(p) for p in (retarget.DONORS, retarget.STOCK, prior,
                              retarget.ROOT / 'stock-witnesses.json.gz')},
                  training=dict(trials=LARGE, rounds=ROUNDS, variable_scale=MAG.tolist(), native_per_round=4,
                                meet_advance_ticks=[0, 2, 4, 6], long_far_y_m=LONG, soft_departure_y_m_s=SOFT),
                  criteria='Exact physical incoming. Actual clean interior face, legal far landing at y>=.38 m, '
                           'outgoing forward speed <=4.35 m/s, and an ordinary native answer. Two unchanged '
                           'physical repeats before transfer training. No lateral pose or tangential twist objective. '
                           'The world simulator supplies development feedback only. Assess with a frozen RGB/encoder learner '
                           'on new releases only after a legal independent practice tape.')
    if not PLAN.exists():
        _write(PLAN, target)
    if _read(PLAN) != target:
        raise ValueError('Changed centered impulse protocol')
    return target


def refs(a, u, tick, frames):
    """Reconstruct absolute references from actually issued actions."""
    axes = a._arm_ids[1][2]
    stride = a._joint_increments[axes] * DT / a.model.opt.timestep
    lo, hi = a._joint_low[axes], a._joint_high[axes]
    ref = a.data.ctrl[axes].copy()
    T = [ref]
    for cmd in [*u[tick:], *([ZERO] * frames)][:frames]:
        ref = np.clip(ref + stride * cmd, lo, hi)
        T.append(ref)
    return np.asarray(T[1:])


def centers(a, donor):
    output = []
    original = _read(retarget.STOCK)
    alt = _read(retarget.ROOT / 'equivalence.json.gz')
    f = donor['fork']
    for d in original, alt:
        row = next(r for r in d['rows'] if r['case']['id'] == donor['case']['id'])
        candidates = [r for r in row['physically_tested'] if r['contact'] is not None and
                      r['depart'] and not r['clearance']['penetration_samples']]
        # Reobserve the real centered locations with full-face near contact,
        # then a second class of contacts with some forward speed.
        candidates.sort(key=lambda r: (r['contact']['radial_m'] > LIMIT,
                        -r['depart'][1] - .18 * r['depart'][2]))
        for hit in candidates[:3]:
            k = hit['contact']['time_s'] / DT - f
            us = np.asarray(hit['commands'])
            output.append(dict(ref=refs(a, us, f, 120), at=k,
                    actual_depart=hit['depart'], contact=hit['contact'], actual_events_sha256=digest_json(hit['events'])))
    return output


def smooth(s):
    p = np.clip(s, 0., 1.)
    return p * p * (3. - 2. * p)


def schedules(bases, selected, controls):
    """Advance achieved preparation and teach separate impulse bends about the strike."""
    out = []
    for index, raw in zip(selected, controls, strict=True):
        b = bases[index]
        x = raw * MAG
        ref = b['ref']
        ticks = np.arange(len(ref))
        # The start of all trajectories remains the physically observed stock
        # neutral fork. The late swing may occur up to six ticks earlier.
        advance = np.clip(x[-1], -2., 6.)
        at = b['at']
        phase = np.clip(ticks + advance * smooth((ticks - at + 29.) / 16.), 0., len(ref) - 1.)
        moved = np.stack([np.interp(phase, ticks, q) for q in ref.T], axis=1)
        prep = smooth((ticks - at + 31.) / 17.) * (1. - smooth((ticks - at + 5.) / 14.))
        strike = smooth((ticks - at + 17.) / 11.) * (1. - smooth((ticks - at - 4.) / 13.))
        twist = np.clip((ticks - at + 1.) / 6., -1., 1.) * strike
        moved[:, :5] += prep[:, None] * x[10:15] + strike[:, None] * x[:5] + twist[:, None] * x[5:10]
        out.append(moved)
    return np.asarray(out)


def output_target(p, vy):
    # A soft legal arc is a band of impulses, not a replica of the lateral
    # stroke. Aim 0.76 m into the far court, with elapsed time adjusted for
    # actual separation depth and linear drag's effect on horizontal flight.
    y = np.maximum(-p[:, 1], .15)
    time = np.clip((y + .76) / (np.clip(vy, 2.65, 4.1) * .91), .32, .87)
    net_t = np.clip(y / (np.clip(vy, 2.65, 4.1) * .95), .12, .75)
    vz_landing = (.775 - p[:, 2]) / time + 4.905 * time
    vz_net = (1.00 - p[:, 2]) / net_t + 4.905 * net_t
    return -.64 * p[:, 0], np.clip(vy, 2.8, 3.90), np.maximum(vz_net, vz_landing)


def feedback(scores, fb):
    p, v = fb[:, 5:8], fb[:, 2:5]
    tx, ty, tz = output_target(p, v[:, 1])
    connected = fb[:, 1] > 0.5
    errors = np.c_[v[:, 0] - tx, v[:, 1] - ty, v[:, 2] - tz, np.maximum(fb[:, 8] - .049, 0.)]
    miss = 10. + 21. * np.minimum(fb[:, 0], 1.)
    weak = 2. + np.linalg.norm(errors * [1.1, 1.6, 1.45, 31.], axis=1)
    legal = np.isfinite(scores[:, 0])
    interior = legal & (scores[:, 12] <= LIMIT)
    earned = interior & (scores[:, 4] >= LONG) & (v[:, 1] <= SOFT)
    quality = .51 * np.abs(scores[:, 4] - .76) + .68 * np.abs(scores[:, 3]) + .26 * np.abs(v[:, 1] - 3.4)
    quality += 6. * np.maximum(scores[:, 12] - .050, 0.) + 2. * np.maximum(.04 - scores[:, 5], 0.)
    cost = np.where(legal, quality - 4. - 2. * interior - 3. * earned, np.where(connected, weak, miss))
    return cost, connected, errors, legal, interior, earned


def perturbations(bases, seed):
    rng = np.random.default_rng(seed)
    sobol = qmc.Sobol(len(MAG), scramble=True, seed=seed).random_base2(7)[:LARGE]
    pars = (2. * sobol - 1.) * .95
    labels = np.arange(LARGE) % len(bases)
    j = 0
    for i in range(len(bases)):
        for u in (0., 2., 4., 6.):
            pars[j] = 0.
            pars[j, -1] = u / MAG[-1]
            labels[j] = i
            j += 1
    return pars, labels, rng


def proposal(pars, labels, rank, errors, connected, rng, k):
    top = rank[:16]
    select = np.r_[top, rng.choice(top, size=LARGE - len(top), replace=True)]
    x = pars[select].copy()
    scale = (.56, .51, .45, .36, .27, .20)[k]
    x[len(top):] += rng.normal(size=(LARGE - len(top), len(MAG))) * scale
    # Learn a local signed map from actual modeled separation failures in
    # the same attained stock branch, rather than throw out their impulse.
    j = len(top)
    for i in np.unique(labels[top]):
        ix = [z for z in rank if connected[z] and labels[z] == i]
        if len(ix) < 20:
            continue
        ix = np.asarray(ix[:32])
        origin = pars[ix[0]]
        delta = pars[ix] - origin
        J = np.linalg.solve(delta.T @ delta + .12 * np.eye(len(MAG)), delta.T @ (errors[ix] - errors[ix[0]]))
        step = np.linalg.solve(J @ J.T + .20 * np.eye(len(MAG)), -J @ errors[ix[0]])
        for gain in (.60, 1., 1.35):
            x[j] = origin + gain * np.clip(step, -.75, .75)
            select[j] = ix[0]
            j += 1
    return np.clip(x, -2.1, 2.1), labels[select]


def actual_soft(x):
    if not qual.received(x, True) or x['depart'] is None:
        return False
    far = [e['ball'][1] for e in x['events'] if e['kind'] == 'legal_return' and e.get('side') == 1
           and e['time'] > x['contact']['time_s']]
    return bool(far and far[0] >= LONG and x['depart'][1] <= SOFT)


def run():
    spec = freeze()
    d = _read(retarget.DONORS)['rows'][0]
    if d['case']['id'] != spec['case_order'][0]:
        raise ValueError('Unregistered centered training shot')
    signatures = dict(protocol=_hash(PLAN), case=d['case'])
    out = _read(TRAIN) if TRAIN.exists() else dict(**signatures, rounds=[], complete=False)
    if any(out.get(k) != v for k, v in signatures.items()):
        raise ValueError('Changed centered development contract')
    if out['complete']:
        return
    ready, xy = retarget.receipt()
    c, row, anchor = qual.get(_read(Path(d['case']['source'])), d['case'])
    origin = next(x for x in xy[c['id'], .12]['motion']['physical'] if qual.received(x, True))
    with retarget.lateral.fixture.school.enabled(cameras=False), mpc.centered(ready[0.]):
        a, cad = mpc.arena(width=64, height=48), mpc.arena(width=64, height=48, detail='cad')
        guard = qual.GripClearance(cad.model, stock=True)
        try:
            def fork():
                tape.setup(a, c, row, anchor)
                for zero in prefix:
                    a.exchange(zero, mpc.FAR_COMMAND)
            prefix = [ZERO] * d['fork']
            fork()
            bases = centers(a, d)
            if not bases:
                raise ValueError('No actual achieved stock training motion')
            pars, labels, rng = perturbations(bases, 7509254)
            rolling = Rollout(a, 1)
            if out['rounds']:
                last = out['rounds'][-1]
                pars, labels = np.asarray(last['next_params']), np.asarray(last['next_labels'])
                rng.bit_generator.state = last['next_rng']
            physical_hash = {v['proposal_sha256'] for rr in out['rounds'] for v in rr['physical']}
            for k in range(len(out['rounds']), ROUNDS):
                fork()
                goals = schedules(bases, labels, pars)
                fb = np.zeros((len(pars), 9))
                moves, y = rolling.trajectories(goals, qual.AIM, 1.08, feedback=fb)
                loss, connect, errors, legal, interior, earned = feedback(y, fb)
                rank = np.argsort(loss)
                tests = []
                for j in rank:
                    if len(tests) == spec['training']['native_per_round']:
                        break
                    if not connect[j]:
                        continue
                    length = int(np.ceil(bases[labels[j]]['at'])) + 19
                    cmds = prefix + moves[j, :length].tolist() + [ZERO] * 155
                    uid = digest_json(cmds)
                    if uid in physical_hash:
                        continue
                    physical_hash.add(uid)
                    actual = tape.matched_motion(a, c, row, anchor, cmds, guard)
                    if not actual['same'] or not retarget.first_ball(actual, origin['placements']['delivered'][0]):
                        raise ValueError('Impulse practice changed the real incoming delivery')
                    actual.update(variant=int(j), genotype=pars[j].tolist(), stock_seed=int(labels[j]), proposal_sha256=uid,
                                  model_outgoing=fb[j, 2:5].tolist(), model_radius=float(fb[j, 8]),
                                  model_legal=bool(legal[j]), model_soft=bool(earned[j]), actual_soft=actual_soft(actual))
                    if actual['actual_soft']:
                        repeat = tape.matched_motion(a, c, row, anchor, actual['commands'], guard)
                        if digest_json(actual['events']) != digest_json(repeat['events']) or not actual_soft(repeat):
                            raise ValueError('Centered practice reward failed exact physical replay')
                        actual['repeat_sha256'] = digest_json(repeat['events'])
                    # Actual failure corrects the modeled outgoing residual
                    # at the executed action before credit is assigned.
                    if actual['depart'] is not None and actual['contact'] is not None:
                        e = next(ev for ev in actual['events'] if ev['kind'] == 'separation'
                                 and ev.get('collider') == 'learner_racket'
                                 and ev['time'] > actual['contact']['time_s'])
                        v = np.asarray(actual['depart'])
                        xyz = np.asarray(e['ball'])
                        vx, vy, vz = (z[0] for z in output_target(xyz[None, :], np.array([v[1]])))
                        errors[j] = np.r_[v - [vx, vy, vz], max(actual['contact']['radial_m'] - .049, 0.)]
                    if actual['clearance']['penetration_samples'] or (legal[j] and not actual['landed']):
                        loss[j] = max(loss[j], 6.)
                    tests.append(actual)
                    print('stock practice', k, int(j), actual['landed'], actual['answered'], actual['actual_soft'],
                          actual['depart'], None if actual['contact'] is None else actual['contact']['radial_m'],
                          actual['clearance']['penetration_samples'], flush=True)
                    if actual['actual_soft']:
                        break
                entry = dict(round=k, model_touches=int(connect.sum()), model_legal=int(legal.sum()),
                             model_interior=int(interior.sum()), model_long_soft=int(earned.sum()),
                             best_model_loss=float(loss[rank[0]]), physical=tests,
                             best_unrealized=[[int(j), pars[j].tolist(), int(labels[j]), errors[j].tolist(), fb[j].tolist()]
                                              for j in rank[:12]])
                pars, labels = proposal(pars, labels, np.argsort(loss), errors, connect, rng, k)
                entry.update(next_params=pars.tolist(), next_labels=labels.tolist(), next_rng=rng.bit_generator.state)
                out['rounds'].append(entry)
                out['winner'] = next((x['repeat_sha256'] for x in tests if x['actual_soft']), None)
                out['complete'] = bool(out['winner']) or k + 1 == ROUNDS
                _write(TRAIN, out)
                print('practice round', k, 'touch/legal/interior/soft', *[entry[q] for q in (
                    'model_touches', 'model_legal', 'model_interior', 'model_long_soft')], flush=True)
                if out['complete']:
                    break
        finally:
            a.close()
            cad.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('freeze', 'dev'))
    {'freeze': freeze, 'dev': run}[p.parse_args().stage]()
