"""Train on delayed stereo observations, then audit on held real RGB and evaluator-only spin."""

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt37_rebound import events
from .tt38_flight_fit import FlightFit
from .tt39_spin import Estimate, MODES, Observer, solve, velocity_fit


OUT = Path("artifacts/tt39")
TT37 = Path("artifacts/tt37")
RATES = (35., 70., 120.)
FRICTIONS = (.035, .065, .095, .145, .21)


def opening(path):
    with np.load(path) as f:
        return f['sensor'], f['groups'], float(f['period']) if 'period' in f else 1/30


def config(sensor, period):
    bounce = events(dict(sensor=sensor, period=period))
    ratio = (1. - bounce['rebound_delta'][:, 2] / bounce['rebound_states'][:, 2])
    return dict(height=float(np.median(bounce['rebound_z'])), restitution=float(np.median(ratio)),
                spin_prior=70., friction=.065), len(ratio)


def delayed(sensor, groups, i, period):
    """Label a prospective decision using *later* uninterrupted RGB, not a hidden track."""
    window = max(4, round(.10 / period))
    stop = i + window + 1
    if stop > len(sensor) or not np.all(groups[i:stop] == groups[i]):
        return None
    s = sensor[i:stop]
    if not np.isfinite(s[:, :4]).all() or not np.allclose(np.diff(s[:, 0]), period):
        return None
    t = s[:, 0] - s[0, 0]
    p = s[:, 1:4]
    v = velocity_fit(t[1:], p[1:])
    if v[1] > -.95:
        return None
    acc = np.array([0., 0., -9.81]) - .133 * np.linalg.norm(v) * v
    corrected = p - .5 * t[:, None] ** 2 * acc
    e = corrected - corrected[1:].mean(axis=0) - (t - t[1:].mean())[:, None] * v
    # Reject a future bounce, near racket reversal or occlusion from the prediction label.
    if np.linalg.norm(e, axis=1).max() > .007 or np.min(p[:, 2]) < .779:
        return None
    h = min(window-1, max(2, round(.0667 / period)))
    return float(s[h, 0]), p[h].copy(), v + t[h] * acc


def earlier(observer, sensor, groups, period, config, legacy=None):
    cases = []
    for i, (s, g) in enumerate(zip(sensor, groups, strict=True)):
        if not i or np.any(g != groups[i-1]):
            observer.reset()
            if legacy is not None:
                legacy.last_time = None
        now, point = s[0], s[1:4]
        observer.ingest(now, point)
        standard = None
        if legacy is not None:
            base = s[4:10] if np.isfinite(s[4:10]).all() else None
            saved = legacy.update(now, point, base)
            if saved is not None and np.isfinite(saved).all():
                standard = saved.copy(), legacy.rotation(now)
        if not np.isfinite(point).all() or not -.05 > point[1] > -1.56 or i % 2:
            continue
        label = delayed(sensor, groups, i, period)
        if label is None:
            continue
        packet = observer.evidence()
        if packet is None:
            continue
        a, b, p, stamp, impact = packet
        phase = 'descent' if impact is None else ('rebound' if now-impact < .083 else 'settled')
        if impact is None and velocity_fit(a, b)[2] > .2:
            phase = 'independent-rise'
        cases.append((i, int(g[0]), phase, packet, label, standard))
    return cases


def stat(x, g):
    _, inv, counts = np.unique(g, return_inverse=True, return_counts=True)
    w = 1. / counts[inv]
    return (w / w.sum()) @ x


def predict(case, settings, mode, now):
    _, _, _, packet, label, standard = case
    t, p, pack, stamp, impact = packet
    if mode.startswith('legacy'):
        if standard is None:
            return None
        x, w = standard
        posterior = np.zeros(3)
        if mode == 'legacy-zero':
            w = np.zeros(3)
        elif mode != 'legacy':
            local = solve(t, p, pack, settings, mode.removeprefix('legacy-'), stamp, impact)
            w = local.state(now)[1]
            posterior = local.posterior
        estimate = Estimate(now, None, x[:3], x[3:], None, w, True, (x[3:],), 0., posterior)
    else:
        estimate = solve(t, p, pack, settings, mode, stamp, impact)
    return estimate, estimate.state(now), estimate.state(label[0])[0]


def practice(sensor, cases, settings):
    rows = []
    for mode in ('arc','coupled'):
        frictions = FRICTIONS if mode == 'coupled' else (settings['friction'],)
        for prior in RATES:
            for k in frictions:
                config = dict(settings, spin_prior=prior, friction=k)
                errors, groups = [], []
                for c in cases:
                    i, g, phase, _, (_, target, future_v), _ = c
                    if phase != 'rebound' and phase != 'settled':
                        continue
                    _, _, future = predict(c, config, mode, sensor[i, 0])
                    acc_loss = np.linalg.norm(future[:3] - target)
                    errors.append(acc_loss)
                    groups.append(g)
                rows.append(dict(mode=mode, phase='post-reversal', loss_m=float(stat(errors, groups)),
                                 spin_prior=prior, friction=k))
    return rows


def evaluate(sensor, evaluation, cases, settings, tuned):
    rows = defaultdict(list)
    for case in cases:
        i, g, phase, pkt, label, standard = case
        now = sensor[i, 0]
        actual_pv, angular = evaluation[i, :6], evaluation[i, 6:9]
        # Strictly evaluator-only independent check, after all model parameters are frozen.
        if actual_pv[4] > -.95:
            continue
        error_axis = actual_pv[3:6] / np.linalg.norm(actual_pv[3:6])
        for mode in (*MODES, 'legacy', 'legacy-zero', 'legacy-arc', 'legacy-coupled'):
            local = tuned.get(mode.removeprefix('legacy-'),
                              tuned.get('coupled', settings) if mode == 'contact0' else settings)
            result = predict(case, local, mode, now)
            if result is None:
                continue
            e, (pv, w), future = result
            dw = w-angular
            values = np.r_[np.linalg.norm(future[:3]-label[1]),
                           np.linalg.norm(future[3:]-label[2]),
                           np.linalg.norm(pv[:3]-actual_pv[:3]),
                           np.linalg.norm(pv[3:]-actual_pv[3:]),
                           dw, dw @ error_axis, np.linalg.norm(np.cross(dw, error_axis)),
                           e.posterior, w, angular]
            rows[(phase, mode)].append((i, g, values))
    result = {}
    for (phase, mode), items in rows.items():
        q = np.stack([a for _, _, a in items])
        g = [g for _, g, _ in items]
        error = stat(np.abs(q[:, :12]), g)
        # Projecting along actual travel is strictly an audit. The deployed fit has no direction label.
        result.setdefault(phase, {})[mode] = dict(
            samples=len(q), first_releases=len(np.unique(g)), future_position_mean_mm=1000.*error[0],
            future_velocity_mean_mps=error[1], live_position_mean_mm=1000.*error[2], live_velocity_mean_mps=error[3],
            spin_mae_xyz_rad_s=error[4:7].tolist(), spin_parallel_mae_rad_s=error[7],
            spin_transverse_mean_error_rad_s=error[8], posterior_sd_rad_s=error[9:12].tolist(),
            spin_corr_xyz=[float(np.corrcoef(q[:, 12+j],q[:, 15+j])[0,1])
                           if np.std(q[:,12+j])>.00001 and np.std(q[:,15+j])>.00001 else None for j in range(3)])
    intervals = {}
    rng = np.random.default_rng(390767)
    for phase in sorted({key[0] for key in rows}):
        for name, control in (('arc','zero'), ('coupled','contact0'), ('legacy-arc','legacy-zero'),
                              ('legacy-coupled','legacy-zero'), ('legacy','legacy-zero')):
            a = {i:(g,v[0]) for i,g,v in rows[phase,name]}
            b = {i:v[0] for i,_,v in rows[phase,control]}
            groups = defaultdict(list)
            for i in sorted(a.keys() & b.keys()):
                g,x = a[i]
                groups[g].append(1000.*(x-b[i]))
            if groups:
                means = [np.mean(groups[g]) for g in sorted(groups)]
                dem = rng.choice(means, size=(2500,len(means))).mean(axis=1)
                intervals.setdefault(phase,{})[name+'_minus_'+control] = np.quantile(dem,[.025,.975]).tolist()
    return result, intervals


def run(train, held, legacy, checkpoint, output):
    ts, train_groups, train_period = opening(train)
    hs, held_groups, held_period = opening(held)
    for index in range(3):
        if set(train_groups[:, index]) & set(held_groups[:, index]):
            raise ValueError('Development and held groups overlap')
    settings, bounces = config(ts, train_period)
    obs = Observer(train_period, settings)
    practice_cases = earlier(obs, ts, train_groups, train_period, settings)
    sweep = practice(ts, practice_cases, settings)
    tuned = {mode: dict(settings, **{k: best[k] for k in ('spin_prior','friction')})
             for mode in ('arc','coupled') for best in [
                 min((m for m in sweep if m['mode'] == mode), key=lambda item: item['loss_m'])]}
    input_hash = dict(practice=_hash(train), held=_hash(held), legacy=_hash(legacy))
    _write(checkpoint, dict(schema='tt39-visual-spin-v1', optics=input_hash, settings=settings, tuned=tuned))
    old = FlightFit(legacy)
    old.period = held_period
    cases = earlier(Observer(held_period, settings), hs, held_groups, held_period, settings, legacy=old)
    # The learner and tuning paths have used only the public array above.
    with np.load(held) as archived:
        evaluator = archived['evaluation']
    results, intervals = evaluate(hs, evaluator, cases, settings, tuned)
    receipt = dict(schema='tt39-spin-diagnostic-v1', input_hash=input_hash, tuning_dispatch=sweep,
                   settings=settings, tuned=tuned, optical_practice_bounces=bounces,
                   practice_cases=len(practice_cases), held_cases=len(cases),
                   held_results=results, paired_release_bootstrap_95_mm=intervals,
                   periods_s=dict(practice=train_period, held=held_period))
    _write(output, receipt)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--train', type=Path, default=TT37/'hz60-practice.npz')
    p.add_argument('--held', type=Path, default=TT37/'hz60-diagnosis.npz')
    p.add_argument('--legacy', type=Path, default=TT37/'hz60-rebound.npz')
    p.add_argument('--checkpoint', type=Path, default=OUT/'spin-checkpoint.json')
    p.add_argument('--output', type=Path, default=OUT/'spin-diagnostic.json')
    a = p.parse_args()
    a.output.parent.mkdir(exist_ok=True, parents=True)
    run(a.train,a.held,a.legacy,a.checkpoint,a.output)
