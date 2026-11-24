"""Fit a raw pupil's joint correction from the visible outcomes of missed strokes."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from time import process_time

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt24_source_rally import blade_sighting
from .tt29_endurance import audit
from .tt29_fast_camera import native_ball_pixels
from .tt29_rally_physics import measured_launches
from .tt31_curriculum_opponent import replay_arena
from .tt31_curriculum_summary import compare_delivery, first_deliveries, held_difference
from .tt31_ready import centered
from .tt33_joint import FAR_COMMAND, JointPolicy, Vision, resources, verify
from .tt35_error import triangulate


ROOT = Path(__file__).resolve().parents[2] / 'artifacts'
OUTPUT = ROOT / 'tt36'
BASE = ROOT / 'tt33' / 'joint-raw.json.gz'
PRACTICE = ROOT / 'tt35' / 'search-practice.json.gz'
HELD = {'fresh': ROOT / 'tt33' / 'joint-fresh-raw.json.gz',
        'novel': ROOT / 'tt34' / 'body-novel-stock.json.gz'}
EXCLUDE = {5530, 14832, 3025, 8239}
SCALE = np.array([.21, .55, .16, 1.1, 1.8, 1.9, 1., 1., 1., 1., 1.])
MAX_CORRECTION = .105


class CameraCredit:
    """Observe actual stereo motion during replay, without extrapolating a flight."""

    def __init__(self, rig):
        self.rig = rig
        self.trace = []

    def observe(self, frames, tick):
        names = ('left', 'right')
        ball = [native_ball_pixels(frames[name]) for name in names]
        if any(x is None or x.clipped for x in ball):
            return
        p = triangulate([x.center for x in ball], self.rig)
        if p is None or not (-1.72 < p[1] < 1.57 and .61 < p[2] < 1.75 and abs(p[0]) < .97):
            return
        relative = None
        if p[1] < -.88:
            blades = [blade_sighting(frames[name]) for name in names]
            if all(b is not None for b in blades):
                center = triangulate([b[0] * frames[name].shape[0] / 360
                                      for name, b in zip(names, blades, strict=True)], self.rig)
                if center is not None and -1.67 < center[1] < -.82:
                    relative = p - center
        self.trace.append((tick, p, relative))

    def result(self, onset):
        indexes = [i for i, (s, p, e) in enumerate(self.trace) if onset <= s <= onset + 105
                   and e is not None and p[1] < -.88 and abs(e[1]) < .28]
        if not indexes:
            return None
        nearest = min(indexes, key=lambda i: np.linalg.norm(self.trace[i][2]))
        tick, position, relative = self.trace[nearest]
        burst = self.trace[nearest:]
        outgoing = []
        for a, b, c in zip(burst, burst[1:], burst[2:]):
            if a[0] > tick + 12 or a[1][1] > -.72:
                break
            # Sustained visible motion toward the opponent following the encounter.
            if b[0] - a[0] <= 6 and c[0] - b[0] <= 6 and min(b[1][1] - a[1][1], c[1][1] - b[1][1]) > .022:
                outgoing = [a, b, c]
                break
        if outgoing:
            last = next(i for i in range(nearest, len(self.trace)) if self.trace[i][0] == outgoing[-1][0])
            for following in self.trace[last + 1:]:
                if following[0] - outgoing[-1][0] > 8 or following[1][1] < outgoing[-1][1][1] - .022:
                    break
                outgoing.append(following)
        gap = float(np.linalg.norm(relative))
        farthest = max((p[1] for _, p, _ in outgoing), default=None)
        lateral = max((abs(p[0]) for _, p, _ in outgoing), default=0.)
        # A visible outward arc gets credit for forward reach even if no legal return is earned.
        value = -2. * min(gap, .4)
        if farthest is not None:
            value += .38 + .55 * np.clip((farthest + 1.1) / 2.1, 0., 1.) - max(0., lateral - .76)
        return dict(gap_xyz_m=relative.tolist(), gap_m=gap, encounter_tick=tick, encounter_xyz_m=position.tolist(),
                    visible_forward_y_m=float(farthest) if farthest is not None else None,
                    forward_samples=len(outgoing), grade=float(value))


def missing_points():
    source = _read(PRACTICE)
    held_ids = {c['id'] for p in HELD.values() for c in _read(p)['cards']}
    for batch in source['rounds']:
        for entry, rows in zip(batch['cards'], zip(*batch['rows'], strict=True), strict=True):
            if entry['id'] in EXCLUDE | held_ids:
                continue
            if any(row['score']['learner'] or len(row['placements']['delivered']) != 1 for row in rows):
                continue
            yield entry, rows, batch['candidates_rad']


def replay_eye(arena, renderer, rig, entry, row):
    arena.launch(row['shot'], card=entry)
    arena.partner.reset(seed=entry['seed'])
    eye = Vision(rig, arena.control_dt, raw=True)
    credit = CameraCredit(rig)
    context = onset = None
    for tick, u in enumerate(row['actions']):
        q = arena.joints
        image = renderer.capture() if tick % 2 == 0 else None
        sensed = eye.update(tick, image, q)
        if image is not None:
            credit.observe(image, tick)
        if context is None and sensed is not None:
            context = np.r_[sensed, q[:5]].tolist()
            onset = tick
        arena.exchange(u, FAR_COMMAND)
    if digest_json(row['events']) != digest_json(arena.events):
        raise ValueError('Replayed zero-score point changed')
    return context, credit.result(onset) if onset is not None else None


def collect(output):
    ready = _read(BASE)['ready']
    points = []
    started = process_time()
    with centered(ready):
        arena = replay_arena([])
        renderer, rig, _, _, _ = resources(arena)
        try:
            for entry, rows, biases in missing_points():
                traces = [replay_eye(arena, renderer, rig, entry, row) for row in rows]
                contexts = [context for context, _ in traces]
                if any(c != contexts[0] for c in contexts[1:]):
                    raise ValueError('First public observation changed before the compared motor commands')
                scores = [label for _, label in traces]
                points.append(dict(card=entry, context=contexts[0], biases_rad=biases, optical=scores, zero_scores=[0]*len(rows)))
                print(entry['id'], [x and round(x['grade'], 3) for x in scores], flush=True)
        finally:
            renderer.close()
            arena.close()
    _write(output / 'miss-practice.json.gz', dict(source_sha256=_hash(PRACTICE), base_sha256=_hash(BASE),
           practice_cards=points, only_zero_legal=True, process_cpu_s=process_time()-started))


def fit(output):
    path = output / 'miss-practice.json.gz'
    practice = _read(path)
    entries = []
    for point in practice['practice_cards']:
        if point['context'] is None:
            continue
        bias = np.asarray(point['biases_rad'])
        gradient = np.zeros(5)
        terms = []
        # Centered finite differences whenever both observations exist.
        for i, j in ((1, 2), (1, 0), (0, 2)):
            a, b = point['optical'][i], point['optical'][j]
            if a is None or b is None:
                continue
            direction = bias[i] - bias[j]
            benefit = a['grade'] - b['grade']
            if np.linalg.norm(direction) <= .015:
                continue
            gradient = benefit * direction / np.linalg.norm(direction)
            terms.append(dict(positive=i, negative=j, observed_advantage=benefit))
            break
        if terms:
            entries.append(dict(card=point['card'], context=point['context'], gradient=gradient.tolist(), terms=terms))
    if len(entries) < 4:
        raise ValueError(f'Too few matched visible misses ({len(entries)})')
    _write(output / 'miss-checkpoint.json', dict(schema='tt36-miss-1', source_sha256=_hash(path),
           base_sha256=_hash(BASE), observations=entries, scale=SCALE.tolist(), max_correction_rad=MAX_CORRECTION,
           only_zero_legal=True))
    print('fit', len(entries), [r['terms'] for r in entries], flush=True)


class MissCorrection:
    """Add a bounded motor residual using only the current raw observation and encoders."""

    def __init__(self, checkpoint, sign=1.):
        self.sign = sign
        self.scale = np.asarray(checkpoint['scale'])
        self.centers = np.asarray([r['context'] for r in checkpoint['observations']])
        self.gradients = np.asarray([r['gradient'] for r in checkpoint['observations']])
        self.limit = checkpoint['max_correction_rad']
        self.begin()

    def begin(self):
        self.last_depth = self.current = None
        self.blind = 0
        self.picks = []

    def offset(self, optical, q):
        if optical is None:
            self.blind += 1
            if self.blind >= 7:
                self.current = self.last_depth = None
            return None
        self.blind = 0
        if self.last_depth is not None and self.last_depth < -.72 and optical[1] > -.15:
            self.current = None
        if self.current is None:
            observed = np.r_[optical, q[:5]]
            weights = np.exp(-.5 * np.sum(np.square((self.centers - observed) / self.scale), axis=1))
            gradient = (weights @ self.gradients) / (.24 + weights.sum())
            # Smooth saturation limits the per-axis residual and vanishes with weak evidence.
            self.current = self.sign * self.limit * np.tanh(gradient / .055)
            self.picks.append(dict(observed=observed.tolist(), offset_rad=self.current.tolist()))
        self.last_depth = optical[1]
        return self.current if np.any(self.current) else None


def play(arena, renderer, rig, model, stride, lo, hi, shot, entry, corrector):
    arena.launch(shot, card=entry)
    arena.partner.reset(seed=entry['seed'])
    pupil = JointPolicy(model, stride, lo, hi)
    eye = Vision(rig, arena.control_dt, raw=True)
    corrector.begin()
    actions = []
    started = process_time()
    for tick in range(arena.max_steps):
        q = arena.joints
        x = eye.update(tick, renderer.capture() if tick % 2 == 0 else None, q)
        offset = corrector.offset(x, q)
        u = pupil.command(x, q, offset=offset) if offset is not None else pupil.command(x, q)
        _, _, done, capped, _ = arena.exchange(u, FAR_COMMAND)
        actions.append(u.tolist())
        if done or capped or arena.legal_returns + arena.rival_returns - 1 >= 16:
            break
    return dict(shot=shot, opponent_seed=entry['seed'], actions=actions, events=arena.events.copy(),
                score=audit(arena.events, opening_side=1), placements=arena.point_placements(),
                corrections=corrector.picks.copy(), motor_audit=arena.motor_audit(), runtime_cpu_s=process_time()-started)


def assess(feeds, output, mode):
    state_path = output / 'miss-checkpoint.json'
    state = _read(state_path)
    model = _read(BASE)
    if state['base_sha256'] != _hash(BASE):
        raise ValueError('Frozen raw policy changed')
    corrector = MissCorrection(state, sign={'fixed':0.,'learned':1.,'reversed':-1.}[mode])
    trained = set(model['training_release_ids']) | {r['card']['id'] for r in state['observations']}
    shots = measured_launches(feeds)
    with centered(model['ready']):
        arena = replay_arena([])
        renderer, rig, stride, lo, hi = resources(arena)
        try:
            for cohort, path in HELD.items():
                reference = _read(path)
                cards = reference['cards']
                if {c['id'] for c in cards} & trained:
                    raise ValueError('The selected held release appeared in fitting')
                report = dict(ready=model['ready'], cards=cards, opening_side=1, cap=16, rows=[],
                              checkpoint_sha256=_hash(state_path), base_sha256=_hash(BASE), mode=mode, cohort=cohort,
                              reference_sha256=_hash(path))
                for i, entry in enumerate(cards):
                    row = play(arena, renderer, rig, model, stride, lo, hi, shots[entry['id']], entry, corrector)
                    if mode == 'fixed' and any(digest_json(row[k]) != digest_json(reference['rows'][i][k])
                                               for k in ('actions','events','placements')):
                        raise ValueError(f'Frozen nominal controller diverged at {entry["id"]}')
                    report['rows'].append(row)
                    _write(output / f'miss-{cohort}-{mode}.json.gz', report)
                    print(mode, cohort, entry['id'], row['score']['learner'], row['score']['returns'], flush=True)
        finally:
            renderer.close()
            arena.close()


def stats(rows):
    first = sum(bool(r['placements']['delivered']) and r['placements']['delivered'][0]['near_landed'] for r in rows)
    returns = [r['score']['learner'] for r in rows]
    return dict(arm=sum(returns), first=first, later=sum(returns)-first, points=returns,
                alternating=sum(r['score']['returns'] for r in rows), cpu_s=sum(r['runtime_cpu_s'] for r in rows))


def summarize(output):
    groups = {}
    for cohort, reference in HELD.items():
        paths = {k:output/f'miss-{cohort}-{k}.json.gz' for k in ('fixed','learned','reversed')}
        docs = {k:_read(p) for k,p in paths.items()}
        expected = _read(reference)['cards']
        if any(d['cards'] != expected or len(d['rows']) != len(expected) for d in docs.values()):
            raise ValueError('Held results are incomplete or contain different offers')
        fixed = docs['fixed']['rows']
        groups[cohort] = dict(ids=[c['id'] for c in expected], sha256={k:_hash(p) for k,p in paths.items()},
            scores={k: stats(d['rows']) for k,d in docs.items()}, paired={k:held_difference(d['rows'],fixed) for k,d in docs.items()},
            first_offer={k:compare_delivery(first_deliveries(d['rows']),first_deliveries(fixed)) for k,d in docs.items()})
    _write(output/'miss-summary.json', dict(checkpoint_sha256=_hash(output/'miss-checkpoint.json'), held=groups))
    print({k:d['scores'] for k,d in groups.items()}, flush=True)


def audit_changed(output):
    """Replay changed-score games with both stereo credit and native CAD checks."""
    cards, tapes = [], []
    for cohort in HELD:
        docs = {k: _read(output / f'miss-{cohort}-{k}.json.gz') for k in ('fixed', 'learned', 'reversed')}
        for i, entry in enumerate(docs['fixed']['cards']):
            rows = {k: d['rows'][i] for k, d in docs.items()}
            if len({r['score']['learner'] for r in rows.values()}) > 1:
                cards.append(entry)
                tapes.append(rows)
    selection = 'All games with a different score'
    if not cards:
        docs = {k: _read(output / f'miss-novel-{k}.json.gz') for k in ('fixed', 'learned', 'reversed')}
        i = max(range(len(docs['learned']['rows'])), key=lambda j: max(
            (max(abs(x) for x in c['offset_rad']) for c in docs['learned']['rows'][j]['corrections']), default=0.))
        cards.append(docs['fixed']['cards'][i])
        tapes.append({k: d['rows'][i] for k, d in docs.items()})
        selection = 'Maximum learned offset among the 16 novel games, all scores tied'
    ready = _read(BASE)['ready']
    results = []
    with centered(ready):
        arena = replay_arena([])
        renderer, rig, _, _, _ = resources(arena)
        try:
            for entry, rows in zip(cards, tapes, strict=True):
                measures = {k: replay_eye(arena, renderer, rig, entry, row)[1] for k, row in rows.items()}
                results.append(dict(id=entry['id'], seed=entry['seed'], optical=measures,
                    arm_returns={k: row['score']['learner'] for k, row in rows.items()}))
        finally:
            renderer.close()
            arena.close()
    with TemporaryDirectory(prefix='tt36-miss-') as location:
        temp = Path(location)
        geometry = {}
        for mode in ('fixed','learned','reversed'):
            source = temp / 'point.json.gz'
            result = temp / 'cad.json'
            _write(source, dict(cards=cards, rows=[t[mode] for t in tapes], ready=ready, opening_side=1))
            verify(source, result, cad=True)
            geometry[mode] = _read(result)
    tapes_hash = {f'{cohort}-{mode}': _hash(output / f'miss-{cohort}-{mode}.json.gz')
                  for cohort in HELD for mode in ('fixed','learned','reversed')}
    _write(output / 'miss-audit.json', dict(selection=selection, source_sha256=tapes_hash, points=results, cad=geometry))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('feeds', type=Path)
    p.add_argument('mode', choices=('collect','fit','fixed','learned','reversed','summary','audit'))
    p.add_argument('--output', type=Path, default=OUTPUT)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode in ('collect','fit','summary','audit'):
        {'collect':collect,'fit':fit,'summary':summarize,'audit':audit_changed}[args.mode](args.output)
    else:
        assess(args.feeds, args.output, args.mode)


if __name__ == '__main__':
    main()
