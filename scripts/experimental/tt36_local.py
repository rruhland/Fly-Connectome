"""Compare camera credit and score-only local refinement across a few physical attempts."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np

from . import tt33_joint as joint
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_view_full_game import digest_json
from .tt29_rally_physics import measured_launches
from .tt31_curriculum import card, neutral_delivery
from .tt31_curriculum_opponent import replay_arena
from .tt31_curriculum_summary import compare_delivery
from .tt31_ready import centered
from .tt35_replies import varied_opponent
from .tt36_miss import BASE, CameraCredit, OUTPUT, ROOT


IDS = (6253, 6861, 12373, 3928, 4363, 12403)
PROBES = np.array([[0.18, 0., 0., 0., 0.], [0., .145, -.08, 0., 0.], [0., 0., 0., .15, .065]])
TRUST = (.115, .085)
LIMIT = .24
SCALE = np.array([.115, .14, .11, .85, 1.25, 1.7, .50, .55, .55, .55, .65])
MODES = ('fixed', 'dense', 'score')


def prepare(feeds, output):
    paths = (ROOT / 'tt34' / 'body-novel-stock.json.gz', ROOT / 'tt35' / 'confirm-nominal.json.gz')
    examples = {}
    for path in paths:
        document = _read(path)
        for c, row in zip(document['cards'], document['rows'], strict=True):
            if c['id'] in IDS:
                first = row['placements']['delivered'][0]
                if row['score']['learner'] or (first['near_contact'] is not None) != (c['id'] in IDS[3:]):
                    raise ValueError('A declared missed example changed')
                examples[c['id']] = (c, first['receive_line']['xyz'][0])
    shots = measured_launches(feeds)
    ready = _read(BASE)['ready']
    factors = []
    with centered(ready):
        arena = replay_arena([])
        try:
            for i, sid in enumerate(IDS):
                original, old_x = examples[sid]
                for j in range(30):
                    changed = card(sid, 'full', i, 362790 + 71*i + j)
                    offer = neutral_delivery(arena, shots, changed)
                    if offer and abs(offer['receive_line']['xyz'][0] - old_x) >= .06:
                        break
                else:
                    raise ValueError(f'Could not screen a different first ball for {sid}')
                factors.append(dict(original=original, changed=changed, changed_delivery=offer,
                    reply_seeds=[364001+31*i, 364002+31*i], practice_reply_seeds=[363001+101*i+k for k in range(11)]))
        finally:
            arena.close()
    _write(output/'local-plan.json.gz', dict(cases=factors, raw_sha256=_hash(BASE), feed_sha256=_hash(feeds),
           source_sha256={p.name:_hash(p) for p in paths}, ready=ready, cap=16))


class Approach:
    """Select a target residual from the current observation inside the near region."""

    def __init__(self, choose):
        self.choose = choose
        self.current = self.last = None
        self.blind = self.tick = 0
        self.picks = []

    def update(self, optical, q):
        tick = self.tick
        self.tick += 1
        if optical is None:
            self.blind += 1
            if self.blind >= 7:
                self.current = self.last = None
            return None
        self.blind = 0
        if self.last is not None and self.last < -.74 and optical[1] > 0.:
            self.current = None
        self.last = optical[1]
        if self.current is None and optical[1] < -.22:
            feature = np.r_[optical, q[:5]]
            self.current = self.choose(feature)
            self.picks.append(dict(tick=tick, observation=feature.tolist(), offset_rad=self.current.tolist()))
        return self.current if self.current is not None and np.any(self.current) else None


class StereoWitness:
    """Read RGB alongside play. Only its after-episode result enters practice."""

    def __init__(self, stereo, rig):
        self.camera = CameraCredit(rig)
        self.stereo = stereo
        self.tick = 0

    def capture(self):
        image = self.stereo.capture()
        self.camera.observe(image, self.tick)
        self.tick += 2
        return image

    def grade(self, onset):
        measure = self.camera.result(onset)
        if measure is None:
            return None
        tick = measure['encounter_tick']
        outgoing = None
        if measure['forward_samples']:
            samples = [pair for pair in self.camera.trace if tick <= pair[0] <= tick + 18]
            for (i, p, _), (j, q, _) in zip(samples, samples[1:]):
                if 0 < j - i <= 6 and q[1] - p[1] > .035:
                    velocity = (q - p) * 60 / (j - i)
                    outgoing = velocity.tolist()
                    break
        measure['observed_outgoing_mps'] = outgoing
        if outgoing is not None:
            v = np.asarray(outgoing)
            directional = max(0., v[1]) / max(.1, np.linalg.norm(v))
            measure['grade'] += .18 * directional + .018 * min(5., max(0., v[1]))
        return measure


def trial(arena, renderer, rig, stride, lo, hi, model, shots, entry, choose, reply_seed, progress=True):
    approach = Approach(choose)
    witness = StereoWitness(renderer, rig) if progress else renderer
    base = joint.JointPolicy

    class Taught(base):
        def command(self, optical, q):
            return super().command(optical, q, offset=approach.update(optical,q))

    arena.cards = iter([entry])
    with varied_opponent(entry['index'], reply_seed), patch.object(joint, 'JointPolicy', Taught):
        result = joint.point(arena, witness, rig, model, stride, lo, hi, shots[entry['id']], entry, cap=16)
    result['picks'] = approach.picks
    result['reply_seed'] = reply_seed
    if progress:
        result['optical'] = witness.grade(approach.picks[0]['tick']) if approach.picks else None
    return result


def value(row, mode):
    reward = row['score']['learner']
    if mode == 'score':
        return reward
    saw = row['optical']
    return reward + .42 * (saw['grade'] if saw is not None else -.6)


def winner(episodes, mode):
    return max(episodes, key=lambda e: (e['row']['score']['learner'], value(e['row'], mode),
                                       -np.linalg.norm(e['offset_rad'])))


def direction(episodes, baseline, mode, probe):
    x = np.asarray([e['offset_rad'] for e in episodes]) - baseline['offset_rad']
    y = np.asarray([value(e['row'], mode) for e in episodes]) - value(baseline['row'], mode)
    search = np.linalg.solve(x.T @ x + .025 * np.eye(5), x.T @ y)
    norm = np.linalg.norm(search)
    return .85 * search/norm + .15*probe if norm > 1e-7 else probe


def practice(feeds, output, index):
    path = output/'local-plan.json.gz'
    document = _read(path)
    case = document['cases'][index]
    original = case['original']
    model = _read(BASE)
    shots = measured_launches(feeds)
    forks = {}
    common = []
    probe_rng = np.random.default_rng(362424 + index)
    directions = probe_rng.choice([-1.,1.], size=(2,5)) / np.sqrt(5)
    with centered(model['ready']):
        arena = replay_arena([])
        renderer, rig, stride, lo, hi = joint.resources(arena)
        try:
            def play(offset, trial_index):
                seed = case['practice_reply_seeds'][trial_index]
                row = trial(arena, renderer, rig, stride, lo, hi, model, shots, original, lambda _:offset, seed)
                return dict(attempt=trial_index+1, offset_rad=offset.tolist(), row=row)

            initial = [np.zeros(5)] + [sign*axis for axis in PROBES for sign in (1,-1)]
            for number, offset in enumerate(initial):
                common.append(play(offset, number))
            for mode in ('dense','score'):
                attempts = common.copy()
                for step, probe in zip(TRUST,directions,strict=True):
                    current = winner(attempts, mode)
                    move = step * direction(attempts, current, mode, probe)
                    q = np.asarray(current['offset_rad'])
                    for sign in (1,-1):
                        attempts.append(play(np.clip(q + sign*move, -LIMIT, LIMIT), len(attempts)))
                best = winner(attempts, mode)
                forks[mode] = dict(added=attempts[len(common):], best_attempt=best['attempt'], best=best['offset_rad'],
                    first_conversion=next((e['attempt'] for e in attempts if e['row']['score']['learner']),None),
                    returns_by_attempt=[e['row']['score']['learner'] for e in attempts])
        finally:
            renderer.close()
            arena.close()
    _write(output/f'local-practice-{original["id"]}.json.gz', dict(card=original, plan_sha256=_hash(path),
           common=common,forks=forks))
    print(original['id'], {k:(r['first_conversion'],r['returns_by_attempt']) for k,r in forks.items()}, flush=True)


def fit(output):
    path = output/'local-plan.json.gz'
    plan = _read(path)
    observations = []
    for case in plan['cases']:
        entry = case['original']
        practice = output/f'local-practice-{entry["id"]}.json.gz'
        data = _read(practice)
        if data['card'] != entry or data['plan_sha256'] != _hash(path):
            raise ValueError('Mismatched practice deck')
        first = data['common'][0]['row']['picks'][0]['observation']
        for e in data['common'] + sum((f['added'] for f in data['forks'].values()), []):
            picks = e['row']['picks']
            if not picks or picks[0]['observation'] != first:
                raise ValueError('Starting optical context changed before the first correction')
        observations.append(dict(context=first, id=entry['id'], sha256=_hash(practice),
            outcomes={m: {k:v for k,v in data['forks'][m].items() if k != 'added'} for m in ('dense','score')}))
    _write(output/'local-checkpoint.json', dict(plan_sha256=_hash(path), base_sha256=_hash(BASE),
           scale=SCALE.tolist(), observations=observations))


def choose_from(checkpoint, mode):
    centers = np.asarray([v['context'] for v in checkpoint['observations']])
    options = np.asarray([v['outcomes'][mode]['best'] for v in checkpoint['observations']])
    scale = np.asarray(checkpoint['scale'])

    def choose(observed):
        distance = np.linalg.norm((centers - observed) / scale,axis=1)
        best = int(np.argmin(distance))
        return options[best] * np.exp(-.5*(distance[best]/1.25)**2) if distance[best] < 2.35 else np.zeros(5)

    return choose


def evaluate(feeds, output, mode):
    plan_path = output/'local-plan.json.gz'
    plan = _read(plan_path)
    state_path = output/'local-checkpoint.json'
    checkpoint = _read(state_path)
    if _hash(feeds) != plan['feed_sha256'] or _hash(BASE) != checkpoint['base_sha256'] or _hash(plan_path) != checkpoint['plan_sha256']:
        raise ValueError('Held input changed')
    choose = (lambda _: np.zeros(5)) if mode == 'fixed' else choose_from(checkpoint, mode)
    model = _read(BASE)
    shots = measured_launches(feeds)
    schedule = [("reply",c['original'],seed) for c in plan['cases'] for seed in c['reply_seeds']]
    schedule += [("first",c['changed'],365061+i) for i,c in enumerate(plan['cases'])]
    result = dict(ready=model['ready'], opening_side=1, phase=mode, cap=16, cards=[r[1] for r in schedule], rows=[],
                  schedule=[dict(kind=kind, reply_seed=seed) for kind,_,seed in schedule], checkpoint_sha256=_hash(state_path))
    with centered(model['ready']):
        arena = replay_arena([])
        renderer, rig, stride, lo, hi = joint.resources(arena)
        try:
            for part, entry, seed in schedule:
                row = trial(arena, renderer, rig, stride, lo, hi, model, shots, entry, choose, seed, progress=False)
                result['rows'].append(row)
                _write(output/f'local-held-{mode}.json.gz', result)
                print(mode, part, entry['id'], seed, row['score']['learner'], flush=True)
        finally:
            renderer.close()
            arena.close()


def summary(output):
    def signatures(rows, cards):
        return {f'{c["key"]}-{r["reply_seed"]}': next(iter(r['placements']['delivered']),None)
                for r,c in zip(rows,cards,strict=True)}

    paths = {m:output/f'local-held-{m}.json.gz' for m in MODES}
    docs = {m:_read(path) for m,path in paths.items()}
    frozen = docs['fixed']
    if any(d['cards'] != frozen['cards'] or d['schedule'] != frozen['schedule'] or len(d['rows']) != len(frozen['cards']) for d in docs.values()):
        raise ValueError('Held schedules are incomplete or differ')
    result = {}
    for kind in ('reply','first'):
        indices = [i for i,v in enumerate(frozen['schedule']) if v['kind']==kind]
        source = [frozen['rows'][i] for i in indices]
        cards = [frozen['cards'][i] for i in indices]
        results = {}
        for m,d in docs.items():
            rows = [d['rows'][i] for i in indices]
            outcomes = [r['score']['learner'] for r in rows]
            first = sum(bool(r['placements']['delivered']) and r['placements']['delivered'][0]['near_landed'] for r in rows)
            results[m] = dict(legal=sum(outcomes), per_point=outcomes, first=first, later=sum(outcomes)-first,
                              cpu_s=sum(r['runtime_cpu_s'] for r in rows),
                              first_offer=compare_delivery(signatures(rows,cards),signatures(source,cards)))
        result[kind] = dict(ids=[c['id'] for c in cards],scores=results)
    state = _read(output/'local-checkpoint.json')
    practice = []
    for item in state['observations']:
        data = _read(output/f'local-practice-{item["id"]}.json.gz')
        attempts = data['common'] + sum((f['added'] for f in data['forks'].values()), [])
        base = attempts[0]['row']['placements']['delivered'][0]
        for episode in attempts:
            actual = episode['row']['placements']['delivered'][0]
            if any(actual[k] != base[k] for k in ('target', 'far_contact', 'near_bounce')):
                raise ValueError('An initial practice offer changed across variants')
        practice.append(dict(item, exact_training_offers=len(attempts),
                             cpu_s=sum(e['row']['runtime_cpu_s'] for e in attempts)))
    _write(output/'local-summary.json', dict(held=result, practice=practice, hashes={m:_hash(p) for m,p in paths.items()}))
    print({k:{m:x['legal'] for m,x in v['scores'].items()} for k,v in result.items()}, flush=True)


def audit(output):
    """Independently replay each initial conversion and each transferred gain."""
    plan = _read(output/'local-plan.json.gz')
    selected = []
    for case in plan['cases']:
        entry = case['original']
        doc = _read(output/f'local-practice-{entry["id"]}.json.gz')
        heard = set()
        for mode, trace in doc['forks'].items():
            attempt = trace['first_conversion']
            if attempt is None:
                continue
            row = (doc['common'] + trace['added'])[attempt - 1]['row']
            identity = digest_json(row['actions'])
            if identity not in heard:
                selected.append((f'practice-{mode}-{attempt}', entry, row))
                heard.add(identity)
        if heard:
            selected.append(('practice-fixed', entry, doc['common'][0]['row']))
    fixed = _read(output/'local-held-fixed.json.gz')
    for mode in ('dense', 'score'):
        trained = _read(output/f'local-held-{mode}.json.gz')
        for entry, row, base in zip(trained['cards'], trained['rows'], fixed['rows'], strict=True):
            if row['score']['learner'] > base['score']['learner']:
                selected.append((f'held-{mode}', entry, row))
    receipts = []
    with TemporaryDirectory(prefix='tt36-local-') as temporary:
        temp = Path(temporary)
        for label, entry, row in selected:
            source = temp/'point.json.gz'
            result = temp/'cad.json'
            _write(source, dict(cards=[entry], rows=[row], ready=plan['ready'], opening_side=1))
            with varied_opponent(entry['index'], row['reply_seed']):
                joint.verify(source, result, cad=True)
            receipts.append(dict(condition=label, reply_seed=row['reply_seed'], point=_read(result)['points'][0]))
    _write(output/'local-audit.json', dict(checkpoint_sha256=_hash(output/'local-checkpoint.json'), replays=receipts))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('feeds', type=Path)
    p.add_argument('phase', choices=('prepare','practice','fit',*MODES,'summary','audit'))
    p.add_argument('--index', type=int, choices=range(len(IDS)), default=0)
    p.add_argument('--output',type=Path,default=OUTPUT)
    a = p.parse_args()
    a.output.mkdir(exist_ok=True,parents=True)
    if a.phase == 'prepare':
        prepare(a.feeds,a.output)
    elif a.phase == 'practice':
        practice(a.feeds,a.output,a.index)
    elif a.phase == 'fit':
        fit(a.output)
    elif a.phase == 'summary':
        summary(a.output)
    elif a.phase == 'audit':
        audit(a.output)
    else:
        evaluate(a.feeds,a.output,a.phase)


if __name__ == '__main__':
    main()
