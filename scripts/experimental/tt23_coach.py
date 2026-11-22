"""Learn which physically legal far strokes the optical learner can return."""

from collections import Counter

import numpy as np
from scipy.special import expit

from .tt4_arena import AIR_DRAG, GRAVITY, MAGNUS_COEFFICIENT, TABLE_HEIGHT
from .tt4_transfer_trial import _hash, _read, _write
from .tt22_ordinary_game import DIR


OUTPUT = DIR.parent/'tt23'
PRACTICE = OUTPUT/'coach-practice.json.gz'
INPUTS = {
    'ordinary-game.json.gz': ('frozen', 'online'),
    'ordinary-zero-practice.json.gz': ('rows',),
    'flight-inverse-held.json.gz': ('rows',),
}
RIDGES = (1., 3., 10.)
FEATURES = {
    'geometry': 3,
    'flight': 8,
    'spin': 12,
}


def approach(separation, contact, target):
    """Describe the outgoing flight at separation from the learner's table."""
    position = np.asarray(separation['ball'], dtype=float)
    speed = np.asarray(separation['velocity'], dtype=float)
    spin = np.asarray(separation['spin'], dtype=float)
    if speed[1] >= -.25:
        return None
    duration = float(np.clip((target[1]-position[1])/speed[1], 0., .30))
    acceleration = -AIR_DRAG*np.linalg.norm(speed)*speed + MAGNUS_COEFFICIENT*np.cross(spin, speed)
    acceleration[2] -= GRAVITY
    projected = position + duration*(speed+.5*duration*acceleration)
    velocity = speed+duration*acceleration
    return [abs(projected[0]-target[0])/.11, abs(projected[2]-TABLE_HEIGHT-target[2])/.055,
            abs(-velocity[1]-target[3])/.52, (projected[2]-TABLE_HEIGHT-target[2])/.055,
            (velocity[2]-.40)/1.2, abs(velocity[0])/.5, (duration-.085)/.065,
            (separation['time']-contact['time']-.24)/.11,
            spin[0]/80., spin[1]/45., spin[2]/45., np.linalg.norm(spin)/100.]


def episodes(row, target):
    events = row['events']
    far_contact = None
    for index, event in enumerate(events):
        if event['kind'] == 'contact' and event['side'] == -1:
            far_contact = event
        if event['kind'] != 'legal_return' or event['side'] != -1 or far_contact is None:
            continue
        following = []
        for item in events[index+1:]:
            if item['kind'] == 'contact' and item['side'] == -1:
                break
            following.append(item)
        separation = next((item for item in following if item['kind'] == 'separation'
                           and item['collider'] == 'table'), None)
        if separation is None or separation['step']-event['step'] > 2:
            continue
        features = approach(separation, far_contact, target)
        if features is not None:
            legal = any(item['kind'] == 'legal_return' and item['side'] == 1 for item in following)
            yield dict(seed=row['seed'], tick=event['step'], features=features, legal=legal)


def draw_practice(inputs, target):
    samples = []
    seen = set()
    for file, groups in inputs.items():
        evidence = _read(DIR/file)
        for group in groups:
            rows = evidence['rows'] if group == 'rows' else evidence['branches'][group]
            source = 'ordinary_frozen' if group == 'frozen' else f'{file}:{group}'
            for row in rows:
                for item in episodes(row, target):
                    fingerprint = (item['seed'], item['tick'], tuple(item['features']))
                    if fingerprint not in seen:
                        samples.append(dict(item, source=source))
                        seen.add(fingerprint)
    return samples


def design(rows, model):
    features = np.asarray([row['features'][:FEATURES[model]] for row in rows])
    return np.column_stack((np.ones(len(rows)), np.clip(features, -5., 5.)))


def fit(rows, model, ridge):
    x = design(rows, model)
    labels = np.asarray([row['legal'] for row in rows], dtype=float)
    counts = Counter(row['seed'] for row in rows)
    weights = np.asarray([1./counts[row['seed']] for row in rows])
    weights /= weights.mean()
    reg = ridge*np.eye(x.shape[1])
    reg[0, 0] = .1
    coef = np.zeros(x.shape[1])
    for _ in range(10):
        probability = expit(x@coef)
        curvature = weights*probability*(1.-probability)
        gradient = x.T@(weights*(labels-probability))-reg@coef
        coef += np.linalg.solve(x.T@(x*curvature[:, None])+reg, gradient)
    return coef


def validate(rows):
    seeds = sorted({row['seed'] for row in rows})
    results = []
    for model in FEATURES:
        for ridge in RIDGES:
            errors, control, predictions = [], [], []
            for seed in seeds:
                train = [row for row in rows if row['seed'] != seed]
                test = [row for row in rows if row['seed'] == seed]
                prob = expit(design(test, model)@fit(train, model, ridge))
                delta = (prob-[row['legal'] for row in test])**2
                errors.append(delta.mean())
                frozen = [i for i, row in enumerate(test) if row['source'] == 'ordinary_frozen']
                if frozen:
                    control.append(delta[frozen].mean())
                predictions.extend(dict(seed=seed, tick=row['tick'], source=row['source'],
                                        legal=row['legal'], probability=float(estimate))
                                   for row, estimate in zip(test, prob, strict=True))
            results.append(dict(model=model, ridge=ridge, brier=float(np.mean(errors)),
                                frozen_brier=float(np.mean(control)), held=predictions))
    return results


def train(output=PRACTICE):
    profile = _read(DIR/'tt22_opponent-learned.json.gz')['fit']['profile']
    rows = draw_practice(INPUTS, profile)
    validations = validate(rows)
    geometry = min((row for row in validations if row['model'] == 'geometry'), key=lambda row: row['brier'])
    enhanced = min((row for row in validations if row['model'] != 'geometry'), key=lambda row: row['brier'])
    use = (geometry['brier']-enhanced['brier'] >= .005
           and geometry['frozen_brier']-enhanced['frozen_brier'] >= .005)
    model = dict(features=enhanced['model'], ridge=enhanced['ridge'],
                 coefficients=fit(rows, enhanced['model'], enhanced['ridge']).tolist())
    result = dict(schema='tt23-coach-practice-1', profile=profile, input={file: _hash(DIR/file) for file in INPUTS},
                  rows=rows, validation=validations, model=model, live_gate=use)
    _write(output, result)
    return result


if __name__ == '__main__':
    summary = train()
    print(len(summary['rows']), len({r['seed'] for r in summary['rows']}), summary['live_gate'], flush=True)
    for row in summary['validation']:
        print(row['model'], row['ridge'], round(row['brier'], 5), round(row['frozen_brier'], 5), flush=True)
