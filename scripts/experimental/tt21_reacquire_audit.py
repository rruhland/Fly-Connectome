"""Count visible and physically returned second-ball opportunities in TT20."""

from collections import Counter

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _write


def audit(folder):
    outcomes = {key: Counter() for key in ('left', 'wrist')}
    distinct = set()
    files = sorted(folder.glob('camera-carry-*-game-*.json.gz'))
    for path in files:
        game = _read(path)
        eye = game['eye']
        for point in game['points']:
            if not point['outcome']['rival_returns']:
                continue
            events = point['events']
            strike_index = next(i for i, event in enumerate(events)
                                if event['kind'] == 'contact' and event.get('side') == -1)
            landing_index = next(i for i in range(strike_index, len(events))
                                 if events[i]['kind'] == 'legal_return' and events[i].get('side') == -1)
            strike, landing = events[strike_index], events[landing_index]
            fingerprint = (eye, point['seed'], tuple(np.round(strike['location'], 7)),
                           tuple(np.round(landing['location'], 7)))
            if fingerprint in distinct:
                continue
            distinct.add(fingerprint)
            sightings = sum(strike['step'] <= tick <= landing['step'] + 4 and -.81 < ball[1] < -.24
                            for tick, ball in point['optical'])
            count = outcomes[eye]
            count['legal_rival_reply'] += 1
            count['two_sightings'] += sightings >= 2
            count['no_sightings'] += not sightings
            count['next_contact'] += any(event['kind'] == 'contact' and event.get('side') == 1
                                         for event in events[landing_index + 1:])
            count['second_legal_return'] += point['outcome']['legal_returns'] >= 2
            count['no_followup_commands'] += not any(max(map(abs, action)) > .025
                                                     for action in point['actions'][strike['step'] + 5:])
            count['landing_beyond_20cm'] += abs(landing['location'][0]) > .20
    return dict(schema='tt21-reacquire-archive-v1', inputs={path.name: _hash(path) for path in files},
                counts={eye: dict(outcome) for eye, outcome in outcomes.items()})


if __name__ == '__main__':
    records = ROOT / 'artifacts'
    result = audit(records / 'tt20')
    _write(records / 'tt21' / 'reacquire-archive.json', result)
    print(result['counts'])
