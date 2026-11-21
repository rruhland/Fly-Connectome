"""Compare recorded optical departure measurements with evaluator-only racket separations."""
import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _read


def audit(visit, frame_rate):
    measured = pairs = total = 0
    errors = []
    for point in visit['points']:
        events = [event for event in point['events'] if event.get('collider') == 'learner_racket'
                  and event['kind'] == 'separation']
        total += len(events)
        for measurement in point['outgoing_model']['measurements']:
            measured += 1
            if not events:
                continue
            index = min(range(len(events)), key=lambda i: abs(events[i]['step'] - measurement['contact']))
            if abs(events[index]['step'] - measurement['contact']) > 6.:
                continue
            event = events.pop(index)
            pairs += 1
            errors.append(np.abs(frame_rate * np.asarray(measurement['observed']) - event['velocity'][1:]))
    return dict(physical_separations=total, optical_windows=measured, matched=pairs,
                median_speed_errors_m_s=np.median(errors, axis=0).tolist() if errors else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    args = parser.parse_args()
    report = _read(args.receipt)
    from .tt4_arena import TIMESTEP
    for visit in report['visits']:
        print(visit['install'], audit(visit, 1. / (report["geometry"]["frameskip"] * TIMESTEP)))
