"""Compare two motor solvers on matched points from the existing mixed courses."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .tt4_transfer import TransferPrior
from .tt4_transfer_trial import ROOT, _counts, _hash, _metrics, _point, _read, _write, load_sources
from .tt5_vision import motion_tracker_factory
from .tt6_body_arena import TT6Arena, TT6Wiring
from .tt6_body_control import fit_camera_body
from .tt6_body_vision import body_tracker_factory
from .tt6_continuous_trial import point_condition
from .tt6_racket import RacketAdaptation, set_racket_response
from .tt6_render import cache_hall_stereo
from .tt6_select import SelectingPlayer, servo_evidence
from .tt6_timing import DelayedWiring, TimingMemory


ARTIFACTS = ROOT / 'artifacts' / 'tt6'
STATIONS = (('combined-carry', 0), ('combined-carry', 1), ('combined-rotary', 0), ('combined-rotary', 2))


def _calibrate(wiring, visit, *, tracker_factory):
    histories = []
    for probe in visit['probes']:
        history = [(np.zeros(wiring.arena.action_size), wiring.reset(seed=probe['seed']))]
        for action in probe['actions']:
            command = np.asarray(action)
            frames, _, _, _, _ = wiring.step(command)
            history.append((command, frames))
        histories.append(history)
    factory = tracker_factory(wiring.arena.cameras, histories)
    return fit_camera_body(histories, factory), factory


def station(task):
    source, index = task
    reference = _read(ARTIFACTS / f'{source}.json.gz')
    visit = reference['visits'][index]
    install = visit['installation']
    seed = reference['seed']
    native, states, _ = load_sources(ROOT / 'artifacts' / 'tt4' / 'mini-first-person-player.json.gz')
    arena = TT6Arena(install['arena_seed'], body=visit['body'], view_seed=install['view_seed'],
                     **reference['geometry'])
    try:
        cache_hall_stereo(arena, reflections=True)
        wiring = TT6Wiring(arena, install['arena_seed'])
        optics = {'body_tracker_factory': body_tracker_factory, 'motion_tracker_factory': motion_tracker_factory}
        historical_tracker = visit['attempts'][-1]['optics']
        fit, factory = _calibrate(wiring, visit, tracker_factory=optics[historical_tracker])
        receipt = dict(source=source, index=index, body=visit['body'], evidence=servo_evidence(fit),
                       calibration=dict(probes=len(visit['probes']), steps=sum(len(p['actions']) for p in visit['probes'])),
                       fit=fit.stats(), historical=_metrics(visit['points']), conditions={})
        for condition in ('predictive', 'short'):
            prior = TransferPrior('first_person_mini', native, seed, mini=states['first_person_mini'])
            player = SelectingPlayer(fit, prior, factory, RacketAdaptation(gain=reference['gain']), TimingMemory(),
                                     servo=condition)
            rows = []
            for row in visit['points']:
                rubber, disturbance = point_condition(row['phase'])
                set_racket_response(arena, rubber)
                interface = DelayedWiring(wiring, disturbance, seed=seed)
                point = _point(interface, player, row['seed'])
                rows.append(dict(phase=row['phase'], applied=interface.applied, **point, **_counts(point)))
            receipt['conditions'][condition] = dict(points=rows, summary=_metrics(rows))
        return receipt
    finally:
        arena.close()


def summarize(stations):
    totals = {}
    for body in sorted({visit['body'] for visit in stations}):
        group = [visit for visit in stations if visit['body'] == body]
        totals[body] = {}
        for condition in ('predictive', 'short', 'automatic'):
            points = []
            for visit in group:
                solver = visit['evidence']['choice'] if condition == 'automatic' else condition
                points.extend(visit['conditions'][solver]['points'])
            totals[body][condition] = _metrics(points)
    return totals


def run(output, workers):
    report = dict(schema='tt6-camera-servo-selection-v1', initial_experience='first_person_mini',
                  reflected_hall=True, plan=STATIONS, visits=[], inputs_sha256={
                      name: _hash(ARTIFACTS / f'{name}.json.gz') for name, _ in STATIONS})
    output.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending = [pool.submit(station, item) for item in STATIONS]
        for result in as_completed(pending):
            visit = result.result()
            report['visits'].append(visit)
            report['visits'].sort(key=lambda row: STATIONS.index((row['source'], row['index'])))
            report['summary'] = summarize(report['visits'])
            _write(output, report)
            print(visit['body'], visit['index'], visit['evidence'],
                  {key: tuple(value['summary'][name] for name in ('with_return', 'player_returns', 'sustained'))
                   for key, value in visit['conditions'].items()}, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('output', type=Path)
    cli.add_argument('--workers', type=int, default=2)
    args = cli.parse_args()
    run(args.output, args.workers)
