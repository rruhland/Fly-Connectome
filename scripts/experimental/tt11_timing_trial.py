"""Qualify a prefixture RGB timing choice against two fixed models in paired table games."""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace

import numpy as np

from .tt4_arena import TIMESTEP
from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
from .tt6_memory import _serializable_body, load
from .tt10_series_resume import verify as verify_resume
from .tt10_series_trial import SOURCE, prepare as commission, run as play
from .tt10_target import TargetArena
from .tt10_target_dynamics import errors, fit_kernels, observe as cursor_observe
from .tt11_timing_select import MODES, PROBE_COUNT, observe, select


OLD = ROOT / 'artifacts/tt10'
FAMILIAR = {
    'a': ('target-time-a-base-start.json.gz', 'target-time-a-held-start.json.gz', 'target-time-a-fit.json.gz'),
    'b': ('series-b-start.json.gz', 'target-time-b-held-start.json.gz', 'target-time-b-fit.json.gz'),
}
NEW = {'c': (842076133, 992713820), 'd': (510443081, 1886180491)}
PROBE_SEED = 646076469
CURSOR_SEED = 308495177
FIXTURE_SEED = 739294612
START_MODES = (*MODES, 'selected')


def _path(folder, site, label):
    return folder / f'timing-{site}-{label}.json.gz'


def _fit(path):
    return load(path)[0].models[_read(path)['series']['model']]


def _load_candidates(site, folder, index):
    if site in FAMILIAR:
        start, update, fit_path = (OLD / name for name in FAMILIAR[site])
        fit = _fit(start)
        updated = _fit(update)
        receipt = _read(fit_path)
        cost = dict(sessions=len(receipt['training']), targets=16,
                    commands=sum(len(tape) - 1 for tape in receipt['tape']['requested'][:2]),
                    fitting_receipt_sha256=_hash(fit_path), held_cursor_errors={
                        'inherited': receipt['source_errors'], 'cursor': receipt['trained_errors']})
        return start, fit, updated, cost

    start = _path(folder, site, 'commissioned')
    installation, calibration = NEW[site]
    if not start.exists():
        commission(SimpleNamespace(donor=SOURCE, body='offset', output=start, installation_seed=installation,
                                   calibration_seed=calibration, probe_cache=None))
    fit = _fit(start)
    entry = _read(start)['series']
    arena = TargetArena(**dict(entry['arena'], render_mode='projected'))
    seeds = _seeds(CURSOR_SEED, index, 0, 2)['tests']
    try:
        runs = [cursor_observe(arena, entry['arena']['seed'], fit, seed) for seed in seeds]
    finally:
        arena.close()
    actions, motion, outcomes = map(np.asarray, zip(*runs, strict=True))
    updated = fit_kernels(fit, actions, motion)
    cursor_receipt = _path(folder, site, 'cursor-fit')
    _write(cursor_receipt, dict(seeds=seeds, scores=outcomes.tolist(), requested=actions.tolist(),
                                observed_xyz=motion.tolist(), training_errors={
                                    'inherited': errors(fit, actions, motion),
                                    'cursor': errors(updated, actions, motion)}))
    cost = dict(sessions=len(seeds), targets=int(outcomes.size), commands=int(actions.shape[0] * (actions.shape[1] - 1)),
                fitting_receipt_sha256=_hash(cursor_receipt))
    return start, fit, updated, cost


def prepare(site, folder):
    """Freeze the comparison's timing arrays and selected start before opening any fixture."""
    started = perf_counter()
    folder.mkdir(parents=True, exist_ok=True)
    index = (*FAMILIAR, *NEW).index(site)
    start, inherited, cursor_fit, cost = _load_candidates(site, folder, index)
    # Deliberately transfer only the two actuator timing arrays.
    cursor = replace(inherited, kernels=cursor_fit.kernels, lags=cursor_fit.lags)
    if not np.array_equal(inherited.coefficients, cursor_fit.coefficients):
        raise ValueError('cursor timing fit was prepared with a different geometry model')
    state = _read(start)
    entry = state['series']
    seeds = _seeds(PROBE_SEED, index, 0, PROBE_COUNT)['tests']
    rows = observe(entry['arena'], entry['optics'], inherited, seeds)
    models = dict(zip(MODES, (inherited, cursor), strict=True))
    control_dt = entry['arena']['frameskip'] * TIMESTEP
    choice = select(models, rows, control_dt)
    report = dict(schema='tt11-prefixture-timing-v1', site=site, arena=entry['arena'], commissioning_sha256=_hash(start),
                  calibrations=len(entry['probes']), calibration_commands=sum(len(p['actions']) for p in entry['probes']),
                  cursor_cost=cost, choice=choice, probes=rows, timing={name: dict(lags=m.lags.tolist(),
                    kernels=m.kernels.tolist()) for name, m in models.items()},
                  probe_commands=sum(len(row['commands']) for row in rows),
                  probe_stereo_pairs=sum(1 + len(row['commands']) for row in rows),
                  control_dt=control_dt, fixture_seed=FIXTURE_SEED,
                  implementation_sha256={name: _hash(Path(__file__).with_name(name)) for name in
                                         ('tt11_timing_trial.py', 'tt11_timing_select.py')}, starts={})
    for mode in START_MODES:
        candidate = choice['selected'] if mode == 'selected' else mode
        state['models'][entry['model']] = _serializable_body(models[candidate])
        if mode == 'selected':
            state['timing_selection'] = dict(site=site, model=candidate, candidates=report['timing'], decision=choice,
                                             commissioning_sha256=report['commissioning_sha256'])
        path = _path(folder, site, f'{mode}-start')
        _write(path, state)
        report['starts'][mode] = _hash(path)
    report['preparation_wall_seconds'] = perf_counter() - started
    _write(_path(folder, site, 'probes'), report)
    print(site, choice['selected'], choice['error_m'], choice['visible_pairs'], choice['median_peak_forward_mps'],
          flush=True)


def check_selection(site, folder, report):
    paths = {mode: _path(folder, site, f'{mode}-start') for mode in START_MODES}
    if {mode: _hash(path) for mode, path in paths.items()} != report['starts']:
        raise ValueError('initial checkpoint hashes changed')
    states = {mode: _read(path) for mode, path in paths.items()}
    decision = states['selected'].pop('timing_selection')
    if decision['decision'] != report['choice'] or decision['candidates'] != report['timing']:
        raise ValueError('selector checkpoint lost its prefixture evidence')
    if states['selected'] != states[report['choice']['selected']]:
        raise ValueError('selector changed more than what its corresponding fixed model used')
    index = states['inherited']['series']['model']
    for field in ('kernels', 'lags'):
        states['inherited']['models'][index][field] = states['cursor']['models'][index][field]
    if states['inherited'] != states['cursor']:
        raise ValueError('the two fixed baselines differ outside the actuator timing arrays')
    inherited, cursor = (_fit(paths[mode]) for mode in MODES)
    entry = states['inherited']['series']
    seeds = [row['seed'] for row in report['probes']]
    probes = observe(entry['arena'], entry['optics'], inherited, seeds)
    if probes != report['probes']:
        raise ValueError('replaying dry strokes changed the observed RGB or reconstructed blade tape')
    if select(dict(inherited=inherited, cursor=cursor), probes, report['control_dt']) != report['choice']:
        raise ValueError('replaying dry strokes changed the selected timing')
    return len(probes)


def qualify(site, folder, games):
    report = _read(_path(folder, site, 'probes'))
    started = perf_counter()
    checked = check_selection(site, folder, report)
    sections = {}
    for mode in START_MODES:
        start = _path(folder, site, f'{mode}-start')
        output = _path(folder, site, f'{mode}-games')
        checkpoint = _path(folder, site, f'{mode}-player') if mode == 'selected' else None
        begin = perf_counter()
        play(SimpleNamespace(resume=start, output=output, seed=report['fixture_seed'], games=games, mode='fresh',
                             limit=65, checkpoint=checkpoint))
        record = _read(output)
        sections[mode] = dict(sha256=_hash(output), games=[dict(score=g['score'], **g['metrics'],
                              actions=sum(row['steps'] for row in g['points']),
                              stereo_pairs=sum(row['frames'] for row in g['points'])) for g in record['games']],
                              wall_seconds=perf_counter() - begin)

    records = {mode: _read(_path(folder, site, f'{mode}-games')) for mode in START_MODES}
    candidate = report['choice']['selected']
    if records['selected']['games'] != records[candidate]['games']:
        raise ValueError('the chosen fixed timing and selected timing did not reproduce the same games')
    inherited, cursor = (records[mode] for mode in MODES)
    for left, right in zip(inherited['games'], cursor['games'], strict=True):
        if left['schedule'] != right['schedule']:
            raise ValueError('comparison branches used different point schedules')

    checkpoint = _path(folder, site, 'selected-player')
    start = _path(folder, site, 'selected-start')
    receipt = _path(folder, site, 'selected-games')
    resume = verify_resume(start, receipt, points=3)
    if _read(checkpoint)['timing_selection'] != _read(start)['timing_selection']:
        raise ValueError('postgame checkpoint lost the prefixture model decision')
    load(checkpoint)
    result = dict(schema='tt11-paired-timing-games-v1', site=site, selected=candidate, sections=sections,
                  verified_probe_rgb_tapes=checked, verified_live_resume=resume,
                  matched_selected_games=len(records['selected']['games']), checkpoint_sha256=_hash(checkpoint),
                  qualification_wall_seconds=perf_counter() - started)
    _write(_path(folder, site, 'comparison'), result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('prepare', 'qualify', 'all'))
    parser.add_argument('site', choices=(*FAMILIAR, *NEW))
    parser.add_argument('folder', type=Path)
    parser.add_argument('--games', type=int, default=1)
    args = parser.parse_args()
    if args.games < 1:
        parser.error('schedule at least one full game')
    if args.stage in ('prepare', 'all'):
        prepare(args.site, args.folder)
    if args.stage in ('qualify', 'all'):
        qualify(args.site, args.folder, args.games)


if __name__ == '__main__':
    main()
