"""Fork the baseline player before each confirmed service reply in a completed game."""

import argparse
from copy import deepcopy
from pathlib import Path

from .tt4_transfer_trial import ROOT, _counts, _hash, _point, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_continuous_trial import prepare_point
from .tt6_memory import load
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring
from .tt7_flight import FlightMemory
from .tt8_service_arena import TT8ServiceArena
from .tt8_service_drive import DrivePlayer, ServiceDrive
from .tt8_service_trial import PointInterface
from .tt9_recovery import RecoveryPlayer
from .tt9_recovery_trial import point_summary, verify


def run(receipt, source, output, probe_cache):
    original = _read(receipt)
    if original['source_sha256'] != _hash(source):
        raise ValueError('the paired evaluation requires the original checkpoint')
    library, prior, response, timing = load(source)
    arena = TT8ServiceArena(**original['arena'])
    wiring = TT6Wiring(arena, original['installation_seed'])
    report = dict(schema='tt9-recovery-paired-v1', seed=original['seed'], arena=original['arena'],
                  installation_seed=original['installation_seed'], original_sha256=_hash(receipt),
                  source_sha256=_hash(source), serve_rate=original['serve_rate'],
                  implementation_sha256={name: _hash(Path(__file__).with_name(name)) for name in
                                         ('tt9_recovery.py', 'tt9_recovery_paired.py')},
                  sections={name: dict(points=[]) for name in ('fixed', 'early-lift')})
    try:
        cache_hall_stereo(arena, reflections=True)
        fit, factory, probes, attempts = library.commission(
            DelayedWiring(wiring), arena.cameras, original['schedule'], probe_cache=probe_cache)
        if fit is None or attempts != original['attempts'] or probes != original['probes']:
            raise ValueError('body commissioning differs from the baseline game')
        report.update(probes=probes, attempts=attempts, fit=fit.stats())
        settings = dict(predictive=True, drive=ServiceDrive(rate=report['serve_rate']), learn=False)
        baseline = DrivePlayer(fit, prior, factory, response, timing, flight_memory=FlightMemory('fixed'), **settings)
        candidate = RecoveryPlayer(fit, deepcopy(prior), factory, deepcopy(response), deepcopy(timing),
                                   flight_memory=FlightMemory('fixed'), follow_up='early-lift', **settings)

        def play(player, expected):
            server, phase = expected['server'], expected['phase']
            interface = PointInterface(wiring, prepare_point(arena, phase), server=server, seed=original['seed'])
            row = _point(interface, player, expected['seed'])
            row.update(_counts(row), phase=phase, server=server, applied=interface.applied,
                       summary=point_summary(row), optical_trace=player.trace)
            return row

        for expected in original['sections']['fixed']['points']:
            paired = expected['server'] == 1 and expected['evaluator']['rival_returns'] > 0
            if paired:
                candidate.prior = deepcopy(baseline.prior)
                candidate.timing_memory = deepcopy(baseline.timing_memory)
                candidate.response = deepcopy(baseline.response)
                candidate.memory = deepcopy(baseline.memory)
                candidate.role_rows = deepcopy(baseline.role_rows)
            fixed = play(baseline, expected)
            if fixed['action_sha256'] != expected['action_sha256']:
                raise ValueError(f"baseline actions differ on point {expected['seed']}")
            if not paired:
                continue
            alternative = play(candidate, expected)
            cutoff = next(event['step'] for event in fixed['events']
                          if event['kind'] == 'contact' and event['side'] == -1)
            if fixed['actions'][:cutoff] != alternative['actions'][:cutoff]:
                raise ValueError(f"actions before the service reply differ on point {expected['seed']}")
            report['sections']['fixed']['points'].append(fixed)
            report['sections']['early-lift']['points'].append(alternative)
            print(expected['seed'], fixed['summary'], alternative['summary'], flush=True)
    finally:
        arena.close()
    _write(output, report)
    verify(_read(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('receipt', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--resume', type=Path, default=ROOT/'artifacts/tt8/span-after-four.json.gz')
    parser.add_argument('--probe-cache', type=Path)
    args = parser.parse_args()
    run(args.receipt, args.resume, args.output, args.probe_cache)
