"""Qualify the cursor-trained reply predictor in fresh alternating games."""

import argparse
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _seeds, _write
from .tt6_body_arena import TT6Wiring
from .tt6_memory import load
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring
from .tt7_flight import FlightMemory
from .tt8_service_arena import ServiceScore, TT8ServiceArena
from .tt8_service_drive import ServiceDrive
from .tt8_service_trial import check_tapes
from .tt9_precontact_match import save_player
from .tt9_precontact_registration import RegistrationMemory
from .tt9_precontact_registration_trial import _cases
from .tt9_precontact_trial import _play_point
from .tt10_cursor_reply import CursorReplyPlayer, paired, summary


def resume_fit(args, state, reference, original):
    reference_hash = _hash(args.reference)
    index = reference['model_index']
    if args.updated is not None:
        if reference['updated_sha256'] != _hash(args.updated) or reference['source_sha256'] != _hash(args.source):
            raise ValueError('the starting geometry must come from the paired trials')
        updated, _, _, _ = load(args.updated)
        coefficients = updated.models[index].coefficients
    else:
        saved = state['cursor_reply_model']
        if saved['reference_sha256'] != reference_hash or saved['model_index'] != index:
            raise ValueError('the resumed predictor must come from the same paired trials')
        coefficients = np.asarray(saved['coefficients'])
    if not np.array_equal(coefficients - original.coefficients, reference['model_delta']):
        raise ValueError('cursor pose coefficient changes differ from the paired check')
    return replace(original, coefficients=coefficients), dict(reference_sha256=reference_hash,
                                                               model_index=index,
                                                               coefficients=coefficients.tolist())


def match_summary(points):
    own = [row for row in points if row['server'] == 1]
    replies = [row for row in own if paired(row)]
    active = [row for row in points if row['map_activation'] is not None]
    delays = []
    early = 0
    for row in active:
        contact = next((event['step'] for event in row['events']
                        if event['kind'] == 'contact' and event['side'] == -1), None)
        if row['server'] == 1 and contact is not None and contact <= row['map_activation']:
            delays.append(row['map_activation'] - contact)
        else:
            early += 1
    return dict(own_serves=len(own), own_legal=sum(row['evaluator']['legal_serve'] for row in own),
                own_wins=sum(row['points_for'] for row in own), own_replies=summary(replies),
                total_legal_returns=sum(row['player_returns'] for row in points),
                activation_frames=delays, early_or_unconfirmed_activations=early)


def replay(report):
    total = 0
    for section in report['sections'].values():
        points = section['points']
        if match_summary(points) != section['summary']:
            raise ValueError('stored summary differs from the recorded points')
        check_tapes(dict(report, **section, race_to=11, first_server=-1))
        total += len(points)
    return total


def run(args):
    reference = _read(args.reference)
    previous = _read(args.source)
    library, prior, response, timing = load(args.source)
    original = library.models[reference['model_index']]
    adapted, saved_map = resume_fit(args, previous, reference, original)
    calibration = _seeds(reference['seed'], 0, 8, 0)
    if args.seed == reference['seed']:
        raise ValueError('qualify on a fresh point seed')
    starts = _cases(args.seed, 1704, args.limit, ('ordinary', 'mixed', 'soft', 'delay'))
    report = dict(schema='tt10-cursor-reply-match-v1', seed=args.seed, arena=reference['arena'],
                  installation_seed=reference['installation_seed'], source_sha256=_hash(args.source),
                  reference_sha256=saved_map['reference_sha256'], model_delta=reference['model_delta'],
                  implementation_sha256={name: _hash(Path(__file__).with_name(name))
                                         for name in ('tt10_cursor_reply.py', 'tt10_cursor_reply_match.py')},
                  schedule=starts, sections={})
    args.output.parent.mkdir(exist_ok=True, parents=True)
    arena = TT8ServiceArena(**reference['arena'])
    wiring = TT6Wiring(arena, reference['installation_seed'])
    try:
        cache_hall_stereo(arena, reflections=True)
        fit, factory, probes, attempts = library.commission(
            DelayedWiring(wiring), arena.cameras, calibration, probe_cache=args.probe_cache)
        if fit is not original or attempts != reference['attempts'] or probes != reference['probes']:
            raise ValueError('normal commissioning did not identify the original calibrated body')
        report.update(probes=probes, attempts=attempts)
        for mode in ('gated', 'baseline'):
            player = CursorReplyPlayer(fit, deepcopy(prior), factory, deepcopy(response), deepcopy(timing),
                                       predictive=True, learn=False, flight_memory=FlightMemory('fixed'),
                                       drive=ServiceDrive(**previous['service_drive']), learn_registration=True,
                                       registration=RegistrationMemory(**deepcopy(previous['service_registration'])))
            player.reply_fit = adapted if mode == 'gated' else None
            section = report['sections'][mode] = dict(points=[])
            score = ServiceScore(first_server=-1)
            for point_seed, phase in starts:
                row = _play_point(player, arena, wiring, args.seed, point_seed, phase, score.server)
                if row['optical_reply'] is not None:
                    row['optical_trace'] = player.trace
                section['points'].append(row)
                score.record(row['evaluator'])
                section.update(score=list(score.score), finished=score.finished, winner=score.winner,
                               summary=match_summary(section['points']))
                _write(args.output, report)
                print(mode, point_seed, row['server'], row['map_activation'], row['player_returns'],
                      row['points_for'], score.score, flush=True)
                if score.finished:
                    break
            if not score.finished:
                raise RuntimeError(f'point limit reached with {mode} game unfinished')
            if mode == 'gated' and args.checkpoint:
                save_player(args.checkpoint, dict(previous, cursor_reply_model=saved_map), library, player)
                saved = _read(args.checkpoint)
                saved_library, _, _, _ = load(args.checkpoint)
                saved_fit = saved_library.models[reference['model_index']]
                if saved['cursor_reply_model'] != saved_map or not np.array_equal(saved_fit.coefficients,
                                                                                  original.coefficients):
                    raise ValueError('checkpoint changed the default map or lost the gated map')
                report['checkpoint_sha256'] = _hash(args.checkpoint)
                _write(args.output, report)
    finally:
        arena.close()
    print(f'{replay(_read(args.output))} match point tapes replayed', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--reference', type=Path, help='verified reply-fork receipt for this body')
    p.add_argument('--source', type=Path)
    p.add_argument('--updated', type=Path, help='cursor-fit checkpoint, optional when resuming a gated match player')
    p.add_argument('--seed', type=int)
    p.add_argument('--limit', type=int, default=60)
    p.add_argument('--probe-cache', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    if a.verify:
        print(f'{replay(_read(a.output))} match point tapes replayed')
    elif None in (a.reference, a.source, a.seed) or a.limit < 1:
        p.error('supply the reference, starting player, fresh seed and a positive point limit')
    else:
        run(a)
