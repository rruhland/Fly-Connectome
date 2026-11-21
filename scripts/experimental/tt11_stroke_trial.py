"""Compare optical stroke recall on matched replies and alternating-service games."""

import argparse
from copy import copy, deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import ROOT, _hash, _read, _seeds, _write
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
from .tt10_cursor_reply import paired, rival_contact
from .tt10_window_trial import CARRIED
from .tt11_stroke import StrokeMemory, StrokePlayer, optical_stroke


DEFAULT_SOURCE = ROOT / 'artifacts/tt9/precontact-match-b-player.json.gz'
PHASES = ('ordinary', 'mixed', 'soft', 'delay')


def _archives(paths):
    """Derive source examples from images, then audit them with the scorer."""
    samples, audits, starts = [], [], set()
    for path in paths:
        for mode, section in _read(path)['sections'].items():
            points = section if isinstance(section, list) else section['points']
            for row in points:
                starts.add(row['seed'])
                trace = row.get('optical_trace')
                if not trace or row['serve_drive'] is None:
                    continue
                sample = optical_stroke(trace, row['serve_drive']['tick'])
                source = f"{path.name}/{mode}/{row['seed']}"
                if sample is not None and sample not in samples:
                    samples.append(sample)
                    audits.append(dict(source=source, seed=row['seed'],
                                       legal=paired(row) and row['player_returns'] > 0))
    memory = StrokeMemory([dict(sample, source=audit['source'])
                           for sample, audit in zip(samples, audits, strict=True)])
    return memory, audits, sorted(starts)


def _summary(rows):
    own = [row for row in rows if row['server'] == 1]
    replies = [row for row in own if paired(row)]
    active = [row for row in rows if row['stroke_recall']['choice'] is not None]
    return dict(points=len(rows), own_serves=len(own), legal_serves=sum(row['evaluator']['legal_serve'] for row in own),
                replies=len(replies), hits=sum(_reply_hit(row) for row in replies),
                first=sum(row['player_returns'] > 0 for row in replies),
                later=sum(max(0, row['player_returns'] - 1) for row in replies),
                reply_wins=sum(row['points_for'] for row in replies),
                all_returns=sum(row['player_returns'] for row in rows),
                all_sustained=sum(row['player_returns'] >= 2 for row in rows),
                points_won=sum(row['points_for'] for row in rows), recalled=len(active),
                optical_examples=sum(row['stroke_recall']['observed'] is not None for row in rows))


def _reply_hit(row):
    step = rival_contact(row)['step']
    return any(event['kind'] == 'contact' and event['side'] == 1 and event['step'] > step
               for event in row['events'])


def _prefix(left, right):
    if not paired(left) or not paired(right) or left['seed'] != right['seed']:
        raise ValueError('the paired starts did not both generate a legal opponent reply')
    choice = right['stroke_recall']['choice']
    rival = rival_contact(left)
    if rival_contact(right) != rival or left['stroke_recall']['reply'] != right['stroke_recall']['reply']:
        raise ValueError('the opponent stroke or optical recognition changed before recall')
    if choice is None:
        if left['action_sha256'] != right['action_sha256']:
            raise ValueError('command tape changed without a recalled stroke')
        return None
    tick = choice['tick']
    if (not rival['step'] <= right['stroke_recall']['reply'] <= tick
            or left['actions'][:tick] != right['actions'][:tick]
            or left['applied'][:tick] != right['applied'][:tick]):
        raise ValueError('the matched branches diverged before optical stroke selection')
    return tick


def _append(report, mode, row, output):
    section = report['sections'][mode]
    section['points'].append(row)
    section['summary'] = _summary(section['points'])
    _write(output, report)
    recall = row['stroke_recall']['choice']
    print(mode, row['seed'], row['phase'], row['server'], row['player_returns'], row['points_for'],
          row['evaluator']['reason'], None if recall is None else (recall['tick'], recall['example']),
          section.get('score'), flush=True)


def _play(active, report, arena, wiring, seed, phase, server):
    row = _play_point(active, arena, wiring, report['seed'], seed, phase, server)
    if active.reply is not None:
        row['optical_trace'] = active.trace
    return row


def _pairs(report, arena, wiring, baseline, alternative, output):
    count = 0
    for seed, phase in report['schedule']:
        branch = copy(alternative)
        for key in CARRIED:
            setattr(branch, key, deepcopy(getattr(baseline, key)))
        left = _play(baseline, report, arena, wiring, seed, phase, 1)
        _append(report, 'fixed', left, output)
        if not paired(left):
            continue
        right = _play(branch, report, arena, wiring, seed, phase, 1)
        right['verified_prefix'] = _prefix(left, right)
        _append(report, 'recall', right, output)
        count += 1
    if count == 0:
        raise RuntimeError('the held games gave no identical replies to compare')


def _games(args, report, arena, wiring, players, library, state):
    for mode, player in players.items():
        score = ServiceScore(first_server=-1)
        section = report['sections'][mode]
        section.update(race_to=11, first_server=score.server)
        for seed, phase in report['schedule']:
            row = _play(player, report, arena, wiring, seed, phase, score.server)
            score.record(row['evaluator'])
            section.update(score=list(score.score), finished=score.finished)
            _append(report, mode, row, args.output)
            if score.finished:
                break
        if not score.finished:
            raise RuntimeError(f'{mode} game exhausted the point limit')
        section['memory_after'] = player.strokes.state()
        _write(args.output, report)
    if args.checkpoint is not None:
        active = players['recall']
        saved = dict(state, reply_strokes=active.strokes.state())
        save_player(args.checkpoint, saved, library, active)
        load(args.checkpoint)
        if StrokeMemory(**_read(args.checkpoint)['reply_strokes']).state() != active.strokes.state():
            raise ValueError('checkpoint failed to restore the stroke library')
        report['checkpoint_sha256'] = _hash(args.checkpoint)
        _write(args.output, report)


def verify(report):
    count = 0
    for section in report['sections'].values():
        check_tapes(dict(report, **section))
        if _summary(section['points']) != section['summary']:
            raise ValueError('stroke metrics do not match the stored point tapes')
        for row in section['points']:
            tape = np.asarray(row['actions'], dtype='<f8').tobytes()
            if sha256(tape).hexdigest() != row['action_sha256']:
                raise ValueError(f"action digest does not match point {row['seed']}")
            observed = row['stroke_recall']['observed']
            if 'optical_trace' in row:
                if optical_stroke(row['optical_trace'], row['serve_drive']['tick']) != observed:
                    raise ValueError('stored optical stroke disagrees with the replayed measurements')
            elif observed is not None:
                raise ValueError('optical stroke is missing the supporting measurements')
        count += len(section['points'])
    if report['mode'] == 'paired':
        fixed, recall = (report['sections'][mode]['points'] for mode in ('fixed', 'recall'))
        for left, right in zip(filter(paired, fixed), recall, strict=True):
            if _prefix(left, right) != right['verified_prefix']:
                raise ValueError('the stored identical prefix does not match the tapes')
    return count


def run(args):
    library, prior, response, timing = load(args.resume)
    state = _read(args.resume)
    if args.archives:
        memory, audits, starts = _archives(args.archives)
    else:
        memory = StrokeMemory(**state.get('reply_strokes', {}))
        audits, starts = [], []
    if not memory.samples:
        raise ValueError('no optically confirmed practice strokes are available')
    if args.reference is not None:
        reference = _read(args.reference)
        arena_args = reference['arena']
        calibration = _seeds(reference['seed'], 0, 8, 0)
    else:
        reference = None
        installation = _seeds(args.installation_seed, 0, 1, 1)
        calibration = _seeds(args.seed, 0, 8, 0)
        arena_args = dict(seed=installation['arena_seed'], view_seed=installation['view_seed'], body=args.body,
                          venue='hall', view='first_person', render_mode='mujoco', width=192, height=144,
                          baseline=.28, max_steps=420, frameskip=8, distractor=False)
    schedule = _cases(args.seed, 7211 if args.mode == 'paired' else 7212, args.limit, PHASES)
    if set(starts).intersection(seed for seed, _ in schedule):
        raise ValueError('practice and evaluation point seeds overlap')
    report = dict(schema='tt11-stroke-recall-v1', seed=args.seed, mode=args.mode, arena=arena_args,
                  installation_seed=arena_args['seed'], source_sha256=_hash(args.resume), schedule=schedule,
                  reference_sha256=_hash(args.reference) if reference else None, calibration=calibration,
                  archives={path.name: _hash(path) for path in args.archives},
                  implementation={name: _hash(Path(__file__).with_name(name))
                                  for name in ('tt11_stroke.py', 'tt11_stroke_trial.py')},
                  memory_before=memory.state(), optical_archive_audit=audits, learn_strokes=args.learn_strokes,
                  sections={mode: dict(points=[]) for mode in ('fixed', 'recall')})
    args.output.parent.mkdir(exist_ok=True, parents=True)
    arena = TT8ServiceArena(**arena_args)
    wiring = TT6Wiring(arena, arena_args['seed'])
    try:
        cache_hall_stereo(arena, reflections=True)
        fit, factory, probes, attempts = library.commission(
            DelayedWiring(wiring), arena.cameras, calibration, probe_cache=args.probe_cache)
        if fit is None:
            raise RuntimeError(f'camera commissioning failed: {attempts}')
        report.update(probes=probes, attempts=attempts, fit=fit.stats())
        players = {
            mode: StrokePlayer(fit, deepcopy(prior), factory, deepcopy(response), deepcopy(timing),
                               predictive=True, flight_memory=FlightMemory('fixed'), learn=False,
                               drive=ServiceDrive(**state.get('service_drive', {})),
                               registration=RegistrationMemory(**deepcopy(state['service_registration'])),
                               learn_registration=True, retrieve=mode == 'recall', strokes=deepcopy(memory),
                               learn_strokes=args.learn_strokes and mode == 'recall')
            for mode in report['sections']
        }
        if args.mode == 'paired':
            _pairs(report, arena, wiring, players['fixed'], players['recall'], args.output)
        else:
            _games(args, report, arena, wiring, players, library, state)
    finally:
        arena.close()
    saved = _read(args.output)
    print(f"{verify(saved)} point tapes replayed", {mode: section['summary']
                                                    for mode, section in saved['sections'].items()}, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--mode', choices=('paired', 'game'), default='paired')
    p.add_argument('--seed', type=int)
    p.add_argument('--reference', type=Path, help='reuse an earlier camera installation and commissioning schedule')
    p.add_argument('--installation-seed', type=int)
    p.add_argument('--body', choices=('offset', 'gantry'), default='offset')
    p.add_argument('--resume', type=Path, default=DEFAULT_SOURCE)
    p.add_argument('--archives', nargs='+', type=Path, default=[])
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--limit', type=int, default=24)
    p.add_argument('--probe-cache', type=Path)
    p.add_argument('--learn-strokes', action='store_true')
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    if a.verify:
        print(f'{verify(_read(a.output))} point tapes replayed')
    elif (a.reference is None) == (a.installation_seed is None) or a.seed is None or a.limit < 1:
        p.error('supply a course seed, a positive point limit, and one installation source')
    elif a.mode == 'paired' and (a.checkpoint or a.learn_strokes):
        p.error('checkpoint saving and online stroke collection require a full game')
    else:
        run(a)
