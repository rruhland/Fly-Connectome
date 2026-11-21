"""Intervene on the predictive body map after an own-serve reply in RGB."""

import argparse
from copy import copy, deepcopy
import json
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt6_body_arena import TT6Wiring
from .tt6_memory import load
from .tt6_render import cache_hall_stereo
from .tt6_timing import DelayedWiring
from .tt7_flight import FlightMemory
from .tt8_service_arena import TT8ServiceArena
from .tt8_service_drive import ServiceDrive
from .tt8_service_trial import check_tapes
from .tt9_precontact_registration import RegistrationMemory, RegistrationPlayer
from .tt9_precontact_trial import _play_point
from .tt9_recovery import RecoveryPlayer


class CursorReplyPlayer(RegistrationPlayer):
    reply_fit = None
    _observe_reply = RecoveryPlayer._observe_reply

    def reset(self):
        super().reset()
        self.far_sighting = None
        self.reply = None
        self.map_activation = None

    def fork(self, reply_fit):
        # The next reset replaces all per-point tracking, planning and servo state.
        branch = copy(self)
        for name in ('prior', 'timing_memory', 'response', 'memory', 'approach', 'role_rows', 'drive'):
            setattr(branch, name, deepcopy(getattr(self, name)))
        branch.reply_fit = reply_fit
        return branch

    def _stroke_goals(self, estimate):
        self._observe_reply()
        if self.reply_fit is not None and self.reply == self.scene.tick:
            self.servo.fit = self.reply_fit
            self.map_activation = self.scene.tick
        return super()._stroke_goals(estimate)

    def finish(self, reward):
        return dict(super().finish(reward), optical_reply=self.reply, map_activation=self.map_activation)


def paired(row):
    evaluated = row['evaluator']
    return row['server'] == 1 and evaluated['legal_serve'] and evaluated['rival_returns'] > 0


def rival_contact(row):
    return next(event for event in row['events'] if event['kind'] == 'contact' and event['side'] == -1)


def check_pair(baseline, swapped):
    contact = rival_contact(baseline)
    step = contact['step']
    activated = swapped['map_activation']
    if (baseline['map_activation'] is not None or baseline['optical_reply'] != swapped['optical_reply']
            or activated != swapped['optical_reply']):
        raise ValueError(f"reply detections disagree on {baseline['seed']}")
    if activated is not None and activated < step:
        raise ValueError(f"map changed before the rival contact on {baseline['seed']}")
    prefix = step if activated is None else activated
    if (rival_contact(swapped) != contact or baseline['actions'][:prefix] != swapped['actions'][:prefix]
            or baseline['applied'][:prefix] != swapped['applied'][:prefix]):
        raise ValueError(f"pre-intervention tapes differ on {baseline['seed']}")
    if activated is None and baseline['action_sha256'] != swapped['action_sha256']:
        raise ValueError(f"actions differ without activation on {baseline['seed']}")
    return step


def summary(points):
    return dict(opportunities=len(points), first=sum(row['player_returns'] > 0 for row in points),
                later=sum(max(0, row['player_returns'] - 1) for row in points),
                sustained=sum(row['player_returns'] >= 2 for row in points),
                total=sum(row['player_returns'] for row in points), wins=sum(row['points_for'] for row in points),
                activated=sum(row['map_activation'] is not None for row in points))


def replay(report):
    baseline, swapped = (report['sections'][name] for name in ('baseline', 'swapped'))
    for fixed, alternative, metadata in zip(baseline, swapped, report['pairs'], strict=True):
        if check_pair(fixed, alternative) != metadata['opponent_contact_step']:
            raise ValueError('stored opponent contact changed')
    for points in (baseline, swapped):
        check_tapes(dict(report, points=points, practice=True))
    return len(baseline) + len(swapped)


def run(args):
    original = _read(args.baseline)
    trained = _read(args.trained)
    if (original['source'] != _hash(args.source) or trained['source'] != _hash(args.updated)
            or any(original[key] != trained[key] for key in ('seed', 'arena', 'schedule', 'registration_before'))
            or original['registration_mode'] != 'online' or original['drive_mode'] != 'frozen'):
        raise ValueError('receipts must describe the matched registered table games and their exact sources')
    original_state, updated_state = _read(args.source), _read(args.updated)
    model_index = original['attempts'][-1]['recalled']
    if ([key for key in original_state if original_state[key] != updated_state[key]] != ['models']
            or [i for i, (before, after) in enumerate(zip(original_state['models'], updated_state['models'],
                                                         strict=True)) if before != after] != [model_index]):
        raise ValueError('the updated checkpoint must only replace the original selected body model')
    library, prior, response, timing = load(args.source)
    updated_library, _, _, _ = load(args.updated)
    updated_fit = updated_library.models[model_index]
    base_fit = library.models[model_index]
    delta = updated_fit.coefficients - base_fit.coefficients
    allowed = np.zeros_like(delta, dtype=bool)
    allowed[1:base_fit.action_size + 1, :3] = True
    base_body, updated_body = original_state['models'][model_index], updated_state['models'][model_index]
    if (not np.any(delta) or np.any(delta[~allowed]) or any(base_body[key] != updated_body[key]
                                                          for key in base_body if key != 'coefficients')):
        raise ValueError('only linear translational pose coefficients may change')

    report = dict(schema='tt10-cursor-reply-v1', seed=original['seed'], arena=original['arena'],
                  installation_seed=original['installation_seed'], source_sha256=_hash(args.source),
                  updated_sha256=_hash(args.updated), baseline_sha256=_hash(args.baseline),
                  trained_sha256=_hash(args.trained), implementation_sha256=_hash(Path(__file__)),
                  model_index=model_index, model_delta=delta.tolist(),
                  sections=dict(baseline=[], swapped=[]), pairs=[], baseline_hashes=[])
    args.output.parent.mkdir(exist_ok=True, parents=True)
    arena = TT8ServiceArena(**original['arena'])
    wiring = TT6Wiring(arena, original['installation_seed'])
    try:
        cache_hall_stereo(arena, reflections=True)
        fit, factory, probes, attempts = library.commission(
            DelayedWiring(wiring), arena.cameras, original['schedule'], probe_cache=args.probe_cache)
        if fit is not base_fit or probes != original['probes'] or attempts != original['attempts']:
            raise ValueError('commissioning must exactly reproduce the original game')
        report.update(probes=probes, attempts=attempts)
        baseline = CursorReplyPlayer(fit, prior, factory, response, timing, predictive=True, learn=False,
                                     flight_memory=FlightMemory('fixed'), drive=ServiceDrive(**original['drive_before']),
                                     registration=RegistrationMemory(**original_state['service_registration']),
                                     learn_registration=True)

        def play(player, expected):
            row = _play_point(player, arena, wiring, original['seed'], expected['seed'], expected['phase'],
                              expected['server'])
            if paired(expected):
                row['optical_trace'] = player.trace
            return row

        for expected in original['points']:
            branch = baseline.fork(updated_fit) if paired(expected) else None
            fixed = play(baseline, expected)
            if (fixed['action_sha256'] != expected['action_sha256'] or fixed['applied'] != expected['applied']
                    or fixed['evaluator'] != expected['evaluator']
                    or json.loads(json.dumps(fixed['events'])) != expected['events']):
                raise ValueError(f"baseline game diverged on {expected['seed']}")
            report['baseline_hashes'].append(fixed['action_sha256'])
            if branch is None:
                continue
            swapped = play(branch, expected)
            step = check_pair(fixed, swapped)
            report['sections']['baseline'].append(fixed)
            report['sections']['swapped'].append(swapped)
            report['pairs'].append(dict(seed=fixed['seed'], opponent_contact_step=step,
                                        activation=swapped['map_activation']))
            print(fixed['seed'], step, swapped['map_activation'], fixed['player_returns'], swapped['player_returns'],
                  fixed['points_for'], swapped['points_for'], flush=True)
        if baseline.approach.state() != original['registration_after'] or baseline.drive.__dict__ != original['drive_after']:
            raise ValueError('final baseline service state differs from the original game')
    finally:
        arena.close()
    report['summary'] = {name: summary(points) for name, points in report['sections'].items()}
    _write(args.output, report)
    print(f"{replay(_read(args.output))} forked point tapes replayed", report['summary'], flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('output', type=Path)
    p.add_argument('--baseline', type=Path)
    p.add_argument('--trained', type=Path)
    p.add_argument('--source', type=Path)
    p.add_argument('--updated', type=Path)
    p.add_argument('--probe-cache', type=Path)
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    if a.verify:
        print(f"{replay(_read(a.output))} forked point tapes replayed")
    elif None in (a.baseline, a.trained, a.source, a.updated):
        p.error('supply both game receipts and their corresponding starting checkpoints')
    else:
        run(a)
