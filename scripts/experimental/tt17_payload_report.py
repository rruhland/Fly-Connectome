"""Check paired replay receipts and summarize dock scores and change attribution."""

import argparse
from pathlib import Path

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt17_payload import (AFTER, BASELINES, BEFORE, BLOCK_FRAMES, CHANGES, DELIVERY_BLOCKS, MODES, MODEL_SCALE, PROBES,
                           PULSE_FRAMES, SWITCH, summarize)


def scores(rows, start=0):
    dock = [x for row in rows for x in row['docks'][start:]]
    gate = [x for row in rows for x in row['gates'][start:]]
    first = [row['docks'][start] for row in rows]
    cap = DELIVERY_BLOCKS * BLOCK_FRAMES
    return dict(gates=sum(x is not None for x in gate), docks=sum(x is not None for x in dock),
                opportunities=len(dock), first_docks=sum(x is not None for x in first),
                first_docks_by_144=sum(x is not None and x <= 144 for x in first),
                capped_first_dock_frames=float(np.mean([cap if x is None else x for x in first])),
                edge_blocks=sum(row['edge_blocks'] for row in rows),
                rate_frames=sum(row['requested_rate_frames'] for row in rows))


def checked_rows(episodes):
    for ep in episodes:
        for change, strategies in ep['follow'].items():
            for mode, result in strategies.items():
                if summarize(result['blocks']) != result['summary']:
                    raise ValueError('fixture, mode, or stored score differs from the tape')
                if result['blocks'][:SWITCH] != ep['follow']['sham'][mode]['blocks'][:SWITCH]:
                    raise ValueError(f'{change} intervention differs before its hidden switch')
        for name in BASELINES:
            if summarize(ep['prime'][name]['blocks']) != ep['prime'][name]['summary']:
                raise ValueError('stored opening score differs from the tape')
            if (ep['prime'][name]['blocks'][:PROBES] !=
                    ep['follow']['sham']['continuous']['blocks'][:PROBES]):
                raise ValueError('the opening probe histories do not match')
        if ep['prime']['corrected']['summary'] != summarize(ep['follow']['sham']['continuous']['blocks'][:SWITCH]):
            raise ValueError('corrected opening summary differs from its matched tape')


def paired(first, second, index):
    absent = DELIVERY_BLOCKS * BLOCK_FRAMES + 1
    delta = np.asarray([x['docks'][index] or absent for x in first]) - np.asarray([
        x['docks'][index] or absent for x in second])
    return dict(earlier=int((delta < 0).sum()), tied=int((delta == 0).sum()), later=int((delta > 0).sum()))


def prime_metrics(episodes, priors_by_world_seed):
    result = {}
    for label in ('corrected', *BASELINES):
        errs = []
        for row in episodes:
            prior = priors_by_world_seed[row['world_seed']][label]
            probes = row['follow']['sham']['continuous']['blocks'][:PROBES]
            # The final RGB tool observation for block i arrives at block i + 1.
            following = row['follow']['sham']['continuous']['blocks'][1:PROBES + 1]
            for old, new in zip(probes, following, strict=True):
                displacement = MODEL_SCALE * (np.asarray(new['rgb'][0]) - old['rgb'][0])
                predicted = PULSE_FRAMES * prior @ old['u']
                errs.append(np.linalg.norm(displacement - predicted))
        result[label] = dict(scores([row['prime'][label]['summary'] for row in episodes]),
                             frozen_probe_mean_error_m=float(np.mean(errs)), frozen_probe_blocks=len(errs))
    for label in BASELINES:
        result[label]['paired_corrected'] = paired([row['prime']['corrected']['summary'] for row in episodes],
                                                   [row['prime'][label]['summary'] for row in episodes], 0)
    return result


def recovery_metrics(episodes):
    result = {}
    for change in CHANGES:
        result[change] = {}
        for mode in MODES:
            runs = [row['follow'][change][mode] for row in episodes]
            early = [[log for log in run['alarms'] if log['block'] < SWITCH + 1] for run in runs]
            later = [[log for log in run['alarms'] if SWITCH + 1 <= log['block'] <= SWITCH + DELIVERY_BLOCKS]
                     for run in runs]
            first = [log[0] for log in later if log]
            labels = {name: sum(log['links'] == links for log in first)
                      for name, links in [('motor', ['motor']), ('payload', ['payload']),
                                          ('both', ['motor', 'payload'])]}
            delays = [(log['block'] - SWITCH) * BLOCK_FRAMES for log in first if change in log['links']]
            result[change][mode] = dict(
                scores([summarize(run['blocks'][SWITCH:]) for run in runs]),
                pre_switch_alarm_courses=sum(bool(logs) for logs in early),
                first_alarm_by_link=dict(labels, no_alarm=len(runs) - len(first)),
                mean_frames_to_correct_first_alarm=float(np.mean(delays)) if delays else None,
            )
            if mode != 'selective':
                result[change][mode]['paired_selective'] = paired(
                    [row['follow'][change]['selective']['summary'] for row in episodes],
                    [run['summary'] for run in runs], BEFORE)
    return result


def shared_prefix(episodes):
    return [row for row in episodes if all(
        row['follow']['sham']['selective']['blocks'][:SWITCH] == row['follow']['sham'][mode]['blocks'][:SWITCH]
        for mode in MODES[1:]
    )]


def report(paths, output):
    episodes, priors = [], {}
    fixtures = {}
    for path in paths:
        data = _read(path)
        if not data['replayed'] or data['pilot'] or data['implementation_sha256'] != _hash(
                Path(__file__).with_name('tt17_payload.py')):
            raise ValueError('qualification requires final paired physical and policy replays')
        for key, digest in [('setup', 'setup_sha256'), ('checkpoint', 'checkpoint_sha256'),
                            ('signed_checkpoint', 'signed_sha256')]:
            if data[key] and _hash(Path(data[key])) != data[digest]:
                raise ValueError('a motor preparation input differs from its recorded hash')
        cases = data['episodes']
        checked_rows(cases)
        values = {k: np.asarray(v) for k, v in data['priors'].items()}
        new = {row['world_seed']: values for row in cases}
        if priors.keys() & new.keys() or len(new) != len(cases):
            raise ValueError('different held-out courses must have different world seeds')
        fixtures[path.name] = dict(sha256=_hash(path), courses=len(cases), optical=data['optical'],
                                   prime=prime_metrics(cases, new),
                                   recovery=recovery_metrics(cases))
        priors.update(new)
        episodes.extend(cases)
    summary = dict(schema='tt17-payload-summary-v1', courses=len(episodes), fixtures=fixtures,
                   prime=prime_metrics(episodes, priors), recovery=recovery_metrics(episodes),
                   replayed_tapes=len(episodes) * (len(CHANGES) * len(MODES) + len(BASELINES)),
                   replayed_blocks=len(episodes) * (len(CHANGES) * len(MODES) * (BEFORE + AFTER) * DELIVERY_BLOCKS
                                                    + len(BASELINES) * SWITCH))
    shared = shared_prefix(episodes)
    summary['matched_across_recovery_modes'] = dict(courses=len(shared), recovery=recovery_metrics(shared) if shared else {})
    _write(output, summary)
    print(summary['prime'])
    print(summary['recovery'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('fixtures', type=Path, nargs='+')
    args = parser.parse_args()
    report(args.fixtures, args.output)
