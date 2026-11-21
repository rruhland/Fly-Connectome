"""Assemble complete native rollout qualification and the matched source control."""

import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from .tt4_transfer_trial import _hash, _read, _write
from .tt19_catalog import dry_catalog
from .tt19_model_rollout_trial import load_checkpoint, replay, replay_optical


def practice_budget(folder):
    context = folder / 'context-local-practice.json.gz'
    bank = folder / 'consolidated-local-bank.json.gz'
    search = _read(context)['rows']
    consolidated = _read(bank)
    if consolidated['source_sha256'] != _hash(context):
        raise ValueError('the consolidated serving bank names different practice')
    by_role = {}
    for role in (1, -1):
        base = [row for row in search if row[1] == role]
        samples = consolidated['banks'][str(role)]['trials']
        by_role[str(role)] = dict(search_points=len(base), qualification_points=len(samples),
                                  search_frames=sum(row[2] for row in base),
                                  qualification_frames=sum(row[2] for row in samples))
    seeds = ([row[0] for row in search] + [row[1] for role in consolidated['banks'].values()
                                         for row in role['trials']])
    return by_role, set(seeds)


def motion_accuracy(games):
    errors, observed, legal_expected, legal_observed = [], 0, 0, 0
    for game in games:
        for row in game['points']:
            if row['server'] != -1 or not row['plans']:
                continue
            plan = row['plans'][0]
            observed += 1
            if plan['legal']:
                legal_expected += 1
                legal_observed += row['result']['legal_returns'] > 0
            contact = next((event for event in row['events'] if event['kind'] == 'contact'
                            and event.get('side') == 1), None)
            if contact is not None:
                errors.append(abs(contact['time'] * 30. - (plan['tick'] + plan['eta'])))
    return dict(planned_receiving_points=observed, legal_predictions=legal_expected,
                returns_after_legal_prediction=legal_observed, contact_samples=len(errors),
                median_contact_error_frames=None if not errors else float(np.median(errors)))


def verify_atlas(folder):
    original = folder / 'factory-dry-catalog.npz'
    with TemporaryDirectory(prefix='tt19-dry-atlas-') as work:
        rebuilt = Path(work) / 'atlas.npz'
        build = dry_catalog(rebuilt, 2048)
        with np.load(original) as a, np.load(rebuilt) as b:
            equal = a.files == b.files and all(np.array_equal(a[key], b[key]) for key in a.files)
        left, right = _hash(original), _hash(rebuilt)
        receipt = dict(original_sha256=left, reproduced_sha256=right, arrays_equal=equal,
                       compressed_bytes_equal=left == right, build=build)
    _write(folder / 'model-rollout-atlas-replay.json.gz', receipt)
    if not (receipt['compressed_bytes_equal'] and equal):
        raise ValueError('the rebuilt dry atlas differs from the published archive')
    return receipt


def matched_motion(warm, nominal):
    rows = []
    for a, b in zip(warm, nominal, strict=True):
        cold_points = {row['seed']: row for row in b['points'] if row['server'] == -1}
        pairs = [(row, cold_points[row['seed']]) for row in a['points']
                 if row['server'] == -1 and row['seed'] in cold_points]
        rows.append(dict(seed=a['seed'], warm_score=a['score'], nominal_score=b['score'],
                         matched_receiving_points=len(pairs),
                         changed_actions=sum(x['actions'] != y['actions'] for x, y in pairs),
                         changed_events=sum(x['events'] != y['events'] for x, y in pairs)))
    return rows


def assemble(folder):
    archive = folder / 'factory-dry-catalog.npz'
    checkpoint = folder / 'match-model-rollout-native-player.json.gz'
    paths = [folder / f'match-model-rollout{suffix}-native.json.gz' for suffix in ('', '-extra', '-nominal')]
    initial, extra, nominal = map(_read, paths)
    baseline = folder / 'match-model-rollout-analytic-native.json.gz'
    analytic = _read(baseline)
    atlas_receipt = folder / 'model-rollout-atlas-replay.json.gz'
    atlas_version = _read(atlas_receipt)
    if not atlas_version['compressed_bytes_equal'] or atlas_version['original_sha256'] != _hash(archive):
        raise ValueError('independent reconstruction did not match the archival dry atlas')
    sources = (*paths, baseline, atlas_receipt)
    for report in (initial, extra, nominal, analytic):
        if report['cache_sha256'] != _hash(archive) or report['bank_source'] != 'local' or not (
            report['native'] and report['strict_endline'] and report['course'] == 'full'
        ):
            raise ValueError('a full native factory qualification used a different bench or checkpoint')
        if report.get('replayed_points') != sum(len(g['points']) for g in report['games']):
            raise ValueError('a source qualification has unverified physical tapes')
    if initial['seeds'] != analytic['seeds'] or extra['seeds'] != nominal['seeds']:
        raise ValueError('the analytic or source ablation lacks its matched game starts')
    if not (initial['source'] and extra['source'] and analytic['source']) or nominal['source']:
        raise ValueError('the transfer control comparison used different source priors')
    if (extra['checkpoint_sha256'] != _hash(checkpoint) or
            len(extra['games']) != len(extra['seeds']) or len(nominal['games']) != len(nominal['seeds'])):
        raise ValueError('the extra games did not use the complete frozen rollout checkpoint')
    trained, earlier = practice_budget(folder)
    combined = [*initial['games'], *extra['games']]
    warm = dict(seeds=initial['seeds'] + extra['seeds'], games=combined,
                internal_rollouts=initial['internal_rollouts'] + extra['internal_rollouts'],
                seconds=initial['seconds'] + extra['seconds'], motion=motion_accuracy(combined))
    if len(set(warm['seeds'])) != len(warm['seeds']):
        raise ValueError('a recorded warm game was repeated')
    all_games = combined + nominal['games'] + analytic['games']
    if earlier.intersection(point['seed'] for game in all_games for point in game['points']):
        raise ValueError('a scored point was already in the TT19 serving bank practice')
    warm['replayed_points'] = replay(warm)
    warm['game_wins'] = sum(g['score'][0] > g['score'][1] for g in combined)
    policy = load_checkpoint(checkpoint, archive)
    try:
        warm['optical_command_replays'] = replay_optical(policy, warm)
    finally:
        policy.player.close()
    nominal['motion'] = motion_accuracy(nominal['games'])
    result = dict(schema='tt19-model-rollout-qualification-1', warm=warm, nominal_matched=nominal,
                  analytic_matched=analytic, source_receipts={p.name: _hash(p) for p in sources},
                  matched_motion=matched_motion(extra['games'], nominal['games']),
                  serving_practice=trained, catalogue_sha256=_hash(archive),
                  atlas_movements=2048, atlas_control_frames=40960,
                  checkpoint=checkpoint.name, checkpoint_sha256=_hash(checkpoint))
    _write(folder / 'model-rollout-qualification.json.gz', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('--rebuild-atlas', action='store_true')
    args = parser.parse_args()
    if args.rebuild_atlas:
        verify_atlas(args.folder)
    assemble(args.folder)
