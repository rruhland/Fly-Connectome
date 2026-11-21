"""Link the SO-101 training checkpoint to paired gantry and native Pong receipts."""

import argparse
from pathlib import Path

import numpy as np

from .tt3_pong import state_sha256
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_retention import SOURCE, audit_installation, compare_gantry, compare_pong


def numeric_change(before, after):
    result = {}
    for key in ('coefficients', 'totals', 'counts'):
        first, second = np.asarray(before[key]), np.asarray(after[key])
        if first.shape == second.shape:
            difference = second - first
            result[key] = dict(shape=list(first.shape), changed=int(np.count_nonzero(difference)),
                               l2=float(np.linalg.norm(difference)), sum_change=float(difference.sum()))
        else:
            result[key] = dict(before_shape=list(first.shape), after_shape=list(second.shape))
    return result


def compare_memory(before, after):
    keys = ('rng', 'roles', 'clock', 'numeric', 'timing_choices', 'bounce', 'delay_belief', 'racket',
            'speed', 'reference', 'service_memory', 'service_registration', 'service_drive')
    series_keys = ('role_rows', 'flight_history', 'flight_impulses')
    changed = [key for key in keys if before.get(key) != after.get(key)]
    changed += [f'series.{key}' for key in series_keys if before['series'].get(key) != after['series'].get(key)]
    common_before, common_after = before['motor_span'], after['motor_span']
    profiles_before, profiles_after = common_before['profiles'], common_after['profiles']
    bounce = [state['bounce'] for state in (before, after)]
    return dict(legacy_fields_changed=changed, gantry_address_equal=audit_installation(before) == audit_installation(after),
                all_original_keys_equal=all(after.get(key) == value for key, value in before.items()),
                legacy_model_rows=len(before['models']), endpoint_model_rows=len(after['models']),
                legacy_model_prefix_equal=after['models'][:len(before['models'])] == before['models'],
                shared_numeric_change=numeric_change(common_before['shared'], common_after['shared']),
                old_profile_numeric_change={key: numeric_change(profiles_before[key], profiles_after[key])
                                            for key in sorted(profiles_before.keys() & profiles_after.keys())},
                new_profiles=sorted(profiles_after.keys() - profiles_before.keys()),
                removed_profiles=sorted(profiles_before.keys() - profiles_after.keys()),
                bounce_sample_counts=[len(samples) for samples in bounce],
                recent_bounce_medians=[float(np.median(samples[-12:])) if samples else None for samples in bounce],
                flight_history_counts=[len(state['series']['flight_history']) for state in (before, after)],
                flight_impulse_counts=[len(state['series']['flight_impulses']) for state in (before, after)],
                new_endpoint_sections=sorted(after.keys() - before.keys()))


def training_witness(receipt, endpoint, endpoint_hash, source_hash):
    if 'tt19_online' in endpoint:
        recorded = receipt['games'][-1]['checkpoint_sha256']
        stored_origin = endpoint['tt19_online']['donor_sha256']
        end_field = f"games.{len(receipt['games']) - 1}.checkpoint_sha256"
    elif 'tt19_stroke' in endpoint:
        recorded = receipt['checkpoint_sha256']
        stored_origin = endpoint['tt19_stroke']['gantry_checkpoint_sha256']
        end_field = 'checkpoint_sha256'
    else:
        raise ValueError('the checkpoint contains no SO-101 learner')
    if (recorded, receipt.get('donor_sha256'), stored_origin) != (endpoint_hash, source_hash, source_hash):
        raise ValueError('SO-101 training receipt does not connect the source and endpoint')
    return dict(endpoint_field=end_field, donor_field='donor_sha256')


def training_change(endpoint, receipt):
    if 'tt19_online' in endpoint:
        model = endpoint['tt19_online']
        motor = model['motor']
        weights = np.asarray(motor['weights'])
        return dict(kind='online_inverse_model', motor_samples=motor['samples'], motor_updates=motor['updates'],
                    scored_samples=sum(row['samples_added'] for game in receipt['games'] for row in game['points']),
                    weight_shape=list(weights.shape), weight_nonzero=int(np.count_nonzero(weights)),
                    weight_l2_from_zero=float(np.linalg.norm(weights)), selected_lag=motor['lag'],
                    optical_shot_samples=model['shot_samples'], optical_bounce_samples=len(model['bounce_samples']),
                    game_scores=[game['score'] for game in receipt['games']],
                    game_legal_returns=[game['metrics']['total_legal_returns'] for game in receipt['games']],
                    replayed_points=receipt['replayed_points'])
    if 'tt19_stroke' in endpoint:
        archive = endpoint['tt19_stroke']['policy']['archive']
        by_role = {}
        for role, samples in archive.items():
            best = max(samples, key=lambda item: item['quality'], default=None)
            by_role[role] = dict(samples=len(samples), detected_contacts=sum(sample['contact'] for sample in samples),
                                 best_stroke=None if best is None else best['stroke'])
        return dict(kind='direct_stroke_archive', archive=by_role, practice=receipt['practice_summary'],
                    evaluation=receipt['test_summary'], replayed_points=receipt['replayed_points'])
    raise ValueError('the endpoint does not contain an SO-101 learner')


def compare(folder, label, endpoint, training):
    source_hash, endpoint_hash = _hash(SOURCE), _hash(endpoint)
    if source_hash == endpoint_hash:
        raise ValueError('expected a distinct training endpoint')
    before, after = _read(SOURCE), _read(endpoint)
    training_receipt = _read(training)
    lineage = training_witness(training_receipt, after, endpoint_hash, source_hash)
    gantry_before, gantry_after = (_read(folder / f'retention-gantry-{part}.json.gz')
                                   for part in ('before', f'after-{label}'))
    pong_before, pong_after = (_read(folder / f'retention-pong-{part}.json.gz')
                               for part in ('before', f'after-{label}'))
    identity = _read(folder / 'retention-gantry-identity.json.gz')
    if any(receipt['checkpoint_sha256'] != digest for receipt, digest in
           ((gantry_before, source_hash), (pong_before, source_hash),
            (gantry_after, endpoint_hash), (pong_after, endpoint_hash), (identity, source_hash))):
        raise ValueError('a scored receipt used the wrong checkpoint')
    endpoint_ref = endpoint.resolve()
    training_ref = training.resolve()
    folder_ref = folder.resolve()
    result = dict(schema='tt19-cross-task-retention-v1', mode=label, source_sha256=source_hash,
                  endpoint=str(endpoint_ref.relative_to(folder_ref)) if endpoint_ref.is_relative_to(folder_ref) else str(endpoint_ref),
                  endpoint_sha256=endpoint_hash,
                  training=str(training_ref.relative_to(folder_ref)) if training_ref.is_relative_to(folder_ref) else str(training_ref),
                  training_sha256=_hash(training), lineage=lineage,
                  learning=training_change(after, training_receipt), memory=compare_memory(before, after),
                  gantry=compare_gantry(gantry_before, gantry_after),
                  gantry_identity_control=compare_gantry(gantry_before, identity),
                  pong=compare_pong(pong_before, pong_after))
    addressed_path = folder / f'retention-gantry-recalled-{label}.json.gz'
    if addressed_path.exists():
        addressed = _read(addressed_path)
        world = dict(bounce=before['bounce'], flight_history=before['series']['flight_history'],
                     flight_impulses=before['series']['flight_impulses'])
        if (addressed['checkpoint_sha256'] != endpoint_hash or addressed['world_address'] != 'gantry'
                or addressed['stored_world_sha256'] != state_sha256(world)):
            raise ValueError('the addressed return did not use the retained source table evidence')
        result['gantry_addressed_world'] = compare_gantry(gantry_before, addressed)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    parser.add_argument('label')
    parser.add_argument('endpoint', type=Path)
    parser.add_argument('training', type=Path)
    args = parser.parse_args()
    report = compare(args.folder, args.label, args.endpoint, args.training)
    _write(args.folder / f'retention-{args.label}-summary.json.gz', report)
