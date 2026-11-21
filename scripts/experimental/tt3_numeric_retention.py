"""Test native Pong before and after two preselected table-tennis courses."""

import argparse
from dataclasses import asdict
import gzip
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch

from .tt3_numeric_transfer import PortableMemory, run_match, summary as table_summary
from .tt3_pong import PongCase, assess_retention, evaluate_pong, paired_pong, state_sha256, summarize_pong
from .tt3_visual import RoleModel


EXTENDED_SEED = 17_041_069
EXTENDED_POINTS = 48
QUALIFICATION_SEED = 15_537_921
QUALIFICATION_MATCH = 0


def _read(path):
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as handle:
        return json.load(handle)


def _write(path, row):
    payload = gzip.compress((json.dumps(row, allow_nan=False, separators=(',', ':')) + '\n').encode(), mtime=0)
    temporary = path.with_suffix('.tmp')
    temporary.write_bytes(payload)
    temporary.replace(path)


def _hashes():
    directory = Path(__file__).resolve().parent
    inputs = [directory / f'tt3_{name}.py' for name in ('arena', 'continuous', 'flight', 'learning', 'motor',
              'numeric', 'numeric_retention', 'numeric_transfer', 'pong', 'visual')]
    inputs.append(directory.parents[1] / 'src/fly_connectome/pong.py')
    return {path.name: sha256(path.read_bytes()).hexdigest() for path in inputs}


def _native(schedule, state):
    cases = [PongCase(**row) for row in schedule['evaluation']]
    memory = PortableMemory.restore(state['memory'])
    roles = RoleModel(**state['roles'])
    digest = state_sha256(state)
    rows = evaluate_pong(cases, roles, memory, points_per_match=schedule['evaluation_points'],
                         opponent_speed=schedule['opponent_speed'], max_ticks=schedule['max_ticks'])
    if digest != state_sha256(dict(memory=memory.state(), roles=asdict(roles))):
        raise ValueError('frozen return-to-Pong evaluation changed learner state')
    return dict(summary=summarize_pong(rows), points=rows)


def _portable_state(row):
    return {key: row[key] for key in ('roles', 'memory')}


def _course(seed, count, state):
    return run_match((0, 'standard', state['roles'], state['memory'], ('trained',), count, seed, True, True))['trained']


def prepare(roles_path, memory_path, schedule_receipt, work):
    hashes = _hashes()
    roles_checkpoint = _read(roles_path)
    canonical = _read(schedule_receipt)
    if roles_checkpoint['role_model'] != canonical['source']['role_model']:
        raise ValueError('the numeric player and the source schedule require the same role model')
    schedule = canonical['schedule']
    if (len(schedule['evaluation']), schedule['evaluation_points']) != (36, 5):
        raise ValueError('the retained cohort must use the canonical 36-by-five schedule')
    if roles_checkpoint['schedule_sha256'] != state_sha256(schedule):
        raise ValueError('the canonical source does not match the held-out native Pong schedule')
    source = dict(roles=roles_checkpoint['role_model'], memory=_read(memory_path))
    manifest = dict(source=source, schedule=schedule, source_sha256=state_sha256(source),
                    schedule_sha256=state_sha256(schedule), inputs_sha256=hashes,
                    extended_seed=EXTENDED_SEED, extended_points=EXTENDED_POINTS,
                    qualification_seed=QUALIFICATION_SEED, qualification_match=QUALIFICATION_MATCH)
    before = _native(schedule, source)
    _write(work / 'before.json.gz', dict(manifest=manifest, before=before))
    print('Native Pong before table:', json.dumps(before['summary']), flush=True)
    extended = _course(EXTENDED_SEED, EXTENDED_POINTS, source)
    _write(work / 'extended-table.json.gz', extended)
    print('Extended table:', json.dumps(table_summary([{'trained': extended}])['trained']), flush=True)
    received = _evaluate_course(before, schedule, source, extended)
    if hashes != _hashes():
        raise ValueError('experiment source changed during the initial run')
    _write(work / 'extended.json.gz', received)
    print('Native Pong after extended table:', json.dumps(received['retention']['after']), flush=True)


def _evaluate_course(before, schedule, source, course):
    state = _portable_state(course)
    post = _native(schedule, state)
    retention = assess_retention(before['points'], post['points'])
    initial_memory, final_memory = source['memory'], state['memory']
    if initial_memory['coefficients'] != final_memory['coefficients']:
        raise ValueError('the table learner changed its geometry bank outside the documented score updates')
    received_rewards = np.asarray(final_memory['totals']) - initial_memory['totals']
    option_counts = np.asarray(final_memory['counts']) - initial_memory['counts']
    counts = np.bincount([row['option_index'] for row in course['points']], minlength=len(option_counts))
    totals = np.bincount([row['option_index'] for row in course['points']],
                         weights=[row['public_reward'] for row in course['points']], minlength=len(option_counts))
    if not np.array_equal(option_counts, counts) or not np.array_equal(received_rewards, totals):
        raise ValueError('returned memory does not account for every table choice and native reward')
    retention['table_updates'] = state_sha256(source) != state_sha256(state) and int(option_counts.sum()) > 0
    retention['established'] &= retention['table_updates']
    retention['behavior_changes'] = {key: sum(p[key] != q[key] for p, q in zip(before['points'], post['points'], strict=True))
                                     for key in ('relation', 'action_sha256', 'contacts', 'score')}
    result = dict(state_sha256=state_sha256(state), updated_counts=option_counts.tolist(),
                  updated_totals=received_rewards.tolist(), post_state=state, retention=retention,
                  after=post, table=course, table_summary=table_summary([{'trained': course}])['trained'])
    if not retention['established']:
        controls = {'table_scores_source_roles': dict(roles=source['roles'], memory=state['memory']),
                    'source_scores_table_roles': dict(roles=state['roles'], memory=source['memory'])}
        result['interference_controls'] = {}
        for key, control_state in controls.items():
            if state_sha256(control_state) == state_sha256(state):
                rows = post
            elif state_sha256(control_state) == state_sha256(source):
                rows = before
            else:
                rows = _native(schedule, control_state)
            result['interference_controls'][key] = dict(**rows, paired_to_before=paired_pong(before['points'], rows['points']))
    return result


def finish(work, qualification_path, output):
    prepared = _read(work / 'before.json.gz')
    manifest = prepared['manifest']
    if manifest['inputs_sha256'] != _hashes():
        raise ValueError('native Pong must use the same code before and after transfer')
    qualification = _read(qualification_path)
    if (qualification['seed'], qualification['variant'], qualification['stable'], qualification['forecast']) != (
            QUALIFICATION_SEED, 'standard', True, True):
        raise ValueError('the supplied qualification does not use the registered table installation')
    if len(qualification['matches']) != 10 or hash_source(qualification) != manifest['source_sha256']:
        raise ValueError('the receiving qualification uses another source checkpoint or course count')
    course = qualification['matches'][QUALIFICATION_MATCH]['trained']
    if len(course['points']) != 14:
        raise ValueError('the first qualified receiving match must contain fourteen points')
    received = _evaluate_course(prepared['before'], manifest['schedule'], manifest['source'], course)
    if manifest['inputs_sha256'] != _hashes():
        raise ValueError('experiment source changed during retention evaluation')
    extended = _read(work / 'extended.json.gz')
    _write(output, dict(**prepared, qualification_sha256=sha256(qualification_path.read_bytes()).hexdigest(),
                        qualification_match=received, extended_match=extended))
    for label, item in (('qualification', received), ('extended', extended)):
        print(label, json.dumps(dict(after=item['retention']['after'], established=item['retention']['established'],
                                     baseline=item['retention']['capable_baseline'], limits=item['retention']['noninferiority'])),
              flush=True)


def hash_source(qualification):
    return state_sha256(dict(roles=qualification['roles'], memory=qualification['source']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    stages = parser.add_subparsers(dest='stage', required=True)
    initial = stages.add_parser('prepare')
    initial.add_argument('roles', type=Path)
    initial.add_argument('memory', type=Path)
    initial.add_argument('schedule', type=Path)
    initial.add_argument('work', type=Path)
    final = stages.add_parser('finish')
    final.add_argument('work', type=Path)
    final.add_argument('qualification', type=Path)
    final.add_argument('output', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.stage == 'prepare':
        args.work.mkdir(parents=True, exist_ok=False)
        prepare(args.roles, args.memory, args.schedule, args.work)
    else:
        finish(args.work, args.qualification, args.output)


if __name__ == '__main__':
    main()
