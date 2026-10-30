"""Independent, read-only reporting of chronological carried-learning receipts.

No environment or controller is imported. Fixed tails and phase-anchored recovery
bins are reporting only; this program cannot select a model or update a learner.
"""
import argparse
import ast
from bisect import bisect_left
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

RECEIPT = re.compile(r'game-(\d+)\.json\Z')
FEEDBACK_FIELDS = {'own_hits', 'reward', 'terminated', 'truncated', 'elapsed_seconds'}


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            value.update(chunk)
    return value.hexdigest()


def state_digest(state):
    return hashlib.sha256(json.dumps(state, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def load_receipts(directory):
    """Miss-tail JSON lists, checkpoint files and summaries are never receipts."""
    rows, seen = [], set()
    for path in Path(directory).iterdir():
        match = RECEIPT.fullmatch(path.name)
        if not match:
            continue
        row = read(path)
        index = int(match.group(1))
        if index in seen or not isinstance(row, dict) or row['assignment']['index'] != index:
            raise ValueError(f'ambiguous or misnamed game receipt: {path}')
        seen.add(index)
        rows.append(row)
    return sorted(rows, key=lambda row: row['assignment']['index'])


def rates(rows):
    events = [event for row in rows for event in row['events']]
    first = [event for event in events if event['first']]
    later = [event for event in events if not event['first']]
    result = dict(games=len(rows), completed_games=sum(bool(row['complete']) for row in rows))
    for prefix, selected in (('all', events), ('first', first), ('later', later)):
        hits = sum(bool(event['hit']) for event in selected)
        result.update({f'{prefix}_returns': hits, f'{prefix}_opportunities': len(selected),
            f'{prefix}_rate': hits / len(selected) if selected else None})
    return result


def costs(rows):
    seconds = sum(row['report']['physics_seconds'] for row in rows)
    hits = sum(row['report']['hits'] for row in rows)
    return dict(physics_seconds=seconds, own_hits=hits,
        hits_per_physical_second=hits / seconds if seconds else None,
        transition_cameras=sum(row['transitions'] for row in rows),
        initial_cameras=sum(row['initial_cameras'] for row in rows),
        transition_camera_upper_bound=sum(row['transition_camera_upper_bound'] for row in rows),
        initial_camera_upper_bound=sum(row['initial_camera_upper_bound'] for row in rows),
        receipt_wall_seconds=sum(row['wall_seconds'] for row in rows),
        natural_ends=sum(row['report']['losses'] + row['report']['wins'] for row in rows),
        capped_games=sum(row['report']['truncations'] for row in rows),
        starts=len(rows), completed_games=sum(bool(row['complete']) for row in rows),
        incomplete_games=sum(not row['complete'] for row in rows),
        unknown_unsaved_work_excluded=True)


def check_source(spec, root, source_root):
    problems, checked = [], 0
    for group in ('files', 'assets'):
        for name, expected in spec[group].items():
            checked += 1
            if not Path(name).is_file() or digest(name) != expected:
                problems.append(f'{group} hash mismatch: {name}')
    for name, expected in spec['source'].items():
        checked += 1
        path = source_root / name
        if not path.is_file() or digest(path) != expected:
            problems.append(f'package source hash mismatch: {name}')
    actual_source = [source_root / 'pong.py', *sorted((source_root / 'control').glob('*.py')),
        *sorted((source_root / 'vision').glob('*.py'))]
    if {str(path.relative_to(source_root)) for path in actual_source} != set(spec['source']):
        problems.append('package source manifest gained/lost files')
    vision_checked = False
    session_source = source_root / 'control/session.py'
    if session_source.is_file():
        for node in ast.parse(session_source.read_text()).body:
            if (isinstance(node, ast.Assign) and any(isinstance(target, ast.Name)
                    and target.id == 'MODEL_HASH' for target in node.targets)):
                model = source_root / 'vision/data/default.pt'
                vision_checked = True
                checked += 1
                if not model.is_file() or digest(model) != ast.literal_eval(node.value):
                    problems.append('inherited vision model hash mismatch')
    for filename, key in (('schedules.json', 'schedules_sha256'),
                          ('seed-exclusion-proof.json', 'exclusions_sha256')):
        checked += 1
        if digest(root / filename) != spec[key]:
            problems.append(f'{filename} hash mismatch')
    identity = dict(spec)
    identity.pop('identity')
    if state_digest(identity) != spec['identity']:
        problems.append('spec identity does not match its contents')
    boundary = {}
    for name in spec['files']:
        if Path(name).name == 'pong_carried_structure_learner.py':
            tree = ast.parse(Path(name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name in ('begin_game', 'observe'):
                    positional = [arg.arg for arg in node.args.posonlyargs + node.args.args]
                    keywords = {arg.arg for arg in node.args.kwonlyargs}
                    boundary[node.name] = (positional == ['self'] and node.args.vararg is None
                        and node.args.kwarg is None and keywords == (
                            FEEDBACK_FIELDS if node.name == 'observe' else set()))
        if Path(name).name == 'run_pong_carried_learning.py':
            tree = ast.parse(Path(name).read_text())
            for node in tree.body:
                if isinstance(node, ast.FunctionDef) and node.name == 'feedback':
                    calls = [child for child in ast.walk(node) if isinstance(child, ast.Call)
                        and isinstance(child.func, ast.Name) and child.func.id == 'dict']
                    boundary['feedback_whitelist'] = len(calls) == 1 and {
                        kw.arg for kw in calls[0].keywords} == FEEDBACK_FIELDS
    if not all(boundary.get(key, False) for key in ('begin_game', 'observe', 'feedback_whitelist')):
        problems.append('learner signature/feedback whitelist differs from reviewed boundary')
    return dict(passed=not problems, checked_files=checked, problems=problems,
        inherited_vision_model_hash_checked=vision_checked,
        learner_boundary_signature_checks=boundary,
        limitation='hash/signature verification plus prior manual review; not a formal noninterference proof')


def audit_stream(rows, assignments, spec, directory):
    problems, boundaries, opportunities = [], [], {}
    expected = {assignment['index']: assignment for assignment in assignments}
    phase_elapsed = phase_cameras = phase_initials = 0
    previous = None
    tape_files = set()
    for row in rows:
        assignment = row['assignment']
        index, events = assignment['index'], row['events']
        label = f'game {index}'
        if expected.get(index) != assignment:
            problems.append(f'{label}: prospective assignment mismatch')
        if row['runtime_identity'] != spec['identity']:
            problems.append(f'{label}: runtime identity mismatch')
        for side in ('before', 'after'):
            if state_digest(row[f'learner_{side}']) != row[f'learner_{side}_sha256']:
                problems.append(f'{label}: {side} checkpoint hash mismatch')
        before, after = row['learner_before'], row['learner_after']
        if previous is not None:
            contiguous = index == previous['assignment']['index'] + 1
            unchanged = before == previous['learner_after']
            if not contiguous or not unchanged:
                problems.append(f'{label}: broken carried-state boundary')
            if assignment['phase'] != previous['assignment']['phase']:
                boundaries.append(dict(from_phase=previous['assignment']['phase'],
                    to_phase=assignment['phase'], next_game=index,
                    exact_state_carried=unchanged and contiguous,
                    lifetime_successes_carried=sum(before['lifetime_success']),
                    lifetime_failures_carried=sum(before['lifetime_failure']),
                    nonzero_reward_evidence_carried=bool(sum(before['lifetime_success'])
                        + sum(before['lifetime_failure'])),
                    persistent_learner=before['kind'] != 'reset_beta',
                    rng_state_sha256=state_digest(before['rng_state'])))
        if previous is None or assignment['phase'] != previous['assignment']['phase']:
            phase_elapsed = phase_cameras = phase_initials = 0
        if row['complete'] and (after['active_arm'] is not None or after['games'] != before['games'] + 1):
            problems.append(f'{label}: completed game did not close exactly once')
        annotation = row['annotation']
        hit_count = sum(bool(event['hit']) for event in events)
        first = [event for event in events if event['first']]
        if (len(events) != annotation['own_opportunities'] or hit_count != annotation['own_returns']
                or hit_count != row['report']['hits'] or len(first) != bool(events)
                or (events and not events[0]['first'])):
            problems.append(f'{label}: opportunity/first/hit counts disagree')
        if row['complete'] and not row['streaming_audit']['passed']:
            problems.append(f'{label}: completed streaming audit failed')
        if not row['frozen_execution_parameters_checked']:
            problems.append(f'{label}: execution parameters were not checked frozen')
        if before['kind'] != 'reset_beta':
            enabled = before['updates_enabled']
            expected_hits = hit_count if enabled else 0
            expected_losses = row['report']['losses'] if enabled else 0
            if (sum(after['lifetime_success']) - sum(before['lifetime_success']) != expected_hits
                    or sum(after['lifetime_failure']) - sum(before['lifetime_failure']) != expected_losses):
                problems.append(f'{label}: retained lifetime evidence does not match actual public feedback')
        path = Path(row['tape_file'])
        tape_files.add(path.resolve())
        if digest(path) != row['tape_sha256']:
            problems.append(f'{label}: event tape hash mismatch')
        with np.load(path, allow_pickle=False) as tape:
            ticks = np.cumsum(tape['physics_ticks']).tolist()
            durations = tape['elapsed_seconds']
            if (len(ticks) != row['transitions'] or int(tape['hits'].sum()) != hit_count
                    or not math.isclose(float(durations.sum()), row['report']['physics_seconds'], abs_tol=1e-8)
                    or tape['opportunity_tick'].tolist() != [event['tick'] for event in events]
                    or tape['opportunity_hit'].tolist() != [event['hit'] for event in events]
                    or tape['opportunity_first'].tolist() != [event['first'] for event in events]):
                problems.append(f'{label}: tape/receipt counts or timing disagree')
            physical = 2 - tape['action'] if assignment['reverse'] else tape['action']
            if (not np.array_equal(physical, tape['physical_action'])
                    or not np.isin(tape['action'], [0, 1, 2]).all()):
                problems.append(f'{label}: physical/requested action mapping mismatch')
            dt = float(durations.sum()) / ticks[-1] if ticks and ticks[-1] else 1 / 120
            selected = []
            for event in events:
                transition = bisect_left(ticks, event['tick']) + 1
                if transition > len(ticks):
                    problems.append(f'{label}: opportunity beyond saved physical ticks')
                selected.append(dict(hit=bool(event['hit']), first=bool(event['first']),
                    game=index + 1, phase_game=assignment['phase_index'] + 1,
                    phase_seconds=phase_elapsed + event['tick'] * dt,
                    phase_transition_cameras=phase_cameras + transition,
                    phase_initial_cameras=phase_initials + row['initial_cameras']))
            opportunities[index] = selected
        phase_elapsed += row['report']['physics_seconds']
        phase_cameras += row['transitions']
        phase_initials += row['initial_cameras']
        previous = row
    if rows:
        for filename, observed in (('checkpoint-initial.json', rows[0]['learner_before']),
                                   ('checkpoint-latest.json', rows[-1]['learner_after'])):
            if not (directory / filename).is_file() or read(directory / filename) != observed:
                problems.append(f'{filename} does not match receipt endpoint')
        if rows[0]['assignment']['index'] != 0:
            problems.append('first scheduled game is missing')
    orphans = [str(path) for path in directory.glob('game-*-events.npz')
        if path.resolve() not in tape_files]
    if orphans:
        problems.append('event tapes exist without game receipts; exposure may be missing')
    return dict(passed=not problems, checked_receipts=len(rows), problems=problems,
        carried_phase_boundaries=boundaries, orphan_event_tapes=orphans), opportunities


def recovery(events, threshold=.95, window=25):
    """Earliest two passing fixed, nonoverlapping phase-anchored bins."""
    bins = []
    for start in range(0, len(events) - window + 1, window):
        selected = events[start:start + window]
        rate = sum(event['hit'] for event in selected) / window
        bins.append(dict(first_opportunity=start + 1, last_opportunity=start + window,
            returns=sum(event['hit'] for event in selected), rate=rate,
            passed=rate >= threshold, end=selected[-1]))
    confirmed = next((i for i in range(1, len(bins))
        if bins[i - 1]['passed'] and bins[i]['passed']), None)
    return dict(definition='consecutive fixed phase-anchored nonoverlapping25-opportunity bins',
        threshold=threshold, opportunity_window=window, observed_opportunities=len(events),
        confirmed=confirmed is not None,
        confirmation_opportunity=bins[confirmed]['last_opportunity'] if confirmed is not None else None,
        confirmation=bins[confirmed]['end'] if confirmed is not None else None,
        first_passing_window_end=bins[confirmed - 1]['end'] if confirmed is not None else None,
        complete_bins=bins, trailing_incomplete_opportunities=len(events) % window,
        no_endpoint_or_window_selection_for_acceptance=True)


def paired_controls(receipts, schedules):
    """Join only explicitly paired control assignments; never infer a pairing."""
    comparisons = {}
    lookup = {name: {row['assignment']['index']: row for row in rows}
        for name, rows in receipts.items()}
    for name, assignments in schedules.items():
        matched = [assignment for assignment in assignments if 'matched_live_index' in assignment]
        if not matched:
            continue
        phases, problems = {}, []
        for phase in dict.fromkeys(assignment['phase'] for assignment in matched):
            expected = [assignment for assignment in matched if assignment['phase'] == phase]
            live, control, differences = [], [], []
            for assignment in expected:
                target = assignment.get('matched_live_stream')
                if target is None:
                    problems.append('paired assignment does not name its matched_live_stream')
                    continue
                left = lookup.get(target, {}).get(assignment['matched_live_index'])
                right = lookup.get(name, {}).get(assignment['index'])
                if left is None or right is None:
                    continue
                keys = ('seed', 'reverse', 'max_episode_ticks', 'max_transition_cameras')
                if right['assignment'] != assignment or any(
                        left['assignment'][key] != assignment[key] for key in keys):
                    problems.append(f"{name} game {assignment['index']}: paired start/duration mismatch")
                live.append(left)
                control.append(right)
                differences.append(left['report']['hits'] - right['report']['hits'])
            phases[phase] = dict(scheduled_pairs=len(expected), observed_pairs=len(differences),
                complete=len(differences) == len(expected) and all(
                    row['complete'] for row in live + control), live=rates(live), control=rates(control),
                paired_mean_own_hit_difference=sum(differences) / len(differences) if differences else None,
                live_more_hits=sum(value > 0 for value in differences),
                control_more_hits=sum(value < 0 for value in differences),
                ties=sum(value == 0 for value in differences))
        comparisons[name] = dict(problems=problems, matching_valid=not problems, phases=phases,
            uncertainty='Descriptive paired games; no independent-contact confidence interval asserted.')
    return comparisons


def report_experiment(root, source_root=None, tail_games=32, min_later=100, threshold=.95):
    root = Path(root)
    spec, schedules = read(root / 'spec.json'), read(root / 'schedules.json')
    source_root = Path(source_root) if source_root else Path(__file__).parents[2] / 'src/fly_connectome'
    source = check_source(spec, root, source_root)
    streams, receipts = {}, {}
    for kind, assignments in schedules.items():
        directory = root / kind
        if not directory.is_dir():
            continue
        rows = load_receipts(directory)
        receipts[kind] = rows
        audit, opportunities = audit_stream(rows, assignments, spec, directory)
        phases = {}
        for phase in dict.fromkeys(assignment['phase'] for assignment in assignments):
            expected = [assignment for assignment in assignments if assignment['phase'] == phase]
            selected = [row for row in rows if row['assignment']['phase'] == phase]
            tail_indices = {assignment['index'] for assignment in expected[-tail_games:]}
            tail = [row for row in selected if row['assignment']['index'] in tail_indices]
            complete_tail = (len(expected) >= tail_games and len(tail) == tail_games
                and all(row['complete'] for row in tail))
            metrics = rates(tail)
            conditions = dict(exact_fixed_tail_complete=complete_tail,
                enough_later_opportunities=metrics['later_opportunities'] >= min_later,
                first_return_rate=metrics['first_rate'] is not None and metrics['first_rate'] >= threshold,
                later_return_rate=metrics['later_rate'] is not None and metrics['later_rate'] >= threshold)
            phase_events = [event for row in selected
                for event in opportunities[row['assignment']['index']]]
            onset_present = bool(expected) and expected[0]['phase_index'] == 0
            latency = recovery(phase_events, threshold) if onset_present else dict(
                confirmed=None, reason='Phase prefix deliberately absent; onset recovery latency is unavailable.')
            phases[phase] = dict(scheduled_games=len(expected), all_started=rates(selected),
                scheduled_phase_index_start=expected[0]['phase_index'] if expected else None,
                costs=costs(selected), fixed_tail=dict(games_requested=tail_games,
                    scheduled_indices=sorted(tail_indices), metrics=metrics, conditions=conditions,
                    descriptive_window_conditions_met=all(conditions.values())),
                recovery=latency,
                selected_arms=dict(Counter(str(row['arm']) for row in selected)),
                selection_modes=dict(Counter(row['selection_mode'] for row in selected)))
        summary_path = directory / 'summary.json'
        saved_summary = read(summary_path) if summary_path.is_file() else None
        clock = read(directory / 'work-clock.json')
        if clock['spec_sha256'] != digest(root / 'spec.json'):
            audit['problems'].append('work clock refers to a different spec')
            audit['passed'] = False
        trajectory = [dict(end_game=row['assignment']['index'] + 1,
            phase=row['assignment']['phase'], **rates(rows[max(0, i - 19):i + 1]))
            for i, row in enumerate(rows) if i >= 19]
        streams[kind] = dict(complete=len(rows) == len(assignments) and all(row['complete'] for row in rows),
            audit=audit, phases=phases, all_started=rates(rows), costs=costs(rows),
            final_wall_seconds=saved_summary['wall_seconds'] if saved_summary else None,
            between_game_wall_seconds=(saved_summary['wall_seconds'] - sum(row['wall_seconds']
                for row in rows)) if saved_summary else None,
            declared_wall_seconds=clock['seconds'], rolling20_game_trajectory=trajectory,
            descriptive_all_phase_tail_conditions_met=bool(phases) and all(
                phase['fixed_tail']['descriptive_window_conditions_met'] for phase in phases.values()))
    return dict(schema='independent-carried-learning-report-v1', directory=str(root),
        reporter_source_sha256=digest(__file__),
        spec_sha256=digest(root / 'spec.json'), source_audit=source, streams=streams,
        matched_control_comparisons=paired_controls(receipts, schedules),
        criteria=dict(fixed_tail_games=tail_games, min_later_opportunities=min_later,
            first_and_later_threshold=threshold, includes_all_exploration=True),
        qualification_declared=False,
        interpretation='Reporting only; controls, independent initialization count and parent review remain necessary. No checkpoint/model is selected.')


def plot_trajectory(report, path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figure, axes = plt.subplots(len(report['streams']), 1, figsize=(10, 3.5 * len(report['streams'])),
        squeeze=False, sharey=True)
    for axis, (kind, stream) in zip(axes[:, 0], report['streams'].items()):
        tape = stream['rolling20_game_trajectory']
        for metric, label in (('all_rate', 'All returns'), ('first_rate', 'First'), ('later_rate', 'Later')):
            axis.plot([row['end_game'] for row in tape], [row[metric] for row in tape], label=label)
        boundary = 0
        for phase, data in stream['phases'].items():
            if boundary:
                axis.axvline(boundary + .5, color='black', alpha=.4, linestyle='--')
            axis.text(boundary + 1, 1.035, phase, fontsize=8)
            boundary += data['scheduled_games']
        axis.axhline(report['criteria']['first_and_later_threshold'], color='grey', linestyle=':')
        axis.set(title=kind, ylim=(-.02, 1.12), xlim=(.5, boundary + .5),
            ylabel='Last20 games, pooled return rate')
        axis.legend(loc='lower right', ncol=3)
    axes[-1, 0].set_xlabel('Chronological game (exploration included)')
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('directory')
    parser.add_argument('--output')
    parser.add_argument('--plot')
    args = parser.parse_args()
    report = report_experiment(args.directory)
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    else:
        print(json.dumps(report, indent=2, allow_nan=False))
    if args.plot:
        plot_trajectory(report, args.plot)
