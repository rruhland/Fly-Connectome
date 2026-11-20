"""Run the independently sealed confirmation of visual placement gating."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import copy
import json
from pathlib import Path
import secrets
from statistics import mean
import subprocess
from time import monotonic

from arcade_transfer_schedule import CHECKPOINT_SHA256, FIRST_OPPORTUNITIES, MAX_LIFE_TICKS, canonical_json, _priority, _seed
from run_arcade_transfer_qualification import (
    CHECKPOINT,
    ROOT,
    _candidate_receipt,
    _controller_state,
    _paired_difference,
    _play_life,
    _returns,
    _sha,
    score_course,
    summarize_courses,
)


SCHEMA = 'visual-placement-confirmation-v1'
COURSES = 8
LIVES = 6
TICKS_PER_SECOND = 120
ARMS = ('gated_carry', 'gated_reset_negative', 'gated_reset_positive', 'legacy_carry', 'passive')
RESETS = {'gated_reset_negative': -1, 'gated_reset_positive': 1}


def build_schedule(key):
    if len(key) != 32:
        raise ValueError('the independent schedule key must contain 32 bytes')
    courses = []
    for i in range(COURSES):
        course_id = f'placement-{i + 1:02d}'
        label = f'{SCHEMA}:{course_id}'
        starts = [(side, repeat) for side in (-1, 1) for repeat in range(LIVES // 2)]
        starts.sort(key=lambda item: _priority(key, f'{label}:order:{item}'))
        opening_side = -1 if i % 2 == 0 else 1
        opening = next(j for j, (side, _) in enumerate(starts) if side == opening_side)
        starts.insert(0, starts.pop(opening))
        course = {'id': course_id, 'cohort': 'ordinary', 'policy_seed': _seed(key, f'{label}:policy'), 'lives': []}
        for j, (side, _) in enumerate(starts, 1):
            course['lives'].append(dict(id=f'{course_id}-{j:02d}', serve_side=side, condition='ordinary',
                                        environment_seed=_seed(key, f'{label}:environment:{j}'),
                                        layout_seed=_seed(key, f'{label}:layout:{j}')))
        courses.append(course)
    return dict(schema=SCHEMA, key=key.hex(), courses=courses, layout_split='heldout',
                distractors=True, max_life_ticks=MAX_LIFE_TICKS, checkpoint_sha256=CHECKPOINT_SHA256)


def _new_session(arm, seed, checkpoint):
    from arcade_transfer_session import TransferSession
    from arcade_visual_placement_gate import GatedTransferSession

    if arm == 'gated_carry':
        return GatedTransferSession('full', seed, checkpoint)
    if arm in RESETS:
        return GatedTransferSession('fresh', seed, initial_polarity=RESETS[arm])
    return TransferSession('full' if arm == 'legacy_carry' else 'passive', seed, checkpoint)


def _misses_to_sustained(lives, sustained):
    misses = 0
    for life in lives:
        limit = sustained['opportunity'] if sustained and life['id'] == sustained['life'] else len(life['contacts'])
        misses += sum(not contact['returned'] for contact in life['contacts'][:limit])
        if sustained and life['id'] == sustained['life']:
            break
    return misses


def _run_course(arm, course, schedule, checkpoint, output):
    import torch
    from breakout_arena import BreakoutCamera

    torch.set_num_threads(1)
    started = monotonic()
    session = _new_session(arm, course['policy_seed'], checkpoint)
    initial_state, initial_sha = _controller_state(session)
    lives = []
    previous_sha = initial_sha
    for life in course['lives']:
        arena = BreakoutCamera(life['environment_seed'], layout_seed=life['layout_seed'],
                               serve_side=life['serve_side'], layout_split=schedule['layout_split'],
                               distractors=schedule['distractors'], max_episode_ticks=schedule['max_life_ticks'])
        controller = session.begin_life()
        row = _play_life(controller if arm != 'passive' else None, arena, 1)
        gate = getattr(session, 'gate', None)
        if gate is not None:
            row['gate'] = gate.state()
        session.end_life()
        _, final_sha = _controller_state(session)
        lives.append(dict(row, **life, initial_state_sha256=previous_sha, final_state_sha256=final_sha))
        previous_sha = final_sha

    final_state, final_sha = _controller_state(session)
    scored = score_course(lives, schedule['max_life_ticks'])
    row = dict(**course, arm=arm, initial_state=initial_state, final_state=final_state,
               initial_state_sha256=initial_sha, final_state_sha256=final_sha,
               missed_before_sustained_or_censor=_misses_to_sustained(lives, scored['sustained']),
               evaluation_runtime_seconds=monotonic() - started, **scored)
    row['lives'] = lives
    (output / f'{arm}-{course["id"]}.json').write_bytes(canonical_json(row))
    return row


def _paired_audit(rows):
    for group in zip(*(rows[arm] for arm in ARMS), strict=True):
        if len({(course['id'], course['policy_seed']) for course in group}) != 1:
            raise ValueError('all comparator arms must share a physical course and policy seed')
        carried, reset_negative, reset_positive, legacy, _ = group
        if carried['initial_state'] != legacy['initial_state']:
            raise ValueError('the complete checkpoint must be identical in both carried arms')
        negative, positive = (copy.deepcopy(row['initial_state']) for row in (reset_negative, reset_positive))
        negative['polarity'] *= -1
        negative['policy']['choices'][4] *= -1
        if negative != positive or positive['polarity'] != 1:
            raise ValueError('reset initialization must differ only in forced polarity')
        for lives in zip(*(course['lives'] for course in group), strict=True):
            keys = ('id', 'environment_seed', 'layout_seed', 'serve_side', 'initial_bricks')
            if len({tuple(life[key] for key in keys) for life in lives}) != 1:
                raise ValueError('life configuration differs between paired arms')
            if any(not life['contacts'] for life in lives):
                raise ValueError('a scheduled serve did not reach its first incoming opportunity')
            first_contacts = [life['contacts'][0] for life in lives]
            if len({(c['tick'], c['ball_x']) for c in first_contacts}) != 1:
                raise ValueError('paired trajectories differed before their first contact')
            if first_contacts[-1]['returned'] or first_contacts[-1]['offset'] * lives[-1]['serve_side'] <= 0:
                raise ValueError('the first passive opportunity must require the scheduled movement')


def course_metrics(row):
    return dict(first_five_returns=_returns(row['first_five']),
                first_return_achieved=int(row['first_return'] is not None),
                sustained_achieved=int(row['sustained'] is not None),
                capped_sustained_seconds=row['capped_sustained_ticks'] / TICKS_PER_SECOND,
                observed_play_seconds=row['observed_ticks'] / TICKS_PER_SECOND,
                missed_before_sustained_or_censor=row['missed_before_sustained_or_censor'])


def _paired_estimates(candidate, reference):
    result = {}
    signs = dict(first_five_returns=1 / FIRST_OPPORTUNITIES, capped_sustained_seconds=-1,
                 missed_before_sustained_or_censor=-1, sustained_achieved=1)
    for metric, sign in signs.items():
        differences = [sign * (a[metric] - b[metric]) for a, b in zip(candidate, reference, strict=True)]
        result[metric] = dict(_paired_difference(differences), by_course=differences)
    return result


def summarize(rows):
    per_course = {arm: [course_metrics(course) for course in rows[arm]] for arm in ARMS}
    negative, positive = (per_course[arm] for arm in RESETS)
    per_course['gated_reset_equal_weight'] = [
        {metric: mean((a[metric], b[metric])) for metric in a}
        for a, b in zip(negative, positive, strict=True)
    ]
    by_arm = {}
    for arm, courses in rows.items():
        achieved = [course['sustained']['total_ticks'] / TICKS_PER_SECOND for course in courses if course['sustained']]
        gates = [life['gate'] for course in courses for life in course['lives'] if 'gate' in life]
        carry_dormant = gates and not any(gate['opponent_frames'] for gate in gates)
        memory_equal = all(all(course['initial_state'][key] == course['final_state'][key]
                               for key in ('score', 'shot_memory')) for course in courses) if carry_dormant else None
        by_arm[arm] = dict(summarize_courses(courses), sustained_achieved=len(achieved),
                           sustained_censored=len(courses) - len(achieved),
                           sustained_observed_mean_seconds=mean(achieved) if achieved else None,
                           gate_opponent_frames=sum(gate['opponent_frames'] for gate in gates),
                           gate_nonzero_offsets=sum(gate['nonzero_offsets'] for gate in gates),
                           dormant_offensive_state_equal=memory_equal)
    comparisons = {
        f'gated_carry_vs_{arm}': _paired_estimates(per_course['gated_carry'], per_course[arm])
        for arm in ('gated_reset_equal_weight', 'legacy_carry', 'passive')
    }
    return dict(course_ids=[course['id'] for course in rows['gated_carry']], by_arm=by_arm,
                course_metrics=per_course, paired=comparisons)


def seal(schedule_path, seal_path, candidate):
    frozen = _candidate_receipt(candidate)['candidate_commit']
    content = canonical_json(build_schedule(secrets.token_bytes(32)))
    with schedule_path.open('xb') as stream:
        stream.write(content)
    with seal_path.open('xb') as stream:
        stream.write(canonical_json(dict(schema=SCHEMA, candidate_commit=frozen, schedule_sha256=_sha(content),
                                         checkpoint_sha256=CHECKPOINT_SHA256)))


def run(schedule_path, seal_path, output, workers):
    content = schedule_path.read_bytes()
    stored_seal = seal_path.read_bytes()
    path = seal_path.resolve().relative_to(ROOT)
    committed_seal = subprocess.check_output(('git', 'show', f'HEAD:{path}'), cwd=ROOT)
    if committed_seal != stored_seal:
        raise ValueError('schedule digest must be committed before the native run')
    registration = json.loads(stored_seal)
    schedule = json.loads(content)
    if registration['schema'] != SCHEMA or registration['schedule_sha256'] != _sha(content):
        raise ValueError('schedule digest differs from the independent registration')
    if canonical_json(build_schedule(bytes.fromhex(schedule['key']))) != content:
        raise ValueError('private schedule differs from the registered generator')
    checkpoint_content = CHECKPOINT.read_bytes()
    if _sha(checkpoint_content) != registration['checkpoint_sha256'] or _sha(checkpoint_content) != CHECKPOINT_SHA256:
        raise ValueError('Pong checkpoint differs from the registered digest')
    checkpoint = json.loads(checkpoint_content)
    receipt = dict(_candidate_receipt(registration['candidate_commit']), **registration, arms=ARMS)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'receipt.json').write_bytes(canonical_json(receipt))
    (output / 'schedule.json').write_bytes(content)
    started = monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = {(arm, course['id']): pool.submit(_run_course, arm, course, schedule, checkpoint, output)
                for course in schedule['courses'] for arm in ARMS}
        rows = {arm: [jobs[arm, course['id']].result() for course in schedule['courses']] for arm in ARMS}
    _paired_audit(rows)
    summary = dict(summarize(rows), evaluation_runtime_seconds=monotonic() - started)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
    print(json.dumps(summary['paired'], indent=2, allow_nan=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    registration = commands.add_parser('seal')
    registration.add_argument('private_schedule', type=Path)
    registration.add_argument('registration', type=Path)
    registration.add_argument('--candidate', required=True)
    execute = commands.add_parser('run')
    execute.add_argument('private_schedule', type=Path)
    execute.add_argument('registration', type=Path)
    execute.add_argument('output', type=Path)
    execute.add_argument('--workers', type=int, choices=(1, 2), default=2)
    args = parser.parse_args()
    if args.command == 'seal':
        seal(args.private_schedule, args.registration, args.candidate)
    else:
        run(args.private_schedule, args.registration, args.output, args.workers)


if __name__ == '__main__':
    main()
