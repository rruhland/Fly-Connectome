"""Continue a learned Pong state through 3-D table practice and return to Pong."""

import argparse
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import torch

from .tt3_arena import TT3Case, VARIANTS, heldout_cases
from .tt3_learning import ReturnMemory
from .tt3_learning_trial import play_course, summarize
from .tt3_pong import PongCase, assess_retention, evaluate_pong, paired_pong, state_sha256, summarize_pong
from .tt3_pong_source import write_json
from .tt3_visual import RoleModel


def _reference_digests(source, schedule, before):
    source_sha = state_sha256(source)
    schedule_sha = state_sha256(schedule)
    if source_sha != before['source_state_sha256'] or schedule_sha != before['schedule_sha256']:
        raise ValueError('the paired Pong reference differs from the supplied source or schedule')
    return source_sha, schedule_sha


def evaluate_continued_pong(source, schedule, before, course, memory):
    """Reopen the paired Pong cases with the memory used in an actual table course."""
    source_sha, schedule_sha = _reference_digests(source, schedule, before)
    cases = [PongCase(**case) for case in schedule['evaluation']]
    physics = dict(points_per_match=schedule['evaluation_points'], max_ticks=schedule['max_ticks'],
                   opponent_speed=schedule['opponent_speed'])
    if course['final_return_memory'] != memory.state():
        raise ValueError('table course did not expose the actual learner state')
    receiving_roles = RoleModel(**course['final_role_model'])
    post_state = dict(role_model=asdict(receiving_roles), return_memory=memory.state())
    post_state_sha = state_sha256(post_state)
    post = evaluate_pong(cases, receiving_roles, memory, **physics)
    if post_state_sha != state_sha256(dict(role_model=asdict(receiving_roles), return_memory=memory.state())):
        raise ValueError('the post-table Pong evaluation changed persistent learner state')
    comparison = assess_retention(before['points'], post)
    post_outcomes = post_state['return_memory']
    updated = post_outcomes.get('return_memory', post_outcomes) != source['return_memory']
    comparison['table_updates'] = updated
    comparison['established'] &= updated
    comparison['behavior_changes'] = {
        key: sum(pre[key] != after[key] for pre, after in zip(before['points'], post, strict=True))
        for key in ('relation', 'action_sha256', 'contacts', 'score')
    }
    comparison['behavior_changes']['visual_effects'] = sum(
        tuple(pre['visual_effects'] or ()) != tuple(after['visual_effects'] or ())
        for pre, after in zip(before['points'], post, strict=True)
    )
    result = dict(source_state_sha256=source_sha, post_table_state_sha256=post_state_sha,
                  schedule_sha256=schedule_sha,
                  table_summary=summarize([course]), table_course=course, post_table_state=post_state,
                  retained=comparison, post_table_pong=post)
    if not comparison['established']:
        # These controls isolate the numerical source of any performance change
        # on the same starting cases if the main comparison is inconclusive.
        controls = {
            'post_memory_pre_roles': (RoleModel(**source['role_model']), memory),
            'pre_memory_post_roles': (receiving_roles, ReturnMemory.restore(source['return_memory'])),
        }
        if hasattr(memory, 'carried'):
            controls['post_shared_state'] = (receiving_roles, memory.carried)
        result['interference_controls'] = {}
        for name, (role_model, return_memory) in controls.items():
            rows = evaluate_pong(cases, role_model, return_memory, **physics)
            result['interference_controls'][name] = dict(summary=summarize_pong(rows), points=rows,
                                                         paired_to_before=paired_pong(before['points'], rows))
    return result


def run_retention(source, schedule, before, *, table_seed, table_points, variant, distractor,
                  explore, probe_frames):
    """Continue the canonical source checkpoint with the default receiving learner."""
    _reference_digests(source, schedule, before)
    roles = RoleModel(**source['role_model'])
    memory = ReturnMemory.restore(source['return_memory'])
    table = heldout_cases(f'{table_seed}/continuation', table_points, variant=variant, distractor=distractor)
    course = play_course(table, roles, memory, seed=table_seed, probe_frames=probe_frames, explore=explore)
    result = evaluate_continued_pong(source, schedule, before, course, memory)
    result['table_seed'] = table_seed
    return result


def run_stroke_retention(source, schedule, before, receipt, *, progress=None):
    """Reproduce a scored receiving course and carry its actual trained stroke back."""
    from .tt3_learning_stroke import StrokeMemory, stroke_learner_factory

    _reference_digests(source, schedule, before)
    expected = receipt['qualification']['results']['practiced']['course']
    cases = [TT3Case(**row['case']) for row in expected['points']]
    roles = RoleModel(**receipt['role_model'])
    memory = StrokeMemory(ReturnMemory.restore(receipt['return_memory']), receipt['stroke'])
    table_seed = receipt['qualification_seed']
    factory = stroke_learner_factory(forecast=receipt['forecast'], integrated_motor=receipt['integrated_motor'])
    course = play_course(cases, roles, memory, seed=table_seed, learner_factory=factory)
    same_points = all(state_sha256({key: actual[key] for key in reference}) == state_sha256(reference)
                      for reference, actual in zip(expected['points'], course['points'], strict=True))
    same_state = all(state_sha256(expected[key]) == state_sha256(course[key])
                     for key in ('final_return_memory', 'final_role_model', 'motor_samples'))
    if not same_points or not same_state:
        raise ValueError('the reproduced table qualification differed from the supplied receiving receipt')
    if progress is not None:
        progress(summarize([course]))
    result = evaluate_continued_pong(source, schedule, before, course, memory)
    result.update(table_seed=table_seed, exact_table_reproduction=True, stroke_receipt_sha256=state_sha256(receipt))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_dir', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--table-seed', type=int, default=23547330)
    parser.add_argument('--table-points', type=int, default=32)
    parser.add_argument('--variant', choices=VARIANTS, default='standard')
    parser.add_argument('--explore', type=float, default=.08)
    parser.add_argument('--probe-frames', type=int, default=16)
    parser.add_argument('--distractor', action='store_true')
    parser.add_argument('--stroke-receipt', type=Path)
    args = parser.parse_args(argv)
    if args.table_points < 1:
        parser.error('table practice must include at least one native point')
    torch.set_num_threads(1)
    source, schedule, before = (json.loads((args.source_dir / f'pong-{name}.json').read_text())
                                for name in ('source', 'schedule', 'before'))
    if args.stroke_receipt:
        open_receipt = gzip.open if args.stroke_receipt.suffix == '.gz' else open
        with open_receipt(args.stroke_receipt, 'rt') as handle:
            receipt = json.load(handle)
        result = run_stroke_retention(source, schedule, before, receipt,
                                      progress=lambda summary: print(json.dumps({'receiving': summary}), flush=True))
    else:
        result = run_retention(source, schedule, before, table_seed=args.table_seed, table_points=args.table_points,
                               variant=args.variant, distractor=args.distractor, explore=args.explore,
                               probe_frames=args.probe_frames)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(dict(table=result['table_summary'], pong_before=result['retained']['before'],
                          pong_after=result['retained']['after'], established=result['retained']['established'])),
          flush=True)


if __name__ == '__main__':
    main()
