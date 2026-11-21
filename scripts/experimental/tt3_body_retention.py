"""Associate score predictions and stroke adjustments with observed actuator responses."""

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import torch

from .tt3_arena import TT3Case
from .tt3_learning import ReturnMemory, relation_goal
from .tt3_learning_stroke import StrokeMemory, stroke_learner_factory
from .tt3_learning_trial import play_course, summarize
from .tt3_pong import state_sha256
from .tt3_pong_retention import evaluate_continued_pong
from .tt3_pong_source import write_json
from .tt3_visual import RoleModel


def response_rank(response):
    """Distinguish a single measured motion direction from a broader span."""
    singular = np.linalg.svd(response, compute_uv=False)
    if not len(singular) or singular[0] < 1e-6:
        return None
    return min(2, int(np.count_nonzero(singular >= .22 * singular[0])))


class ResponseConditionedMemory:
    """Share new outcome evidence while preserving observations for the current body."""

    def __init__(self, source, observed_rank):
        self.shared = ReturnMemory.restore(source)
        self.by_rank = {observed_rank: ReturnMemory.restore(source)}
        self.stroke = None
        self.rank = None
        self.profile_stroke = {}
        self.profile_observations = {}

    def prepare_stroke(self, offsets):
        self.stroke = StrokeMemory(self.shared, offsets)
        if self.rank is not None:
            self.profile_stroke[self.rank] = self.stroke

    def observe_response(self, matrix):
        rank = response_rank(matrix)
        if rank is None:
            return
        self.profile_observations[rank] = 1 + self.profile_observations.get(rank, 0)
        if self.rank is not None and rank <= self.rank:
            return
        self.rank = rank
        if rank not in self.by_rank:
            self.by_rank[rank] = ReturnMemory.restore(self.shared.state())
            if self.stroke is not None:
                self.profile_stroke[rank] = self.stroke

    def choose(self, rng, explore=0.):
        return self.by_rank.get(self.rank, self.shared).choose(rng, explore)

    def choose_for_response(self, response, rng):
        return self.by_rank.get(response_rank(response), self.shared).choose(rng)

    def observe(self, option, reward):
        self._observe(self.rank, option, reward)

    def observe_for_response(self, response, option, reward):
        self._observe(response_rank(response), option, reward)

    def _observe(self, rank, option, reward):
        if rank is not None and rank not in self.by_rank:
            self.by_rank[rank] = ReturnMemory.restore(self.shared.state())
        self.shared.observe(option, reward)
        if rank is not None:
            self.by_rank[rank].observe(option, reward)

    def relation_name(self, option):
        return self.shared.relation_name(option)

    def goal(self, *arguments, **options):
        return self._goal(self.rank, arguments, options)

    def goal_for_response(self, response, *arguments, **options):
        return self._goal(response_rank(response), arguments, options)

    def _goal(self, rank, arguments, options):
        stroke = self.profile_stroke.get(rank)
        return relation_goal(*arguments, **options) if stroke is None else stroke.goal(*arguments, **options)

    def state(self):
        state = dict(return_memory=self.shared.state(), active_response_rank=self.rank,
                     by_response_rank={str(rank): values.state() for rank, values in self.by_rank.items()},
                     strokes_by_response_rank={str(rank): stroke.offsets.tolist()
                                               for rank, stroke in self.profile_stroke.items()},
                     response_observations={str(rank): count for rank, count in self.profile_observations.items()})
        if self.stroke is not None and self.stroke not in self.profile_stroke.values():
            state['pending_stroke'] = self.stroke.offsets.tolist()
        return state

    @classmethod
    def restore(cls, state):
        contexts = state['by_response_rank']
        model = cls(state['return_memory'], min(map(int, contexts)))
        model.by_rank = {int(rank): ReturnMemory.restore(values) for rank, values in contexts.items()}
        model.rank = state['active_response_rank']
        model.profile_stroke = {int(rank): StrokeMemory(model.shared, offsets)
                                for rank, offsets in state['strokes_by_response_rank'].items()}
        model.profile_observations = {int(rank): count for rank, count in state['response_observations'].items()}
        model.stroke = model.profile_stroke.get(model.rank)
        if 'pending_stroke' in state:
            model.stroke = StrokeMemory(model.shared, state['pending_stroke'])
        return model


def observing_factory(base):
    class ObservingLearner(base):
        def _update_motor(self, controlled):
            super()._update_motor(controlled)
            if self.frame >= self.probe_frames - 1 and self.motor.samples >= 12:
                self.memory.observe_response(self.motor.matrix(np.asarray(controlled.position)))

    return ObservingLearner


def run_conditioned(source, schedule, before, receipt, *, forecast, integrated_motor, progress=None):
    if not all(row['visual_effects'] for row in before['points']):
        raise ValueError('the source reference must have confirmed native single-axis action effects')
    # The source probe model identified two opposite effects on one fitted visual axis.
    memory = ResponseConditionedMemory(source['return_memory'], observed_rank=1)
    source_roles = RoleModel(**source['role_model'])
    factory = observing_factory(stroke_learner_factory(forecast=forecast, integrated_motor=integrated_motor))
    practice = []
    roles = source_roles
    for generation, record in enumerate(receipt['training']):
        selected = record['runs'][record['selected']]
        cases = [TT3Case(**point['case']) for point in selected['course']['points']]
        memory.prepare_stroke(selected['stroke'])
        course = play_course(cases, roles, memory, seed=receipt['seed'] + generation, learner_factory=factory)
        roles = RoleModel(**course['final_role_model'])
        practice.append(course)
    expected = receipt['qualification']['results']['practiced']['course']
    cases = [TT3Case(**point['case']) for point in expected['points']]
    course = play_course(cases, roles, memory, seed=receipt['qualification_seed'], learner_factory=factory)
    if progress is not None:
        progress(dict(practice=summarize(practice), receiving=summarize([course]), memory=memory.state()))
    result = evaluate_continued_pong(source, schedule, before, course, memory)
    result.update(stroke_receipt_sha256=state_sha256(receipt), practice=practice,
                  receiving_forecast=forecast, integrated_motor=integrated_motor)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source_dir', type=Path)
    parser.add_argument('stroke_receipt', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--forecast', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--integrated-motor', action=argparse.BooleanOptionalAction, default=None)
    args = parser.parse_args()
    source, schedule, before = (json.loads((args.source_dir / f'pong-{name}.json').read_text())
                                for name in ('source', 'schedule', 'before'))
    opener = gzip.open if args.stroke_receipt.suffix == '.gz' else open
    with opener(args.stroke_receipt, 'rt') as stream:
        receipt = json.load(stream)
    forecast = receipt['forecast'] if args.forecast is None else args.forecast
    integrated = receipt['integrated_motor'] if args.integrated_motor is None else args.integrated_motor
    torch.set_num_threads(1)
    result = run_conditioned(source, schedule, before, receipt, forecast=forecast, integrated_motor=integrated,
                             progress=lambda report: print(json.dumps(report), flush=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, result)
    print(json.dumps(dict(retained=result['retained']['established'], after=result['retained']['after'])), flush=True)


if __name__ == '__main__':
    main()
