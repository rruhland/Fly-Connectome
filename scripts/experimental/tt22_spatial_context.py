"""Condition shared numeric preferences on observed movement rather than game labels."""

import argparse
from pathlib import Path

import numpy as np
import torch

from .tt3_numeric import NumericMemory
from .tt3_pong import PongCase, _PongReturnAgent, _native_rallies, state_sha256, summarize_pong
from .tt3_pong_probe import ShorterDwell
from .tt3_visual import RoleModel
from .tt4_transfer_trial import _hash, _read, _write
from .tt19_flight import FlightLaw, optical_start
from .tt22_spatial_trial import OUTPUT, SOURCE_REFINEMENT


FULL_SUPPORT = 1.10
ZERO_SUPPORT = 2.00
RANK_FLOOR = .002


def signature(history):
    """Estimate the secondary movement energy relative to the dominant direction."""
    if len(history) < 7:
        return None
    moves = np.array([np.asarray(b)-a for (t, a), (u, b) in zip(history, history[1:]) if u-t == 1])
    if len(moves) < 6:
        return None
    speed = np.linalg.norm(moves, axis=1)
    lower, upper = np.quantile(speed, (.75, .95))
    moves = moves[(speed > .02*lower) & (speed < 4.*upper)]
    if len(moves) < 6:
        return None
    energy = np.linalg.svd(moves, compute_uv=False)
    if len(energy) < 2 or energy[0] <= 1.e-8:
        return None
    return dict(value=float(np.log(max(RANK_FLOOR, energy[1]/energy[0]))), displacements=len(moves))


class ContextMemory(NumericMemory):
    """One fixed numeric prior and a shared list of continuously weighted outcomes."""

    def __init__(self, prior, evidence=()):
        super().__init__(prior['coefficients'], prior['prior_total'], np.asarray(prior['prior_count'], dtype=int))
        self.prior = prior
        self.evidence = list(evidence)

    def observe_context(self, index, reward, returns, context):
        self.evidence.append(dict(index=index, reward=reward, returns=returns, context=context))

    def weights(self, context):
        if context is None:
            return np.zeros(len(self.evidence))
        distances = [abs(row['context']['value']-context['value']) if row['context'] is not None else np.inf
                     for row in self.evidence]
        return np.clip((ZERO_SUPPORT-np.asarray(distances))/(ZERO_SUPPORT-FULL_SUPPORT), 0., 1.)

    def choose_context(self, context, *, explore=False):
        count = np.zeros(len(self.coefficients))
        total = np.zeros(len(self.coefficients))
        for row, weight in zip(self.evidence, self.weights(context), strict=True):
            index = row['index']
            count[index] += weight
            total[index] += weight*(row['reward']+.35*min(2, row['returns']))
        score = (self.scores+total)/(self.counts+count)
        if explore:
            score += .42*np.sqrt(np.log(2+count.sum())/(1+count))
        return int(np.argmax(score+1.e-8*np.asarray(self.prior['ties'])))

    def state(self):
        return dict(prior=self.prior, evidence=self.evidence,
                    support=dict(full=FULL_SUPPORT, zero=ZERO_SUPPORT, rank_floor=RANK_FLOOR))

    @classmethod
    def restore(cls, state):
        model = cls(state['prior'], state['evidence'])
        if model.state()['support'] != state['support']:
            raise ValueError('context estimator changed since the checkpoint')
        return model


def target_context(row):
    """Use only the inherited ready movement before the first search can fire."""
    gravity = FlightLaw().gravity
    ball, actor = [], []
    for read in row['optical']:
        tick = read['tick']
        ball.append((tick, read['mover']))
        if read['actor'] is not None:
            actor.append((tick, read['actor']))
        estimate = optical_start(ball, gravity)
        if estimate is not None and -.84 < estimate[0][1] < -.30 and estimate[1][1] < -.025:
            return tick, signature(actor)
    return None, signature(actor)


def audit_target(target, memory):
    train, held = [], []
    for row in target['practice']['pong']:
        tick, context = target_context(row)
        picked = memory.choose_context(context, explore=True)
        full_weights = memory.weights(context).tolist()
        train.append(dict(seed=row['seed'], context=context, first_search_tick=tick,
                          before=row['selected'], after=picked, evidence_weights=full_weights))
        if picked != row['selected']:
            raise ValueError(f"context changed the logged target choice at seed {row['seed']}")
        memory.observe_context(picked, row['reward'], row['outcome']['legal_returns'], context)
    for row in target['held']['pong']:
        tick, context = target_context(row)
        picked = memory.choose_context(context)
        same = memory.coefficients[picked].tolist() == row['coefficients']
        held.append(dict(seed=row['seed'], context=context, first_search_tick=tick, after=picked,
                         equivalent_actions=same, original_return=row['outcome']['legal_returns'],
                         original_reward=row['reward'], action_sha256=row['action_sha256']))
        if not same:
            raise ValueError(f"context changed the held target choice at seed {row['seed']}")
    return dict(practice=train, held=held, exact_choices=len(train)+len(held))


class ObservedDwell(ShorterDwell):
    def __init__(self, rng):
        super().__init__(rng)
        self.context = None

    def observe(self, binding, preceding_action):
        unknown = self.axis is None
        super().observe(binding, preceding_action)
        if unknown and self.axis is not None:
            self.context = signature([(tick, point) for tick, _, point in self.history[self.identity]])


class ContextAgent(_PongReturnAgent):
    def choose(self):
        if self.motor.axis is not None and not self._response_evaluated:
            self.option = self.memory.choose_context(self.motor.context)
            self._response_evaluated = True
        return super().choose()


def native(store, cases, points_per_match):
    """The ordinary native driver runs the restored post-arm store and camera probes."""
    memory = ContextMemory.restore(store['memory'])
    roles = RoleModel(**store['roles'])
    agents = [ContextAgent(roles, memory, case.exploration, motor_factory=ObservedDwell) for case in cases]
    point_cases = cases
    matches = [[] for _ in cases]
    before = state_sha256(memory.state())
    for index in range(points_per_match):
        rows = _native_rallies(point_cases, agents, max_ticks=store['max_ticks'], opponent_speed=store['opponent_speed'])
        for match, row, agent, case in zip(matches, rows, agents, cases, strict=True):
            row.update(match=dict(serve=case.serve, presentation=case.presentation, exploration=case.exploration),
                       point=index+1, relation=int(agent.option), visual_effects=agent.motor.effects,
                       visual_probe_frames=agent.motor.frames, relation_executed=agent.credited,
                       observed_context=agent.motor.context, arm_evidence_weights=memory.weights(agent.motor.context).tolist())
            match.append(row)
        if index+1 < points_per_match:
            point_cases = [PongCase(row['next_serve'], case.presentation, case.exploration)
                           for row, case in zip(rows, cases, strict=True)]
            for agent in agents:
                agent.begin_point()
    if before != state_sha256(memory.state()):
        raise ValueError('return changed the saved shared preference store')
    return [row for match in matches for row in match]


def run(folder):
    target_path = folder/'spatial-score.json.gz'
    target = _read(target_path)
    baseline_path = folder/'spatial-score-pong-return.json.gz'
    baseline = _read(baseline_path)
    if baseline['target_checkpoint_sha256'] != _hash(target_path):
        raise ValueError('baseline used a different arm practice tape')
    qualified = _read(SOURCE_REFINEMENT)
    choice = target['final']['pong']
    prior = {key: choice[key] for key in ('coefficients', 'prior_total', 'prior_count', 'ties')}
    store = ContextMemory(prior)
    target_check = audit_target(target, store)
    path = folder/'spatial-context-player.json.gz'
    package = dict(schema='tt22-optical-context-preference-1', memory=store.state(), roles=qualified['role_model'],
                   target_sha256=_hash(target_path), source_sha256=_hash(SOURCE_REFINEMENT),
                   max_ticks=qualified['max_ticks'], opponent_speed=qualified['opponent_speed'])
    _write(path, package)
    print('target trace', target_check['exact_choices'], 'choices identical', flush=True)
    cases = [PongCase(**case) for case in baseline['matched']]
    controls = baseline['runs']['before']['points']
    rows = native(_read(path), cases, len(controls)//len(cases))
    if any((a['case'], a['point']) != (b['case'], b['point']) for a, b in zip(controls, rows, strict=True)):
        raise ValueError('the matched native Pong launches changed')
    result = dict(schema='tt22-context-pong-replay-1', checkpoint_sha256=_hash(path),
                  baseline_sha256=_hash(baseline_path), target=target_check, native=rows,
                  summary=summarize_pong(rows),
                  action_matches=sum(a['action_sha256'] == b['action_sha256'] for a, b in zip(controls, rows, strict=True)),
                  outcome_matches=sum((a['score'], a['contacts']) == (b['score'], b['contacts'])
                                      for a, b in zip(controls, rows, strict=True)))
    _write(folder/'spatial-context-audit.json.gz', result)
    print('native', result['summary'], 'exact', result['action_matches'], result['outcome_matches'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder', type=Path, default=OUTPUT)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.folder)
