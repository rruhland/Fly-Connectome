"""Detect changing motor effects during wrist-only paddle-bounce matches."""
import argparse
from collections import deque
import gzip
import json
from pathlib import Path

import numpy as np

from .tt24_bounce import BounceArena
from .tt24_effects import MotorEffects, excitation, transmission
from .tt24_motor import MotorEffects as Gains, MotorPolicy

DIR = Path('artifacts/tt24')


class History:
    def __init__(self, obs):
        self.commands = deque([np.zeros(5) for _ in range(4)], maxlen=4)
        self.deltas = deque([np.zeros(5) for _ in range(2)], maxlen=2)
        self.q = obs['joints'].copy()
        self.ticks = 0

    def transition(self, action, obs):
        self.commands.appendleft(action.copy())
        features = np.r_[np.concatenate(self.commands), np.concatenate(self.deltas), 1.]
        delta = (obs['joints'] - self.q) / .08
        self.deltas.appendleft(delta)
        self.q = obs['joints'].copy()
        self.ticks += 1
        return features, delta


class Survey:
    def __init__(self, rng):
        self.plan = excitation(rng)
        self.effects = MotorEffects(5, blocks=len(self.plan))
        self.states, self.responses = [], []
        self.restart()

    def restart(self):
        self.tick = 0
        self.previous = np.zeros(5)

    def act(self, obs):
        if not self.tick:
            self.origin = obs['joints'].copy()
        return self.plan[0].copy() if self.tick < 6 else np.zeros(5)

    def receive(self, transition, obs, history):
        if history.ticks >= 4:
            self.states.append(transition[0])
            self.responses.append(transition[1])
        self.tick += 1
        if self.tick == 12:
            delta = obs['joints'] - self.origin
            self.effects.update(self.plan.pop(0), delta, self.previous)
            self.previous = delta
            self.tick = 0
        return not self.plan

    def result(self):
        x, y = np.array(self.states), np.array(self.responses)
        weights = np.linalg.solve(x.T@x + np.eye(x.shape[1])*.025, x.T@y)
        return self.effects.coefficients()[:5].T, weights


def controls(record):
    gains = Gains(5)
    gains.readout[:] = record['motor_effects']
    return MotorPolicy(np.array(record['weights']), gains, readout='rate')


def residual(weights, pairs):
    x, y = map(np.array, zip(*pairs))
    return float(np.sqrt(np.mean((x@weights-y)**2)))


def fit_native(env, record, seed):
    rng = np.random.default_rng(seed)
    survey = Survey(rng)
    cost = dict(probe_drops=0, probe_ticks=0, reference_drops=5, reference_ticks=0)
    complete = False
    while not complete:
        obs = env.reset(seed=int(rng.integers(2**31-1)), split='practice', render=False)
        survey.restart()
        history = History(obs)
        cost['probe_drops'] += 1
        while True:
            action = survey.act(obs)
            obs, _, done = env.step(action, render=False)
            complete = survey.receive(history.transition(action, obs), obs, history)
            cost['probe_ticks'] += 1
            if complete or done:
                break
    effect, predictor = survey.result()
    errors = []
    for _ in range(cost['reference_drops']):
        obs = env.reset(seed=int(rng.integers(2**31-1)), split='practice')
        pupil, history = controls(record), History(obs)
        recent = deque(maxlen=8)
        while True:
            t = history.ticks
            action = pupil.command(obs, sampled=t%2==0)
            obs, _, done = env.step(action, render=(t+1)%2==0)
            pair = history.transition(action, obs)
            if history.ticks >= 4:
                recent.append(pair)
            if len(recent) == 8:
                errors.append(residual(predictor, recent))
            cost['reference_ticks'] += 1
            if done:
                break
    threshold = max(.055, 2.2*np.percentile(errors, 95))
    return effect, predictor, threshold, cost


def stream(arenas, record, native, starts, wiring, seed, mode):
    adaptive = mode == 'adaptive'
    source, original, threshold, _ = native
    rng = np.random.default_rng(seed)
    bank = [(source, original)]
    active, inverse, survey = 0, np.eye(5), None
    data = []
    for index, seed in enumerate(starts):
        # The hardware and rule switch live exclusively in the evaluator.
        changed = 7 <= index < 22
        env = arenas[int(changed)]
        obs = env.reset(seed=seed, split='held', regime='changed' if changed else 'standard')
        if survey:
            survey.restart()
        pupil, history = controls(record), History(obs)
        recent = deque(maxlen=8)
        probes = triggers = recalls = 0
        first_trigger = None
        first_recall = None
        before = active
        worst = 0.
        while True:
            t = history.ticks
            desired = pupil.command(obs, sampled=t%2==0)
            action = survey.act(obs) if survey else np.clip(inverse@desired, -1., 1.)
            actual = wiring@action if changed and mode != 'familiar_wiring' else action
            obs, _, done = env.step(actual, render=(t+1)%2==0)
            pair = history.transition(action, obs)
            if survey:
                probes += 1
                if survey.receive(pair, obs, history):
                    bank.append(survey.result())
                    active = len(bank)-1
                    inverse = np.linalg.lstsq(bank[active][0], source, rcond=.03)[0]
                    survey = None
                    recent.clear()
            elif adaptive and history.ticks >= 5:
                recent.append(pair)
                if len(recent)==8 and history.ticks%4==0:
                    errors = [residual(w, recent) for _,w in bank]
                    worst = max(worst, errors[active])
                    if errors[active] > threshold:
                        best = int(np.argmin(errors))
                        if best != active and errors[best] < threshold*.75:
                            active = best
                            inverse = np.linalg.lstsq(bank[best][0], source, rcond=.03)[0]
                            recalls += 1
                            if first_recall is None:
                                first_recall = history.ticks
                        else:
                            survey = Survey(rng)
                            triggers += 1
                            if first_trigger is None:
                                first_trigger = history.ticks
                        recent.clear()
            if done:
                break
        judge = env.judge_metrics()
        data.append(dict(seed=seed, changed=changed, score=judge['rebounds'], run=judge['longest_consecutive_rebounds'],
                         ticks=history.ticks, probes=probes, triggers=triggers, recalls=recalls,
                         before=before, after=active, first_trigger=first_trigger, first_recall=first_recall, max_error=worst))
    return data


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--seed', type=int, default=590314)
    p.add_argument('--streams', type=int, default=3)
    args = p.parse_args()
    with gzip.open(DIR/'recurrence-arm.json.gz','rt') as f:
        policy = json.load(f)['policies']['pong']
    with gzip.open(DIR/'motor-rate-target-only.json.gz','rt') as f:
        gains = json.load(f)['motor_effects']
    record = dict(weights=policy,motor_effects=gains)
    arenas = [BounceArena(view='wrist',width=64,height=48,arm=m) for m in ('stock','alternate')]
    rows = []
    try:
        native = fit_native(arenas[0],record,args.seed+1)
        rng = np.random.default_rng(args.seed)
        for _ in range(args.streams):
            wire = transmission(rng)
            starts = rng.integers(1,2**31-1,30).tolist()
            seed = int(rng.integers(2**31-1))
            modes = {m:stream(arenas,record,native,starts,wire,seed,m) for m in ('adaptive','frozen','familiar_wiring')}
            print([(k,sum(x['score'] for x in v[7:22]),sum(x['triggers'] for x in v)) for k,v in modes.items()],flush=True)
            rows.append(dict(wire=wire.tolist(),modes=modes))
        args.out.write_text(json.dumps(dict(nominal=native[3],threshold=native[2],rows=rows),separators=(',',':'))+'\n')
    finally:
        for arena in arenas:
            arena.close()


if __name__=='__main__':
    main()
