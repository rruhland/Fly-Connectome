"""Causal paired continual-learning audit of the production visual dynamics."""

import copy
import json
import math
import time
from collections import deque
from pathlib import Path

import torch

from fly_connectome.vision import load_default
from fly_connectome.vision.dynamics import mixture_log_prob, mixture_pit


class CausalDynamicsStream:
    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.frame = -1
        self.histories, self.pending = {}, {}

    def step(self, observed, *, available=True, learn=False):
        self.frame += 1
        for identity, horizon, history, origin, credit in self.pending.pop(self.frame, []):
            if learn and available and identity in observed:
                self.dynamics[horizon].observe(history, observed[identity]-origin, credit=credit)
        forecasts = []
        for identity in sorted(set(self.histories) | set(observed)):
            history = self.histories.setdefault(identity, deque(maxlen=5))
            history.append(observed.get(identity))
            if len(history) < 5 or any(p is None for p in history):
                continue
            positions = torch.stack(list(history))
            differences = positions[1:]-positions[:-1]
            for horizon, model in self.dynamics.items():
                _, credit = model.predict_with_credit(differences)
                centers, weights = credit
                forecasts.append(dict(id=identity, horizon=horizon,
                                      centers=centers+positions[-1], weights=weights))
                if learn:
                    self.pending.setdefault(self.frame+horizon, []).append(
                        (identity, horizon, differences.clone(), positions[-1].clone(), credit))
        return forecasts


SHAPES = dict(dot=((0, 0),), square=tuple((y, x) for y in (-1, 0, 1) for x in (-1, 0, 1)),
              plus=((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)),
              ell=((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
              tee=((-1, -1), (-1, 0), (-1, 1), (0, 0), (1, 0)),
              zigzag=((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)))


def render_scene(domain, seed, index, *, probe=False):
    rng = torch.Generator().manual_seed(seed+index*101)
    kind = index % 4
    shapes = ('ell', 'tee') if domain == 'shared' else ('dot', 'square')
    shape = SHAPES['zigzag' if domain == 'shared' else 'plus'] if probe else SHAPES[shapes[index % 2]]
    angle = float(torch.rand((), generator=rng))*2*math.pi
    scale = .8+.35*float(torch.rand((), generator=rng))
    phase = int(torch.randint(5, (), generator=rng))
    background = bool(index % 2)
    origin = torch.tensor([32., 32.])+torch.randint(-3, 4, (2,), generator=rng)
    direction = torch.tensor([math.sin(angle), math.cos(angle)])
    cycle = (3, 1, 2, 1, 2) if domain == 'changed' else (1, 2, 1, 3, 2)
    turn = .28 if domain == 'changed' else .12
    previous = torch.full((64, 64), float(background))
    frames, events, truth = [], [], []
    traveled = 0.
    for t in range(20):
        if kind < 2:
            center = origin-23*direction+traveled*direction
            traveled += scale*(2 if kind == 0 else cycle[(t+phase) % len(cycle)])
        else:
            theta = angle+(t+phase)*turn*(1 if kind == 2 else -1)
            center = origin+10*scale*torch.tensor([math.sin(theta), math.cos(theta)])
        center = center.round()
        image = torch.full((64, 64), float(background))
        for dy, dx in shape:
            y, x = int(center[0])+dy, int(center[1])+dx
            if 0 <= y < 64 and 0 <= x < 64:
                image[y, x] = float(not background)
        event = torch.stack(((previous-image).clamp(min=0), (image-previous).clamp(min=0)))
        previous = image
        event *= (torch.rand(event.shape, generator=rng) >= .15)
        event = torch.maximum(event, (torch.rand(event.shape, generator=rng) < .0005).float())
        frames.append((.25+.35*image+.02*torch.randn(image.shape, generator=rng)).clamp(0, 1))
        events.append(event)
        truth.append(center+torch.tensor(shape, dtype=torch.float32).mean(0))
    return frames, events, truth


@torch.no_grad()
def observations(domain, seed, count, *, probe=False):
    # Forecasts never feed back into evidence association. Cache that common work.
    model = load_default()
    model.dynamics, model.calibration = {}, {}
    scenes = []
    for index in range(count):
        frames, events, truth = render_scene(domain, seed, index, probe=probe)
        model.reset_state()
        observed = []
        for frame, event in zip(frames, events):
            state = model.step(event, frame)
            observed.append({e['id']: e['position'] for e in state['entities'] if e['observed']})
        scenes.append(dict(observed=observed, truth=truth))
    return scenes


@torch.no_grad()
def evaluate(models, scenes):
    scores = {h: [] for h in models}
    samples = missing = 0
    bounds = {h: m.calibration.bounds() for h, m in models.items()}
    for scene in scenes:
        stream = CausalDynamicsStream(models)
        for t, observed in enumerate(scene['observed']):
            forecasts = stream.step(observed)
            if t < 6:
                continue
            identity = min(observed) if observed else None
            for h in models:
                if t+h >= len(scene['truth']):
                    continue
                samples += 1
                matched = [f for f in forecasts if f['id'] == identity and f['horizon'] == h]
                if not matched:
                    missing += 1
                    continue
                f = matched[0]
                target = scene['truth'][t+h]  # evaluation only; never supplied to a learner
                pit = mixture_pit(f['centers'], f['weights'], target)
                low, high = bounds[h]
                scores[h].append((float(-mixture_log_prob(f['centers'], f['weights'], target)),
                                   float(((pit >= low) & (pit <= high)).float().mean())))
    means = {h: dict(nll=sum(v[0] for v in rows)/len(rows),
                     coverage_90=sum(v[1] for v in rows)/len(rows), samples=len(rows))
             for h, rows in scores.items()}
    return dict(horizons=means, nll=(means[4]['nll']+means[8]['nll'])/2,
                forecast_coverage=1-missing/samples)


@torch.no_grad()
def segment(models, scenes, probe, *, learn):
    start = time.perf_counter()
    curve = [dict(samples=0, **evaluate(models, probe))]
    for index, scene in enumerate(scenes):
        stream = CausalDynamicsStream(models)
        for observed in scene['observed']:
            stream.step(observed, learn=learn)
        if (index+1) % 25 == 0:
            curve.append(dict(samples=(index+1)*20, **evaluate(models, probe)))
    reached = next((r['samples'] for r in curve if r['nll'] <= 3.5 and r['forecast_coverage'] >= .95), None)
    return dict(curve=curve, samples_to_target=reached, seconds=time.perf_counter()-start)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    started = time.perf_counter()
    report = dict(protocol='2026-09-27-continual-vision-protocol.md', seeds=[], complete=False)
    out = Path('docs/experiments/2026-09-27-continual-vision-results.json')
    for seed in range(5):
        baseline = load_default().dynamics
        for h, model in baseline.items():
            model.generator.manual_seed(301000+seed*10+h)
        sets = {domain: observations(domain, 310000+seed*10000+i*1000, 100)
                for i, domain in enumerate(('A', 'shared', 'changed'))}
        probes = {domain: observations(domain, 410000+seed*10000+i*1000, 24, probe=True)
                  for i, domain in enumerate(('A', 'shared', 'changed'))}
        continuous = copy.deepcopy(baseline)
        first = segment(continuous, sets['A'], probes['A'], learn=True)
        # A fresh learner on return receives the same A stream and bundled prior;
        # its curve is exactly first_A, reused rather than recomputed.
        row = dict(seed=seed, first_A=first, fresh_return_A=first,
                   frozen_A=first['curve'][0], branches={})
        for domain in ('shared', 'changed'):
            retained = copy.deepcopy(continuous)
            before = evaluate(retained, probes['A'])
            online = segment(retained, sets[domain], probes[domain], learn=True)
            after = evaluate(retained, probes['A'])
            fresh = segment(copy.deepcopy(baseline), sets[domain], probes[domain], learn=True)
            frozen = evaluate(baseline, probes[domain])
            delta = after['nll']-before['nll']
            transfer = None if fresh['samples_to_target'] in (None, 0) else (
                online['samples_to_target'] is not None and online['samples_to_target'] <= .8*fresh['samples_to_target'])
            row['branches'][domain] = dict(online=online, fresh=fresh, frozen=frozen,
                A_before=before, A_after=after, retention_delta=delta,
                retention_gate=delta <= .2 and after['forecast_coverage'] >= before['forecast_coverage']-.05,
                forward_transfer_gate=transfer,
                return_A=segment(retained, sets['A'], probes['A'], learn=True))
        report['seeds'].append(row)
        report['seconds'] = time.perf_counter()-started
        out.write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(dict(seed=seed, branches={k: dict(retention_delta=v['retention_delta'],
            transfer=v['forward_transfer_gate'], online_target=v['online']['samples_to_target'],
            fresh_target=v['fresh']['samples_to_target']) for k, v in row['branches'].items()})), flush=True)
    report['complete'] = True
    out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
