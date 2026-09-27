"""Eight-step conditional retrieval experiment, never a production default."""

import copy
import json
import time
from pathlib import Path

import torch

from fly_connectome.vision import load_default
from fly_connectome.vision.dynamics import SpatialBelief, mixture_log_prob, mixture_pit
from continual_vision_audit import CausalDynamicsStream, observations
from memory_context_comparison import MemoryComparison


class ContextBelief(SpatialBelief):
    history_steps = 8

    def __init__(self, source, context_steps):
        self.short = MemoryComparison(source, 'append')
        self.long = MemoryComparison(SpatialBelief(horizon=source.horizon), 'append')
        self.long.keys = torch.empty(0, context_steps*2)
        self.context_steps = context_steps
        self.calibration = copy.deepcopy(source.calibration)

    def uses_context(self, history):
        return len(history) == 8 and len(self.long.keys) >= 32

    def distribution(self, history):
        if self.uses_context(history):
            return self.long.distribution(history[-self.context_steps:])
        return self.short.distribution(history[-4:])

    def observe(self, history, displacement, *, credit=None):
        if credit is not None:
            self.calibration.observe(mixture_pit(*credit, displacement))
        self.short.observe(history[-4:], displacement)
        if len(history) == 8:
            self.long.observe(history[-self.context_steps:], displacement)


@torch.no_grad()
def evaluate(models, scenes):
    scores = {h: [] for h in models}
    total = missing = 0
    bounds = {h: m.calibration.bounds() for h, m in models.items()}
    for scene in scenes:
        stream = CausalDynamicsStream(models)
        for t, observed in enumerate(scene['observed']):
            forecasts = stream.step(observed)
            if t < 6:
                continue
            identity = min(observed) if observed else None
            for h, model in models.items():
                if t+h >= len(scene['truth']):
                    continue
                total += 1
                matched = [f for f in forecasts if f['id'] == identity and f['horizon'] == h]
                if not matched:
                    missing += 1
                    continue
                f = matched[0]
                positions = torch.stack([p for p in stream.histories[identity] if p is not None])
                history = positions[1:]-positions[:-1]
                target = scene['truth'][t+h]
                centers, weights = model.short.distribution(history[-4:])
                shadow = -mixture_log_prob(centers+positions[-1], weights, target)
                pit = mixture_pit(f['centers'], f['weights'], target)
                low, high = bounds[h]
                scores[h].append((float(-mixture_log_prob(f['centers'], f['weights'], target)),
                    float(shadow), model.uses_context(history),
                    float(((pit >= low) & (pit <= high)).float().mean())))
    summary = {}
    for h, rows in scores.items():
        used = [r for r in rows if r[2]]
        summary[h] = dict(nll=sum(r[0] for r in rows)/len(rows),
            shadow_nll=sum(r[1] for r in rows)/len(rows), samples=len(rows),
            context_fraction=len(used)/len(rows),
            context_nll=sum(r[0] for r in used)/len(used) if used else None,
            context_shadow_nll=sum(r[1] for r in used)/len(used) if used else None,
            coverage_90=sum(r[3] for r in rows)/len(rows))
    return dict(horizons=summary, nll=(summary[4]['nll']+summary[8]['nll'])/2,
                shadow_nll=(summary[4]['shadow_nll']+summary[8]['shadow_nll'])/2,
                forecast_coverage=1-missing/total)


@torch.no_grad()
def segment(models, scenes, probe):
    curve = [dict(samples=0, **evaluate(models, probe))]
    for index, scene in enumerate(scenes):
        stream = CausalDynamicsStream(models)
        for observed in scene['observed']:
            stream.step(observed, learn=True)
        if (index+1) % 25 == 0:
            curve.append(dict(samples=(index+1)*20, **evaluate(models, probe)))
    target = next((r['samples'] for r in curve if r['nll'] <= 3.5 and r['forecast_coverage'] >= .95), None)
    return dict(curve=curve, samples_to_target=target)


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    reference = json.loads(Path('docs/experiments/2026-09-27-memory-context-comparison-results.json').read_text())
    report = dict(protocol='2026-09-27-long-context-protocol.md',
        reference='2026-09-27-memory-context-comparison-results.json', seeds=[], complete=False)
    path = Path('docs/experiments/2026-09-27-long-context-results.json')
    for seed in range(5):
        source = load_default().dynamics
        sets = {d: observations(d, 310000+seed*10000+i*1000, 100)
                for i, d in enumerate(('A', 'shared', 'changed'))}
        probes = {d: observations(d, 410000+seed*10000+i*1000, 24, probe=True)
                  for i, d in enumerate(('A', 'shared', 'changed'))}
        row = dict(seed=seed, arms={})
        for context in (4, 8):
            baseline = {h: ContextBelief(m, context) for h, m in source.items()}
            model = copy.deepcopy(baseline)
            first = segment(model, sets['A'], probes['A'])
            branches = {}
            for domain in ('shared', 'changed'):
                branch = copy.deepcopy(model)
                before = evaluate(branch, probes['A'])
                online = segment(branch, sets[domain], probes[domain])
                after = evaluate(branch, probes['A'])
                fresh = segment(copy.deepcopy(baseline), sets[domain], probes[domain])
                sizes = {h: dict(short=len(m.short.keys), specialist=len(m.long.keys)) for h, m in branch.items()}
                branches[domain] = dict(online=online, fresh=fresh, A_before=before, A_after=after,
                    retention_delta=after['nll']-before['nll'], memory_examples=sizes,
                    return_A=segment(branch, sets['A'], probes['A']))
                old = reference['seeds'][seed]['arms']['append']['branches'][domain]
                for phase in ('online', 'fresh', 'return_A'):
                    for new, previous in zip(branches[domain][phase]['curve'], old[phase]['curve']):
                        assert abs(new['shadow_nll']-previous['nll']) < 1e-6
            row['arms'][str(context)] = dict(first_A=first, branches=branches)
            print(json.dumps(dict(seed=seed, context=context, branches={d: dict(
                final=b['online']['curve'][-1]['nll'], retention=b['retention_delta'],
                target=b['online']['samples_to_target']) for d, b in branches.items()})), flush=True)
        report['seeds'].append(row)
        report['seconds'] = time.perf_counter()-start
        path.write_text(json.dumps(report, indent=2)+'\n')
    report['complete'] = True
    path.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
