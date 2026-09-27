"""Experimental memory comparison; production runtime remains unchanged."""

import copy
import json
import time
from collections import defaultdict
from pathlib import Path

import torch

from fly_connectome.vision import load_legacy_default as load_default
from fly_connectome.vision.dynamics import SpatialBelief, mixture_pit
from continual_vision_audit import observations, evaluate, segment


class MemoryComparison(SpatialBelief):
    def __init__(self, source, mode):
        self.__dict__ = copy.deepcopy(source.__dict__)
        self.mode = mode
        self.recent_keys = self.keys[:0].clone()
        self.recent_values = self.values[:0].clone()

    @torch.no_grad()
    def observe(self, history, displacement, *, credit=None):
        key, basis, scale = self.encode(history)
        value = basis @ displacement/scale
        if self.mode == 'append':
            if credit is not None:
                self.calibration.observe(mixture_pit(credit[0], credit[1], displacement))
            self.seen += 1
            self.keys = torch.cat((self.keys, key[None]))
            self.values = torch.cat((self.values, value[None]))
        else:
            super().observe(history, displacement, credit=credit)
            self.recent_keys = torch.cat((self.recent_keys, key[None]))[-512:]
            self.recent_values = torch.cat((self.recent_values, value[None]))[-512:]

    @torch.no_grad()
    def distribution(self, history):
        if self.mode == 'append' or not len(self.recent_keys):
            return super().distribution(history)
        key, basis, scale = self.encode(history)
        keys = torch.cat((self.keys, self.recent_keys))
        values = torch.cat((self.values, self.recent_values))
        distance = (keys-key).square().mean(1)
        nearest, indices = distance.topk(min(32, len(distance)), largest=False)
        weights = torch.softmax(-nearest/nearest[-1].clamp(min=.01), 0)
        return values[indices] @ basis*scale, weights


def history_groups(scenes, mode, horizon):
    groups = defaultdict(list)
    for scene in scenes:
        observed = scene['observed']
        for t in range(8, len(observed)-horizon):
            for identity, origin in observed[t].items():
                if any(identity not in observed[i] for i in range(t-8, t+horizon+1)):
                    continue
                positions = torch.stack([observed[i][identity] for i in range(t-8, t+1)])
                history = positions[1:]-positions[:-1]
                target = observed[t+horizon][identity]-origin
                if mode == 'normalized4':
                    key, basis, scale = SpatialBelief.encode(history[-4:])
                    # Normalized displacement units, not actual pixel errors.
                    target = basis @ target/scale
                else:
                    key = history[-4:] if mode == 'raw4' else history
                groups[tuple((key.flatten()*10000).round().long().tolist())].append(target)
    return groups


def ambiguity(left, right):
    shared = left.keys() & right.keys()
    gaps = [float((torch.stack(left[k]).mean(0)-torch.stack(right[k]).mean(0)).norm()) for k in shared]
    spread = []
    for groups in (left, right):
        for rows in groups.values():
            if len(rows) > 1:
                values = torch.stack(rows)
                spread.append(float((values-values.mean(0)).norm(dim=1).mean()))
    return dict(keys_A=len(left), keys_B=len(right), shared_keys=len(shared),
                shared_samples_A=sum(len(left[k]) for k in shared),
                shared_samples_B=sum(len(right[k]) for k in shared),
                samples_A=sum(map(len, left.values())), samples_B=sum(map(len, right.values())),
                mean_gap=None if not gaps else sum(gaps)/len(gaps),
                maximum_gap=None if not gaps else max(gaps),
                conflicting_keys=sum(g > 2 for g in gaps),
                within_domain_mean_spread=None if not spread else sum(spread)/len(spread))


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    start = time.perf_counter()
    report = dict(protocol='2026-09-27-memory-context-comparison-protocol.md', seeds=[], complete=False,
                  ambiguity_units=dict(raw4='pixels', raw8='pixels', normalized4='normalized displacement'))
    out = Path('docs/experiments/2026-09-27-memory-context-comparison-results.json')
    for seed in range(5):
        source = load_default().dynamics
        for h, model in source.items():
            model.generator.manual_seed(301000+seed*10+h)
        sets = {d: observations(d, 310000+seed*10000+i*1000, 100)
                for i, d in enumerate(('A', 'shared', 'changed'))}
        probes = {d: observations(d, 410000+seed*10000+i*1000, 24, probe=True)
                  for i, d in enumerate(('A', 'shared', 'changed'))}
        row = dict(seed=seed, arms={}, ambiguity={})
        for mode in ('raw4', 'normalized4', 'raw8'):
            row['ambiguity'][mode] = {h: ambiguity(history_groups(sets['A'], mode, h),
                history_groups(sets['changed'], mode, h)) for h in (4, 8)}
        for arm in ('reservoir', 'recent_stable', 'append'):
            baseline = copy.deepcopy(source) if arm == 'reservoir' else {
                h: MemoryComparison(m, arm) for h, m in source.items()}
            trained = copy.deepcopy(baseline)
            first = segment(trained, sets['A'], probes['A'], learn=True)
            branches = {}
            for domain in ('shared', 'changed'):
                model = copy.deepcopy(trained)
                before = evaluate(model, probes['A'])
                online = segment(model, sets[domain], probes[domain], learn=True)
                after = evaluate(model, probes['A'])
                fresh = segment(copy.deepcopy(baseline), sets[domain], probes[domain], learn=True)
                sizes = {h: len(m.keys)+len(getattr(m, 'recent_keys', [])) for h, m in model.items()}
                branches[domain] = dict(online=online, fresh=fresh, A_before=before, A_after=after,
                    retention_delta=after['nll']-before['nll'], memory_examples=sizes,
                    return_A=segment(model, sets['A'], probes['A'], learn=True))
            row['arms'][arm] = dict(first_A=first, branches=branches)
            print(json.dumps(dict(seed=seed, arm=arm, branches={d: dict(
                final=v['online']['curve'][-1]['nll'], retention=v['retention_delta'],
                target=v['online']['samples_to_target']) for d, v in branches.items()})), flush=True)
        report['seeds'].append(row)
        report['seconds'] = time.perf_counter()-start
        out.write_text(json.dumps(report, indent=2)+'\n')
    report['complete'] = True
    out.write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
