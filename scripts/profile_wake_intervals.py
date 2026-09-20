"""Measure completed exact event gaps; censor observation-window edges explicitly."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

import numpy as np
import torch
from fly_connectome.data import checksum
from fly_connectome.native_cpu import NativeCPU
from fly_connectome.training import load_checkpoint


class Gaps:
    def __init__(self, neurons, ticks):
        self.last = np.full(neurons, -1, dtype=np.int64)
        self.first = self.last.copy()
        self.bins = np.zeros((neurons, 4), dtype=np.int64)
        self.sums = np.zeros(neurons, dtype=np.int64)
        self.histogram = np.zeros(ticks + 1, dtype=np.int64)
        self.event_counts = np.zeros(neurons, dtype=np.int64)

    def observe(self, tick, events):
        indices = np.flatnonzero(events)
        completed = indices[self.last[indices] >= 0]
        gaps = tick - self.last[completed]
        bins = np.searchsorted([3, 9, 99], gaps, side='left')
        np.add.at(self.bins, (completed, bins), 1)
        np.add.at(self.histogram, gaps, 1)
        self.sums[completed] += gaps
        new = indices[self.last[indices] < 0]
        self.first[new] = tick
        self.last[indices] = tick
        self.event_counts[indices] += 1

    def summary(self, ticks, indices=None):
        ids = np.arange(len(self.last)) if indices is None else np.asarray(indices)
        bins = self.bins[ids]
        counts = bins.sum(axis=1)
        eligible = counts > 0
        ever = self.last[ids] >= 0
        total = int(counts.sum())
        per_neuron = bins[eligible] / counts[eligible, None]
        means = self.sums[ids][eligible] / counts[eligible]
        result = dict(neurons=len(ids), completed_gaps=total,
            neurons_with_completed_gaps=int(eligible.sum()),
            never_waking_neurons=int((~ever).sum()),
            one_event_only_neurons=int((self.event_counts[ids] == 1).sum()),
            event_neuron_ticks=int(self.event_counts[ids].sum()),
            event_free_neuron_ticks=int(ticks * len(ids) - self.event_counts[ids].sum()),
            completed_gap_counts=bins.sum(axis=0).tolist(),
            event_weighted_bin_fractions=(bins.sum(axis=0) / total).tolist() if total else None,
            neuron_weighted_bin_fractions=per_neuron.mean(axis=0).tolist() if len(means) else None,
            event_weighted_mean_gap_ticks=float(self.sums[ids].sum()/total) if total else None,
            neuron_weighted_mean_gap_ticks=float(means.mean()) if len(means) else None,
            quantiles_of_per_neuron_mean_completed_gap_ticks=
                dict(zip(['p10','p50','p90','p99'], np.quantile(means,[.1,.5,.9,.99]).tolist())) if len(means) else None,
            left_censored_prefix_event_free_ticks=int(self.first[ids][ever].sum()),
            right_censored_suffix_event_free_ticks=int((ticks-1-self.last[ids][ever]).sum()),
            fully_censored_neuron_ticks=int((~ever).sum()*ticks))
        if indices is None and total:
            cdf = self.histogram.cumsum()
            result['event_weighted_gap_quantiles_ticks'] = {
                name: int(np.searchsorted(cdf, q*total))
                for name,q in [('p10',.1),('p50',.5),('p90',.9),('p99',.99)]}
        return result


@torch.no_grad()
def profile(checkpoint, library, frames):
    torch.set_num_threads(1)
    identities = {str(p): checksum(p) for p in (checkpoint,library,Path(__file__))}
    model = load_checkpoint(checkpoint)
    model.config = replace(model.config, metrics_mode='events')
    kernel = NativeCPU(library, threads=4)
    kernel.enable(model)
    net = model.network
    ticks = frames * model.config.neural_steps
    natural, materialized = Gaps(net.n,ticks), Gaps(net.n,ticks)
    totals = dict(arrivals=0,injection=0,spikes=0,autonomous_spikes=0,refractory_expiry=0)
    tick = 0
    original = net.step

    def observed(injection, **kwargs):
        nonlocal tick
        expiry = (net.refractory == 1).flatten().numpy().copy()
        activity = original(injection, **kwargs)
        arrived = np.zeros(net.n,dtype=bool)
        arrived[net.post[activity.arrival_edges].numpy()] = True
        injected = injection.flatten().numpy() != 0
        spikes = activity.spikes.flatten().numpy()
        external = arrived | injected
        masks = dict(arrivals=arrived,injection=injected,spikes=spikes,
                     autonomous_spikes=spikes & ~external,refractory_expiry=expiry)
        for name,mask in masks.items():
            totals[name] += int(mask.sum())
        events = external | spikes | expiry
        natural.observe(tick, events)
        # Artificial schedule only; no added materialization is executed.
        materialized.observe(tick, np.ones(net.n,dtype=bool)
                             if (tick+1) % 8 == 0 else events)
        tick += 1
        return activity

    net.step = observed
    start = time.perf_counter()
    model.run(frames)
    seconds = time.perf_counter()-start
    assert tick == ticks
    assert all(checksum(path) == identity for path,identity in identities.items())
    classes = {}
    for i,name in enumerate(model.retina.spec['cell_types'] or ['unannotated']*net.n):
        classes.setdefault(name,[]).append(i)
    return dict(checkpoint=checkpoint,native_library=library,sha256_before=identities,
        sha256_after={path:checksum(path) for path in identities},frames=frames,ticks=ticks,
        neurons=net.n,torch_threads=1,native_threads=4,dt_seconds=net.config.dt,
        instrumented_seconds=seconds,gap_bins_ticks=['1-3','4-9','10-99','100+'],
        category_neuron_ticks=totals,natural=natural.summary(ticks),
        artificial_every_8_ticks=materialized.summary(ticks),
        classes={name:natural.summary(ticks,ids) for name,ids in sorted(classes.items())})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--native-library',required=True)
    parser.add_argument('--frames',type=int,default=1000)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    if args.frames < 1:
        parser.error('positive frames required')
    result=profile(args.checkpoint,args.native_library,args.frames)
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='classes'},indent=2))
