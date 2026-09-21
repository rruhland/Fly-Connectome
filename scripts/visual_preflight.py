"""One frozen, sign-compatible visual motif; no learning or parameter search."""
import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from controlled_visual import crop_payload, make_network, run_sequence, trajectory
from fly_connectome.data import checksum
from fly_connectome.sensor import Retina


def delayed_traces(spikes, pre, delays, signs, decay):
    trace = np.zeros((len(spikes), len(pre)), dtype=np.float64)
    state = np.zeros(len(pre), dtype=np.float64)
    for tick in range(len(spikes)):
        state *= decay
        valid = tick >= delays
        state[valid] += signs[valid] * spikes[tick-delays[valid], pre[valid]]
        trace[tick] = state
    return trace


def supported_response(sign, prediction, control_prediction, trace, control_trace):
    compatible = (sign*trace > .001) & (sign*(trace-control_trace) > .001)
    return bool(sign*prediction > 0 and sign*(prediction-control_prediction) > .001 and compatible.any())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/event-v1-combined-rate-initial.pt')
    parser.add_argument('--output', default='runs/visual-preflight-v1')
    args = parser.parse_args()
    torch.set_num_threads(1)
    sha = checksum(args.checkpoint)
    payload = torch.load(args.checkpoint, weights_only=True)
    m = payload['metadata']
    retina = Retina(**m['retina'])
    ids = np.array(m['graph']['body_ids'])
    pre, post = (np.array(m['graph'][k]) for k in ('pre', 'post'))
    paths = np.array(m['pathways'])
    original_target = int(np.flatnonzero(ids == 38366)[0])
    incoming = np.flatnonzero((post == original_target) & (paths == 'predictive'))
    sources = pre[incoming]
    columns = retina.pixel_bins.reshape(32, 64)[16, 28:37].numpy()
    sensory = np.flatnonzero(np.isin(m['retina']['neuron_columns'], columns))
    ancestors = pre[np.isin(post, sources)]
    nodes = np.unique(np.concatenate((sensory, sources, ancestors, [original_target])))
    crop, nodes, edges = crop_payload(payload, nodes)
    target = int(np.searchsorted(nodes, original_target))
    selected = np.flatnonzero((crop['graph'].post == target) & (np.array(crop['pathways']) == 'predictive'))
    edge_pre = crop['graph'].pre[selected]
    signs = crop['graph'].signs[edge_pre]
    if set(signs.tolist()) != {-1, 1}:
        raise ValueError('target lacks a measured predictive source for each polarity')
    if len(nodes) > 128 or len(edges) > 1024:
        raise ValueError('motif exceeds bounded preflight size')
    if not np.isin(ancestors, nodes).all():
        raise AssertionError('a direct input to a predictive source was omitted')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    deadline = started + 120
    base = make_network(crop, m)
    dt = base.config.dt
    delays = np.array(crop['delays'])[selected]
    decay = base.current_decay[base.post[selected]].numpy()
    weights = base.magnitudes[selected].numpy().copy()
    trials = []
    max_error = 0.
    for trial, blank in enumerate(np.random.default_rng(9011).integers(12, 37, size=10)):
        frames = trajectory(32, 64, range(28, 37), 16, int(blank))
        event_frames = dict(on=int(blank)+4, off=int(blank)+5)
        results, traces = {}, {}
        for condition in ('full', 'on', 'off'):
            images = frames.clone()
            if condition != 'full':
                images[:event_frames[condition]] = False
            net = copy.deepcopy(base)
            result = run_sequence(net, crop, m, images, [target], learning=False, deadline=deadline)
            if not torch.equal(net.magnitudes, base.magnitudes):
                raise AssertionError('frozen weights changed')
            trace = delayed_traces(result['spikes'], edge_pre, delays, signs, decay)
            results[condition], traces[condition] = result, trace
            np.savez_compressed(output/f'trial-{trial}-{condition}.npz',
                **{k:v for k,v in result.items() if k != 'stability'}, signed_trace=trace)
        row = dict(trial=trial, blank_frames=int(blank), events={})
        for polarity, sign in (('on', -1), ('off', 1)):
            event_tick = event_frames[polarity]*8
            issue_tick = event_tick-8
            if results['full']['target'][event_tick, 0] != sign:
                raise AssertionError('actual camera target does not match specified event')
            forecasts = [float(results[c]['prediction'][issue_tick, 0]) for c in ('full', polarity)]
            for condition, forecast in zip(('full', polarity), forecasts):
                reconstructed = np.clip(traces[condition][issue_tick] @ weights/base.config.threshold, -1, 1)
                error = abs(reconstructed-forecast)
                max_error = max(max_error, error)
                if error > 1e-6:
                    raise AssertionError('signed spike trace does not reconstruct frozen forecast')
            physical, control = (traces[c][issue_tick] for c in ('full', polarity))
            source_spikes = []
            for src, delay in zip(edge_pre, delays):
                # Include only stimulus-period spikes able to arrive by forecast issue.
                indices = np.flatnonzero(results['full']['spikes'][:, src])
                indices = indices[(indices >= int(blank)*8) & (indices+delay <= issue_tick)]
                source_spikes.append((indices-event_tick).tolist())
            row['events'][polarity] = dict(event_tick=event_tick, issue_tick=issue_tick,
                prediction=forecasts[0], blank_prediction=forecasts[1],
                trace=physical.tolist(), blank_trace=control.tolist(),
                source_spike_ticks_relative_to_event=source_spikes,
                supported=supported_response(sign, *forecasts, physical, control))
        trials.append(row)
    counts = {p: sum(t['events'][p]['supported'] for t in trials) for p in ('on', 'off')}
    if checksum(args.checkpoint) != sha:
        raise AssertionError('source checkpoint changed')
    result = dict(status='pass' if min(counts.values()) >= 8 else 'fail', supported_trials=counts,
        seconds=time.perf_counter()-started, source_checkpoint=args.checkpoint, source_sha256=sha,
        source_unchanged=True, learning=False, seed=9011, neurons=len(nodes), edges=len(edges),
        graph_sha256=crop['graph'].identity(), body_ids=crop['graph'].body_ids.tolist(),
        parent_edge_indices=edges.tolist(), target_body=38366, target_type='L1',
        cut_incoming=int((~np.isin(pre, nodes)&np.isin(post, nodes)).sum()),
        cut_outgoing=int((np.isin(pre, nodes)&~np.isin(post, nodes)).sum()),
        sources=[dict(body=int(crop['graph'].body_ids[i]), cell_type=crop['retina']['cell_types'][i],
                      sign=int(s), delay=int(d), weight=float(w)) for i,s,d,w in zip(edge_pre, signs, delays, weights)],
        forecast_lead_seconds=8*dt, max_reconstruction_error=max_error, trials=trials)
    (output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('status', 'supported_trials', 'seconds', 'max_reconstruction_error')}, indent=2))


if __name__ == '__main__':
    main()
