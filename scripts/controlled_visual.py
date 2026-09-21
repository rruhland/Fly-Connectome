"""Bounded moving-dot learning experiment on an induced measured visual circuit."""
import numpy as np
import torch
import time
import argparse
import json
from pathlib import Path

from fly_connectome.graph import Graph
from fly_connectome.dynamics import Network, NeuronConfig
from fly_connectome.plasticity import Plasticity, LearningConfig
from fly_connectome.sensor import Retina, EventCamera
from fly_connectome.data import checksum


def crop_payload(payload, nodes):
    """Keep every measured edge between selected bodies, without renormalizing."""
    m = payload['metadata']
    g = Graph(**m['graph'], gain=m['gain'])
    nodes = np.unique(nodes)
    keep = np.zeros(len(g.body_ids), dtype=bool)
    keep[nodes] = True
    edges = np.flatnonzero(keep[g.pre] & keep[g.post])
    graph = Graph(g.body_ids[nodes], np.searchsorted(nodes, g.pre[edges]),
                  np.searchsorted(nodes, g.post[edges]), g.contacts[edges], g.signs[nodes], g.gain)
    retina = dict(m['retina'])
    for key in ('neuron_columns', 'cell_types'):
        retina[key] = [retina[key][i] for i in nodes]
    return dict(graph=graph, retina=retina, delays=[m['delays'][i] for i in edges],
                pathways=[m['pathways'][i] for i in edges],
                weights=payload['state']['network']['magnitudes'][edges].clone()), nodes, edges


def trajectory(height, width, xs, y, blank):
    frames = torch.zeros(blank + len(xs) + 1, 1, height, width, dtype=torch.bool)
    for i, x in enumerate(xs):
        frames[blank+i, 0, y, x] = True
    return frames


def score_forecasts(target, prediction, boundaries, lead):
    if lead < 1:
        raise ValueError('forecasts require a positive lead')
    issued = np.asarray(boundaries)
    issued = issued[issued + lead < len(target)]
    y, p = target[issued+lead], prediction[issued]
    event = y != 0
    error = (y-p)**2
    result = dict(samples=int(y.size), events=int(event.sum()), mse=float(error.mean()),
                event_mse=float(error[event].mean()) if event.any() else None,
                quiet_mse=float(error[~event].mean()) if (~event).any() else None,
                mean_signed_event_prediction=float((p[event]*y[event]).mean()) if event.any() else None,
                false_alarm_fraction=float((abs(p[~event]) >= .1).mean()) if (~event).any() else None)
    for name, mask in (('on', y < 0), ('off', y > 0)):
        result[name+'_mse'] = float(error[mask].mean()) if mask.any() else None
        result[name+'_anticipation'] = float((p[mask]*y[mask]).mean()) if mask.any() else None
    return result


def make_network(crop, metadata, weights=None, *, predictive_kinetics='original', capture_warmup=False):
    network_class = Network
    if predictive_kinetics == 'slow-excitation-v1':
        from signed_kinetics import SignedKineticsNetwork
        network_class = SignedKineticsNetwork
    elif predictive_kinetics != 'original':
        raise ValueError('unknown predictive kinetics')
    net = network_class(crop['graph'], crop['delays'], crop['pathways'],
                  config=NeuronConfig(**metadata['neurons']), cell_types=crop['retina']['cell_types'])
    net.magnitudes.copy_(crop['weights'] if weights is None else weights)
    warmup = []
    for _ in range(metadata['config']['warmup_steps']):
        activity = net.step(torch.zeros_like(net.voltage))
        if capture_warmup:
            warmup.append(activity.spikes[0].numpy().copy())
    if capture_warmup:
        net.warmup_spikes = np.asarray(warmup, dtype=bool).reshape(-1, net.n)
    return net


@torch.no_grad()
def run_sequence(net, crop, metadata, frames, targets, *, learning, deadline=float('inf'), visual_schedule='tick-v1'):
    """Continuous state across trials; the only neural input is the current image."""
    cfg = metadata['config']
    rule_cfg = LearningConfig(**metadata['learning'])
    retina = Retina(**crop['retina'])
    camera = EventCamera(1, retina.spec['height'], retina.spec['width'])
    rule_class = Plasticity
    if visual_schedule == 'frame-horizon-v1':
        from frame_prediction import FramePrediction
        if cfg['neural_steps'] != 8:
            raise ValueError('frame-horizon-v1 requires eight ticks per frame')
        rule_class = FramePrediction
    elif visual_schedule != 'tick-v1':
        raise ValueError('unknown visual supervision schedule')
    rule = rule_class(net, rule_cfg, sensory_mask=retina.injected,
                      sensory_gain=cfg['sensory_gain']) if learning else None
    incoming = ((net.pathways == 1) & torch.isin(net.post, torch.tensor(targets))).nonzero().flatten()
    lookup = torch.full((net.e,), -1, dtype=torch.long)
    lookup[incoming] = torch.arange(len(incoming))
    updates = torch.zeros(3, len(incoming), dtype=torch.float64)  # ON, OFF, quiet
    arrivals = torch.zeros(net.e, dtype=torch.long)
    prediction, target, spikes, eligibility = [], [], [], []
    state_names = ('voltage', 'sensory_state', 'feedforward_current', 'predictive_current',
                   'behavioral_current', 'adaptation', 'magnitudes')
    maxima = torch.zeros(len(state_names))
    for frame_index, frame in enumerate(frames):
        if time.perf_counter() > deadline:
            raise TimeoutError('controlled experiment exceeded its wall-clock budget')
        injection = retina.project(camera.observe(frame)) * cfg['sensory_gain']
        for tick in range(cfg['neural_steps']):
            e = torch.zeros(len(incoming))
            if rule is not None:
                positions = lookup[rule.keys]
                keep = positions >= 0
                e[positions[keep]] = rule.values[keep]
            activity = net.step(injection if tick == 0 else torch.zeros_like(injection), capture_increments=True)
            peak = torch.stack([getattr(net, name).abs().max() for name in state_names])
            if not torch.isfinite(peak).all():
                raise ValueError('nonfinite raw neural state')
            if net.magnitudes.min() < 0 or net.magnitudes.max() > rule_cfg.maximum_weight:
                raise ValueError('weight outside fixed bounds')
            maxima = torch.maximum(maxima, peak)
            observed = rule_cfg.observation(activity, net.config.threshold, retina.injected, cfg['sensory_gain'])
            prediction.append(rule_cfg.encode(activity.predicted[0, targets], net.config.threshold).numpy().copy())
            target.append(observed[0, targets].numpy().copy())
            spikes.append(activity.spikes[0].numpy().copy())
            if learning:
                eligibility.append(e.numpy())
            arrivals.index_add_(0, activity.arrival_edges, torch.ones_like(activity.arrival_edges))
            if rule is not None:
                y = observed[0, net.post[incoming]]
                delta = rule_cfg.eta_prediction * (y-rule.expected[0, net.post[incoming]]) * e
                if visual_schedule == 'tick-v1':
                    for category, mask in enumerate((y < 0, y > 0, y == 0)):
                        updates[category, mask] += delta[mask].double()
                rule.observe(activity, torch.zeros(1))
                if visual_schedule == 'frame-horizon-v1' and rule.last_visual_update is not None:
                    edges, y, delta = rule.last_visual_update
                    positions = lookup[edges]
                    for category, mask in enumerate((y < 0, y > 0, y == 0)):
                        keep = mask & (positions >= 0)
                        updates[category, positions[keep]] += delta[keep].double()
        if rule is not None and (frame_index+1) % cfg['sync_steps'] == 0:
            rule.synchronize()
    if rule is not None:
        rule.synchronize()
    result = dict(prediction=np.asarray(prediction), target=np.asarray(target), spikes=np.asarray(spikes),
                  incoming_edges=incoming.numpy(), arrival_counts=arrivals.numpy(),
                  stability=dict(finite=True, within_weight_bounds=True,
                                 max_abs=dict(zip(state_names, maxima.tolist()))))
    if learning:
        result.update(eligibility=np.asarray(eligibility), local_update_sums=updates.numpy())
    return result


def passed(trained, frozen, zero, persistence):
    return (all(trained['mse'] <= .8*b['mse'] for b in (frozen, zero, persistence))
            and all(trained[s+'_mse'] is not None and trained[s+'_mse'] <= .9*b[s+'_mse']
                    for s in ('on', 'off') for b in (zero, frozen))
            and all(trained[s+'_anticipation'] >= .1 for s in ('on', 'off'))
            and trained['false_alarm_fraction'] <= .05)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='checkpoints/event-v1-combined-rate-initial.pt')
    parser.add_argument('--output', default='runs/controlled-visual-v1')
    args = parser.parse_args()
    torch.set_num_threads(1)
    source_sha = checksum(args.checkpoint)
    payload = torch.load(args.checkpoint, weights_only=True)
    m = payload['metadata']
    retina = Retina(**m['retina'])
    height, width = retina.spec['height'], retina.spec['width']
    if (height, width) != (32, 64) or m['config']['neural_steps'] != 8:
        raise ValueError('this fixed protocol requires the original 64x32, eight-tick setup')
    bins = retina.pixel_bins.reshape(height, width)
    columns = bins[16, 28:37].numpy()
    mapped = np.array(retina.spec['neuron_columns'])
    sensory = np.flatnonzero(np.isin(mapped, columns))
    pre, post = (np.array(m['graph'][key]) for key in ('pre', 'post'))
    intermediates = np.intersect1d(post[np.isin(pre, sensory)], pre[np.isin(post, sensory)])
    crop, nodes, edges = crop_payload(payload, np.union1d(sensory, intermediates))
    targets = np.flatnonzero(np.array(crop['retina']['neuron_columns']) == int(bins[16, 32]))
    if len(targets) != 3:
        raise ValueError('expected exactly three central sensory targets')
    # Verify the selected patch is weakly connected before executing any neurons.
    reachable = {0}
    while True:
        linked = set(crop['graph'].post[np.isin(crop['graph'].pre, list(reachable))])
        linked |= set(crop['graph'].pre[np.isin(crop['graph'].post, list(reachable))])
        expanded = reachable | linked
        if expanded == reachable:
            break
        reachable = expanded
    if len(reachable) != len(nodes):
        raise ValueError('fixed anatomical patch is disconnected; revise extraction before training')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    manifest = dict(source_checkpoint=args.checkpoint, source_sha256=source_sha,
        parent_graph_sha256=Graph(**m['graph'], gain=m['gain']).identity(),
        graph_sha256=crop['graph'].identity(), neurons=len(nodes), edges=len(edges),
        body_ids=crop['graph'].body_ids.tolist(), parent_edge_indices=edges.tolist(),
        cut_incoming=int((~np.isin(pre, nodes) & np.isin(post, nodes)).sum()),
        cut_outgoing=int((np.isin(pre, nodes) & ~np.isin(post, nodes)).sum()),
        stimulus_columns=columns.tolist(), target_bodies=crop['graph'].body_ids[targets].tolist(),
        target_types=[crop['retina']['cell_types'][i] for i in targets],
        config=m['config'], neurons_config=m['neurons'], learning=m['learning'],
        train_seed=9001, evaluation_seed=9002, train_trials=200, evaluation_trials=50,
        primary_lead_ticks=8, wall_budget_seconds=600)
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    schedules = []
    for count, seed in ((200, 9001), (50, 9002)):
        rng = np.random.default_rng(seed)
        blanks = rng.integers(12, 37, size=count)
        frames = torch.cat([trajectory(height, width, range(28, 37), 16, int(b)) for b in blanks])
        schedules.append((frames, blanks))
    started = time.perf_counter()
    deadline = started + 600
    training_net = make_network(crop, m)
    print(f'Training {len(nodes)} neurons, {len(edges)} edges, 200 traversals', flush=True)
    train = run_sequence(training_net, crop, m, schedules[0][0], targets, learning=True, deadline=deadline)
    learned = training_net.magnitudes.clone()
    if not torch.isfinite(learned).all() or not torch.isfinite(training_net.voltage).all():
        raise ValueError('nonfinite training state')
    np.savez_compressed(output/'training.npz', **{k:v for k,v in train.items() if k != 'stability'}, weights_initial=crop['weights'].numpy(),
                        weights_trained=learned.numpy(), blank_frames=schedules[0][1])
    metrics, evaluations = {}, {}
    frames = schedules[1][0]
    for name, weights in (('frozen', crop['weights']), ('trained', learned)):
        print(f'Evaluating {name} weights on 50 new traversals', flush=True)
        net = make_network(crop, m, weights)
        result = run_sequence(net, crop, m, frames, targets, learning=False, deadline=deadline)
        if not torch.equal(net.magnitudes, weights):
            raise AssertionError('evaluation changed weights')
        np.savez_compressed(output/(name+'.npz'), **{k:v for k,v in result.items() if k != 'stability'}, blank_frames=schedules[1][1])
        evaluations[name] = result
        boundaries = np.arange(0, len(result['target']), 8)
        metrics[name] = score_forecasts(result['target'], result['prediction'], boundaries, 8)
        metrics[name]['per_target'] = [score_forecasts(result['target'][:, i], result['prediction'][:, i], boundaries, 8)
                                      for i in range(len(targets))]
        metrics[name]['one_tick'] = score_forecasts(result['target'], result['prediction'], boundaries[1:]-1, 1)
        metrics[name]['mean_rate_hz'] = float(result['spikes'].mean()/net.config.dt)
        metrics[name]['stability'] = result['stability']
    y = evaluations['frozen']['target']
    if not np.array_equal(y, evaluations['trained']['target']):
        raise AssertionError('evaluation sensory targets differ')
    oracle = np.zeros_like(y)
    # Uses only each current image and the fixed motion law, never future labels.
    oracle[::8] = (-frames[:, 0, 16, 31].numpy().astype(float)
                    + frames[:, 0, 16, 32].numpy().astype(float))[:, None]
    for name, prediction in (('zero', np.zeros_like(y)), ('persistence', y), ('stimulus_control', oracle)):
        metrics[name] = score_forecasts(y, prediction, boundaries, 8)
    if metrics['stimulus_control']['mse'] != 0:
        raise AssertionError('causal stimulus control did not predict the simple task')
    if checksum(args.checkpoint) != source_sha:
        raise AssertionError('source checkpoint changed')
    result = dict(manifest=manifest, status='pass' if passed(metrics['trained'], metrics['frozen'], metrics['zero'], metrics['persistence']) else 'fail',
                  seconds=time.perf_counter()-started, metrics=metrics,
                  training_frames=len(schedules[0][0]), evaluation_frames=len(frames),
                  weight_change_l1=float((learned-crop['weights']).abs().sum()),
                  training_stability=train['stability'],
                  target_update_sums_on_off_quiet=train['local_update_sums'].sum(axis=1).tolist(),
                  source_unchanged=True)
    (output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('status', 'seconds', 'target_update_sums_on_off_quiet')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
