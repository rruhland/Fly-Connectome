"""Bounded opt-in local next-frame learning on vertical T5 predictive edges."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyarrow.feather as feather
import torch

from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.plasticity import LearningConfig
from fly_connectome.sensor import EventCamera, Retina
from full_context_pong import NETWORK_STATE
from motion_graded_propagation import GRADED_STATE
from motion_stage_audit import SOURCE
from motion_stage_locality import ANNOTATIONS, infer_columns
from motion_t5_axis_aligned import moving_bar
from t5_local_learning import T5BalancedFramePrediction, T5FramePrediction
from t5_local_order_current import ORDER_STATE, T5LocalOrderNetwork


OUT = Path('docs/experiments/2026-09-23-t5-local-learning-results.json')
AUDIT_OUT = Path('docs/experiments/2026-09-23-t5-credit-conflict-results.json')
BALANCED_OUT = Path('docs/experiments/2026-09-23-t5-balanced-credit-results.json')
STATE = NETWORK_STATE+GRADED_STATE+ORDER_STATE
EPISODES = 40
EVENT_THRESHOLD = .01


def metrics(target, prediction, persistence):
    target = np.concatenate(target)
    prediction = np.concatenate(prediction)
    persistence = np.concatenate(persistence)
    event = np.abs(target) > EVENT_THRESHOLD

    def score(candidate):
        error = (target-candidate)**2
        return dict(mse=float(error.mean()),
            event_mse=float(error[event].mean()) if event.any() else None,
            quiet_mse=float(error[~event].mean()) if (~event).any() else None,
            event_anticipation=float((candidate[event]*np.sign(target[event])).mean())
                if event.any() else None,
            false_alarm_fraction=float((np.abs(candidate[~event]) >= .1).mean())
                if (~event).any() else None)

    return dict(samples=int(target.size), events=int(event.sum()),
        target_mean=float(target.mean()), forecast=score(prediction),
        zero=score(np.zeros_like(target)), persistence=score(persistence))


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--credit-audit-only', action='store_true')
    parser.add_argument('--balanced', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(4)
    started = time.perf_counter()
    source_sha = checksum(SOURCE)
    payload = torch.load(SOURCE, weights_only=True)
    metadata = payload['metadata']
    graph = Graph(**metadata['graph'], gain=metadata['gain'])
    retina = Retina(**metadata['retina'])
    types = np.asarray(metadata['retina']['cell_types'])
    config = LearningConfig(**metadata['learning'])
    if config.visual_target != 'input-arrivals-v1':
        raise AssertionError('experiment requires local input-arrival targets')
    net = T5LocalOrderNetwork(graph, metadata['delays'], metadata['pathways'],
        config=NeuronConfig(**metadata['neurons']),
        cell_types=metadata['retina']['cell_types'], target_mask=retina.injected,
        release_cap=.10, voltage_scale=.02,
        source_type='Tm9', source_state='current', gate_gain=8000., gate_cap=1.)
    net.set_weights(payload['state']['network']['magnitudes'].clamp(
        0, config.maximum_weight))
    initial_weights = net.magnitudes.clone()
    for cell_type in ('Tm4', 'Tm9'):
        net.rest_current[torch.tensor(types == cell_type)] = .85
    zero = torch.zeros_like(net.voltage)
    for _ in range(metadata['config']['warmup_steps']):
        net.step(zero)
    net.enable_graded()
    for _ in range(256):
        net.step(zero)
    net.enable_order()
    settled = {name: getattr(net, name).clone() for name in STATE}
    settled_tick = net.step_index

    t5_mask = torch.tensor(np.isin(types, ('T5c', 'T5d')))
    rule_class = T5BalancedFramePrediction if args.balanced else T5FramePrediction
    rule = rule_class(net, config, target_mask=t5_mask,
        sensory_mask=retina.injected,
        sensory_gain=metadata['config']['sensory_gain'])
    learnable = rule.learnable_edges
    if not (net.pathways[learnable] == 1).all():
        raise AssertionError('T5 learning includes nonpredictive edges')
    camera = EventCamera(1, 32, 64)
    camera.previous.fill_(True)
    updates = dict(forecast_confirmations=0, nonzero_updates=0,
        positive_updates=0, negative_updates=0, total_absolute_delta=0.)
    updated_edges = set()
    edge_lookup = torch.full((net.e,), -1, dtype=torch.long)
    edge_lookup[learnable] = torch.arange(int(learnable.sum()))
    credit = {category: dict(sum=torch.zeros(int(learnable.sum()), dtype=torch.float64),
                             absolute=torch.zeros(int(learnable.sum()), dtype=torch.float64),
                             count=torch.zeros(int(learnable.sum()), dtype=torch.long),
                             positive=0, negative=0)
              for category in ('event', 'quiet')}
    target_events = target_samples = target_sum = 0
    train_spikes = 0
    for episode in range(EPISODES):
        center = (10, 22)[(episode//2) % 2]
        direction = 1 if episode % 2 == 0 else -1
        for frame in moving_bar('vertical', center, direction, 1):
            sensory = retina.project(camera.observe(frame)) * metadata['config']['sensory_gain']
            for tick in range(8):
                activity = net.step(sensory if tick == 0 else zero,
                                    capture_increments=True)
                if tick == 0:
                    observed = config.observation(activity, net.config.threshold,
                                                  retina.injected,
                                                  metadata['config']['sensory_gain'])
                    local = observed[0, t5_mask]
                    target_events += int((local.abs() > EVENT_THRESHOLD).sum())
                    target_samples += local.numel()
                    target_sum += float(local.sum())
                train_spikes += int(activity.spikes[0, t5_mask].sum())
                rule.observe(activity, torch.zeros(1))
                if rule.last_visual_update is not None:
                    edges, target, delta = rule.last_visual_update
                    updates['forecast_confirmations'] += len(edges)
                    changed = delta.abs() > 1e-10
                    updated_edges.update(edges[changed].tolist())
                    updates['nonzero_updates'] += int(changed.sum())
                    updates['positive_updates'] += int((delta > 1e-10).sum())
                    updates['negative_updates'] += int((delta < -1e-10).sum())
                    updates['total_absolute_delta'] += float(delta.abs().sum())
                    positions = edge_lookup[edges]
                    if (positions < 0).any():
                        raise AssertionError('non-T5 forecast received credit')
                    for category, mask in (('event', target.abs() > EVENT_THRESHOLD),
                                           ('quiet', target.abs() <= EVENT_THRESHOLD)):
                        selected = mask & changed
                        row = credit[category]
                        row['sum'].index_add_(0, positions[selected], delta[selected].double())
                        row['absolute'].index_add_(0, positions[selected],
                                                   delta[selected].abs().double())
                        row['count'].index_add_(0, positions[selected],
                                                torch.ones_like(positions[selected]))
                        row['positive'] += int((delta[selected] > 0).sum())
                        row['negative'] += int((delta[selected] < 0).sum())
            rule.proposals[~learnable] = 0
            rule.homeostatic_exponent[~learnable] = 0
            rule.synchronize()
        if (episode+1) % 10 == 0:
            print(json.dumps(dict(episodes=episode+1,
                changed_edges=len(updated_edges),
                elapsed_seconds=round(time.perf_counter()-started, 1))), flush=True)
    trained_weights = net.magnitudes.clone()
    change = trained_weights[learnable]-initial_weights[learnable]
    if not (torch.equal(trained_weights[~learnable], initial_weights[~learnable])
            and torch.isfinite(trained_weights).all()
            and (trained_weights >= 0).all()
            and (trained_weights <= config.maximum_weight).all()):
        raise AssertionError('training changed non-T5 edges or violated bounds')

    if args.credit_audit_only:
        event, quiet = credit['event'], credit['quiet']
        both = (event['count'] > 0) & (quiet['count'] > 0)
        conflict = both & (event['sum']*quiet['sum'] < 0)
        denominator = event['sum'].abs()+quiet['sum'].abs()
        cancellation = (1-(event['sum']+quiet['sum']).abs()
                        /denominator.clamp(min=1e-20))
        categories = {category: dict(
            nonzero_proposals=int(row['count'].sum()),
            positive_proposals=row['positive'],
            negative_proposals=row['negative'],
            edges=int((row['count'] > 0).sum()),
            signed_proposal_mass=float(row['sum'].sum()),
            absolute_proposal_mass=float(row['absolute'].sum()))
            for category, row in credit.items()}
        audit = dict(source_sha256=source_sha, graph_sha256=graph.identity(),
            episodes=EPISODES, learnable_edges=int(learnable.sum()),
            updated_edges=len(updated_edges),
            maximum_absolute_weight_change=float(change.abs().max()),
            target_event_fraction=target_events/max(target_samples, 1),
            categories=categories,
            both_category_edges=int(both.sum()),
            conflict_edges=int(conflict.sum()),
            event_positive_quiet_negative=int((conflict & (event['sum'] > 0)).sum()),
            event_negative_quiet_positive=int((conflict & (event['sum'] < 0)).sum()),
            mean_conflict_cancellation=float(cancellation[conflict].mean())
                if conflict.any() else None,
            aggregate_conflict_cancellation=float(1-(
                (event['sum'][conflict]+quiet['sum'][conflict]).abs().sum()
                /denominator[conflict].sum())) if conflict.any() else None,
            elapsed_seconds=time.perf_counter()-started)
        if checksum(SOURCE) != source_sha:
            raise AssertionError('source checkpoint changed')
        AUDIT_OUT.write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
        print(json.dumps(dict(output=str(AUDIT_OUT),
            both_category_edges=audit['both_category_edges'],
            conflict_edges=audit['conflict_edges'],
            categories=categories)), flush=True)
        return

    annotations = feather.read_table(
        ANNOTATIONS, columns=['bodyId', 'assignedOlHex1', 'assignedOlHex2'])
    target_columns, _ = infer_columns(metadata, retina, annotations)
    y, x = torch.meshgrid(torch.arange(32), torch.arange(64), indexing='ij')
    local_groups = {}
    for center in (10, 22, 16):
        field = ((y >= center-8) & (y <= center+8)
                 & (x >= 24) & (x <= 40))
        bins = retina.pixel_bins[field.flatten()].unique().numpy()
        local_groups[center] = {subtype: torch.tensor(np.flatnonzero(
            (types == subtype) & np.isin(target_columns, bins)))
            for subtype in ('T5c', 'T5d')}
        if any(not len(indices) for indices in local_groups[center].values()):
            raise AssertionError('missing retinotopically local T5 cells')

    def capture(weights, center, direction, speed):
        for name, value in settled.items():
            getattr(net, name).copy_(value)
        net.magnitudes.copy_(weights)
        net.step_index = settled_tick
        replay = EventCamera(1, 32, 64)
        replay.previous.fill_(True)
        samples = {name: dict(target=[], prediction=[], persistence=[])
                   for name in ('T5c', 'T5d')}
        prior_prediction = prior_observation = None
        pixel_events = 0
        for frame_index, frame in enumerate(moving_bar('vertical', center,
                                                       direction, speed)):
            events = replay.observe(frame)
            if 3 <= frame_index < 15:
                pixel_events += len(events.pixels)
            sensory = retina.project(events)*metadata['config']['sensory_gain']
            for tick in range(8):
                activity = net.step(sensory if tick == 0 else zero,
                                    capture_increments=True)
                if tick == 0:
                    observed = config.observation(activity, net.config.threshold,
                        retina.injected, metadata['config']['sensory_gain'])[0]
                    predicted = config.encode(activity.predicted[0],
                                              net.config.threshold)
                    if prior_prediction is not None:
                        for subtype, indices in local_groups[center].items():
                            row = samples[subtype]
                            row['target'].append(observed[indices].numpy().copy())
                            row['prediction'].append(
                                prior_prediction[indices].numpy().copy())
                            row['persistence'].append(
                                prior_observation[indices].numpy().copy())
                    prior_prediction = predicted.clone()
                    prior_observation = observed.clone()
        if not torch.isfinite(net.voltage).all():
            raise AssertionError('nonfinite held-out neural state')
        return dict(pixel_events=pixel_events, samples=samples)

    report = dict(source_sha256=source_sha, graph_sha256=graph.identity(),
        episodes=EPISODES, ticks_per_frame=8, target_event_threshold=EVENT_THRESHOLD,
        gate_gain=8000., source_release_cap=.10,
        local_event_balance=dict(enabled=args.balanced, decay_per_frame=.98,
            maximum_event_gain=4. if args.balanced else 1.,
            mean_applied_event_gain=(rule.event_gain_sum/max(rule.event_gain_samples, 1))
                if args.balanced else 1.,
            event_gain_samples=rule.event_gain_samples if args.balanced else 0),
        learnable_edges=int(learnable.sum()),
        training=dict(updates=updates, distinct_updated_edges=len(updated_edges),
            target_events=target_events, target_samples=target_samples,
            target_event_fraction=target_events/max(target_samples, 1),
            target_mean=target_sum/max(target_samples, 1),
            t5_spikes=train_spikes,
            changed_edges=int((change.abs() > 1e-8).sum()),
            median_absolute_weight_change=float(change.abs().median()),
            max_absolute_weight_change=float(change.abs().max()),
            minimum_weight=float(trained_weights.min()),
            maximum_weight=float(trained_weights.max())),
        evaluations={}, passes_pilot=False)
    for center, speed in ((10, 1), (22, 1), (16, 1), (16, 2)):
        pair = {}
        for direction in (1, -1):
            frozen = capture(initial_weights, center, direction, speed)
            trained = capture(trained_weights, center, direction, speed)
            if frozen['pixel_events'] != trained['pixel_events']:
                raise AssertionError('trained and frozen camera events differ')
            for subtype in ('T5c', 'T5d'):
                for label, case in (('frozen', frozen), ('trained', trained)):
                    row = case['samples'][subtype]
                    pair[f'{subtype}/{direction:+d}/{label}'] = metrics(
                        row['target'], row['prediction'], row['persistence'])
                a = np.concatenate(frozen['samples'][subtype]['target'])
                b = np.concatenate(trained['samples'][subtype]['target'])
                pair[f'{subtype}/{direction:+d}/target_difference'] = float(
                    np.abs(a-b).mean())
        report['evaluations'][f'{center}-s{speed}'] = pair
        print(json.dumps(dict(evaluation=f'{center}-s{speed}',
            event_mse={subtype: dict(frozen=pair[f'{subtype}/+1/frozen']
                ['forecast']['event_mse'], trained=pair[f'{subtype}/+1/trained']
                ['forecast']['event_mse']) for subtype in ('T5c', 'T5d')})),
            flush=True)

    def passes(subtype, speed):
        row = report['evaluations'][f'16-s{speed}']
        for direction in (1, -1):
            frozen = row[f'{subtype}/{direction:+d}/frozen']
            trained = row[f'{subtype}/{direction:+d}/trained']
            f, t = frozen['forecast'], trained['forecast']
            if (f['event_mse'] is None or t['event_mse'] is None
                    or t['event_mse'] > .9*f['event_mse']
                    or t['quiet_mse'] > 1.1*f['quiet_mse']
                    or t['event_anticipation'] <= f['event_anticipation']):
                return False
        return True

    report['passes_pilot'] = bool(all(passes(subtype, speed)
        for subtype in ('T5c', 'T5d') for speed in (1, 2)))
    report['elapsed_seconds'] = time.perf_counter()-started
    if checksum(SOURCE) != source_sha:
        raise AssertionError('source checkpoint changed')
    output = BALANCED_OUT if args.balanced else OUT
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output),
        updated_edges=len(updated_edges),
        passes_pilot=report['passes_pilot'],
        elapsed_seconds=round(report['elapsed_seconds'], 1))), flush=True)


if __name__ == '__main__':
    main()
