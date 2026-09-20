"""Frozen audit of event/current target alignment and sign support; never trains."""
import argparse
import json
from pathlib import Path

import torch

from fly_connectome.data import checksum
from fly_connectome.sensor import EventCamera
from fly_connectome.training import load_checkpoint


def sums(target, prediction, previous):
    active = target != 0
    error = (target-prediction).square()
    return torch.stack((torch.tensor(target.numel(), device=target.device), active.sum(),
        error.sum(), target.square().sum(), (target-previous).square().sum(),
        error[active].sum(), error[~active].sum(), (target*prediction).sum(),
        prediction.square().sum(), target.sum(), prediction.sum())).double()


def summary(values):
    count, active, error, zero, persistence, active_error, quiet_error, cross, power, targets, predictions = values.tolist()
    return dict(samples=count, active_samples=active, model_mse=error/count, zero_mse=zero/count,
        persistence_mse=persistence/count, active_mse=active_error/max(1,active),
        quiet_mse=quiet_error/max(1,count-active), target_prediction_cross_mean=cross/count,
        prediction_power=power/count, target_mean=targets/count, prediction_mean=predictions/count)


def lagged_scores(targets, predictions, lags=(-8,-4,-2,-1,0,1,2,4,8)):
    """Offline timing diagnostic; positive lag uses predictions AFTER the target.

    Future offsets are never used in training or acceptance scores.
    """
    result = {}
    for lag in lags:
        if abs(lag) >= len(targets):
            continue
        start, end = max(0,-lag), min(len(targets),len(targets)-lag)
        target, prediction = targets[start:end], predictions[start+lag:end+lag]
        result[str(lag)] = summary(sums(target,prediction,torch.zeros_like(target)))
        del result[str(lag)]['persistence_mse']  # no persistence comparison at shifted offsets
    return result


def audit(checkpoint, steps, seeds, event_groups=None):
    model = load_checkpoint(checkpoint, evaluation=True, seeds=seeds)
    assert model.plasticity is None
    net = model.network
    types = model.retina.spec['cell_types']
    labels = ('L1','L2','L3','Mi1','Tm3','Tm1','Tm2','T4','T5')
    masks = {t:torch.tensor([v == t or (t in ('T4','T5') and v.startswith(t)) for v in types]) for t in labels}
    masks = {t:m for t,m in masks.items() if m.any()}
    current = {t:torch.zeros(11,dtype=torch.float64) for t in masks}
    learning = {t:torch.zeros(11,dtype=torch.float64) for t in masks}
    events = {t:torch.zeros(11,dtype=torch.float64) for t in ('L1','L2','L3') if t in masks}
    grouped_events = {name:torch.zeros(11,dtype=torch.float64) for name in (event_groups or {})}
    for mask in (event_groups or {}).values():
        if mask.shape != model.retina.injected.shape or mask.dtype != torch.bool or (mask&~model.retina.injected).any() or not mask.any():
            raise ValueError('event groups require nonempty sensory-only masks on the checkpoint roster')
    saturated = {t:torch.zeros((),dtype=torch.float64) for t in masks}
    raw_power = {t:torch.zeros((),dtype=torch.float64) for t in masks}
    original_step = net.step
    def record_step(injection, **kwargs):
        activity = original_step(injection, **kwargs)
        observed = model.learning_config.encode(activity.observed, net.config.threshold)
        target = model.learning_config.observation(activity, net.config.threshold, model.retina.injected, model.config.sensory_gain)
        for label, mask in masks.items():
            current[label] += sums(observed[:,mask], model.previous_predicted[:,mask], model.previous_observed[:,mask])
            previous = (model.previous_learning_observed if model.learning_config.visual_target == 'input-arrivals-v1'
                        else model.previous_observed)
            learning[label] += sums(target[:,mask], model.previous_predicted[:,mask], previous[:,mask])
            saturated[label] += (activity.observed[:,mask].abs() >= net.config.threshold).sum()
            raw_power[label] += activity.observed[:,mask].square().sum()
        return activity
    # Only the disposable frozen model is instrumented, with identical step outputs.
    net.step = record_step
    camera = EventCamera(len(seeds), model.retina.spec['height'], model.retina.spec['width'])
    event_targets, event_predictions = [], []
    for _ in range(steps):
        frame = model.environment.render(model.retina.spec['height'], model.retina.spec['width'])
        target = model.retina.project(camera.observe(frame))
        event_targets.append(target[:,model.retina.injected].clone())
        event_predictions.append(model.previous_predicted[:,model.retina.injected].clone())
        for label in events:
            mask = masks[label]
            events[label] += sums(target[:,mask], model.previous_predicted[:,mask], model.previous_events[:,mask])
        for label,mask in (event_groups or {}).items():
            grouped_events[label] += sums(target[:,mask],model.previous_predicted[:,mask],model.previous_events[:,mask])
        model.step('scripted')
    targets, predictions = torch.stack(event_targets), torch.stack(event_predictions)
    permutation = torch.randperm(steps,generator=torch.Generator().manual_seed(421))
    active = targets != 0
    shuffled = predictions[permutation]
    temporal_control = dict(permutation_seed=421,
        model_mse=float((targets-predictions).square().mean()),
        shuffled_mse=float((targets-shuffled).square().mean()),
        model_event_conditioned_mse=float((targets-predictions).square()[active].mean()) if active.any() else None,
        shuffled_event_conditioned_mse=float((targets-shuffled).square()[active].mean()) if active.any() else None)
    lagged = lagged_scores(targets,predictions)
    signs = {}
    for label, mask in masks.items():
        targets = mask.nonzero().flatten()
        edges = mask[net.post] & (net.pathways == 1)
        positive = torch.bincount(net.post[edges & (net.signs > 0)], minlength=net.n)[targets]
        negative = torch.bincount(net.post[edges & (net.signs < 0)], minlength=net.n)[targets]
        source_active = model.evaluation_spike_counts.sum(0)[net.pre] > 0
        active_positive = torch.bincount(net.post[edges & source_active & (net.signs > 0)], minlength=net.n)[targets]
        active_negative = torch.bincount(net.post[edges & source_active & (net.signs < 0)], minlength=net.n)[targets]
        signs[label] = dict(neurons=len(targets), predictive_edges=int(edges.sum()),
            with_positive_input=int((positive>0).sum()), with_negative_input=int((negative>0).sum()),
            with_active_positive_source=int((active_positive>0).sum()),
            with_active_negative_source=int((active_negative>0).sum()))
    return dict(checkpoint=checkpoint, checkpoint_sha256=checksum(checkpoint), graph_sha256=net.graph.identity(),
        training_steps=model.training_step, steps=steps, seeds=seeds, prediction_encoding=model.learning_config.prediction_encoding,
        visual_target=model.learning_config.visual_target, visual_eligibility=model.learning_config.visual_eligibility,
        aggregate_metrics=model.metrics,
        temporal_control=temporal_control,
        lagged_event_scores=lagged,
        learning={t:summary(v) for t,v in learning.items()},
        current={t:dict(**summary(v), saturated_fraction=saturated[t].item()/v[0].item(),
                       raw_target_power=raw_power[t].item()/v[0].item()) for t,v in current.items()},
        events={t:summary(v) for t,v in events.items()}, sign_support=signs,
        event_groups={t:summary(v) for t,v in grouped_events.items()},
        population_spikes={t:int(model.evaluation_spike_counts[:,mask].sum()) for t,mask in masks.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkpoint')
    parser.add_argument('--steps', type=int, default=500)
    parser.add_argument('--seeds', default='1101,1102')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = audit(args.checkpoint, args.steps, [int(s) for s in args.seeds.split(',')])
    Path(args.output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(events=report['events'], sign_support=report['sign_support'])), flush=True)


if __name__ == '__main__':
    main()
