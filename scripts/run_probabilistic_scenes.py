"""Frozen saved-candidate binding, cached-belief and ambiguity integration."""

import json
from pathlib import Path

import torch

from camera_state_audit import crossing_scene
from probabilistic_visual_state import ProbabilisticVisualState
from run_streaming_uncertainty import episode
from run_zero_shot_visual_transfer import corrupt_events
from spatial_belief import mixture_log_prob


@torch.no_grad()
def main(*, model=None, out=Path('docs/experiments/2026-09-26-probabilistic-scenes-results.json')):
    torch.set_num_threads(1)
    if model is None:
        model = ProbabilisticVisualState.load('checkpoints/m1a5/probabilistic-visual-candidate.pt')
    results = {}
    for condition in ('image_noise', 'outage'):
        retained = extras = hits = targets = missing_forecasts = 0
        forecast_nll, interval_hits = [], []
        for index, (axis, offset, background) in enumerate(
                (a, o, b) for a in ('horizontal', 'vertical', 'diagonal')
                for o in (-3, 1, 5) for b in (False, True)):
            images, events, truth = crossing_scene(axis, offset, background)
            events = corrupt_events(events, seed=215000+index, dropout=.15, false_rate=.0005)
            noise = torch.Generator().manual_seed(216000+index)
            model.reset_state()
            identities = []
            for t, (image, event) in enumerate(zip(images, events)):
                unavailable = condition == 'outage' and 16 <= t < 21
                image = (.25+.35*image+.02*torch.randn(image.shape, generator=noise)).clamp(0., 1.)
                state = model.step(torch.zeros_like(event) if unavailable else event,
                                   None if unavailable else image)
                by_id = {e['id']: e for e in state['entities']}
                if t == 6:
                    claimed = set()
                    for target in truth[t]:
                        options = [(float((e['position']-target).norm()), e['id'])
                                   for e in state['entities'] if e['id'] not in claimed]
                        distance, identity = min(options) if options else (64., None)
                        identities.append(identity if distance <= 2 else None)
                        claimed.add(identity)
                if unavailable:
                    for identity, target in zip(identities, truth[t]):
                        targets += 1
                        hits += float(state['support_field'][round(float(target[0])), round(float(target[1]))]) >= .5
                        forecasts = [f for f in state['forecasts'] if f['id'] == identity]
                        if not forecasts:
                            missing_forecasts += 1
                        for forecast in forecasts:
                            end = t+forecast['horizon_samples']
                            if end >= len(truth):
                                continue
                            endpoint = truth[end][identities.index(identity)]
                            forecast_nll.append(float(-mixture_log_prob(forecast['mixture_centers'],
                                                                       forecast['mixture_weights'], endpoint)))
                            lo, hi = forecast['marginal_interval_90']
                            interval_hits.append(float(((endpoint >= lo) & (endpoint <= hi)).float().mean()))
            retained += sum(identity in by_id and float((by_id[identity]['position']-target).norm()) <= 4
                            for identity, target in zip(identities, truth[-1]))
            extras += sum(min(float((e['position']-target).norm()) for target in truth[-1]) > 4
                          and e['association_strength'] >= .5 for e in state['entities'])
        results[condition] = dict(identities=36, retained=retained, final_extras=extras,
            outage_targets=targets, outage_support_hits=hits, missing_cached_forecasts=missing_forecasts,
            cached_forecasts=len(forecast_nll),
            cached_mean_nll=sum(forecast_nll)/len(forecast_nll) if forecast_nll else None,
            cached_marginal_coverage=sum(interval_hits)/len(interval_hits) if interval_hits else None,
            state_gate=retained >= 35 and extras == 0 and (not targets or hits/targets >= .95))
    # Resize only the sensor workspace, keeping the same learned local parameters.
    compact = type(model)(height=32, width=64, memory=model.state.memory, dynamics=model.dynamics)
    compact.observer.weights.copy_(model.observer.weights)
    compact.observer.bias.copy_(model.observer.bias)
    pairs = [[episode(compact, continued=c, start=s, speed=v, learn=False) for c in (True, False)]
             for s in (3, 9, 15, 21) for v in (1, 2)]
    rows = [r for pair in pairs for r in pair]
    results['ambiguous_prefix'] = dict(pairs=8,
        max_probability_difference=max(abs(a['predicted']-b['predicted']) for a, b in pairs),
        max_mean_difference=max(float((torch.tensor(a['position'])-torch.tensor(b['position'])).abs().max())
                                for a, b in pairs),
        max_mixture_difference=max(float((torch.tensor(a[key])-torch.tensor(b[key])).abs().max())
                                   for a, b in pairs for key in ('mixture_centers', 'mixture_weights')),
        brier=sum((r['predicted']-r['reidentified'])**2 for r in rows)/len(rows),
        continued_reidentified=sum(a['reidentified'] for a, b in pairs),
        vanished_reidentified=sum(b['reidentified'] for a, b in pairs),
        vanished_confident_support=sum(b['counterfactual_support'] >= .5 for a, b in pairs),
        max_vanished_support=max(b['counterfactual_support'] for a, b in pairs),
        calibration_samples=len(compact.calibration[4].outcomes))
    out.write_text(json.dumps(results, indent=2)+'\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
