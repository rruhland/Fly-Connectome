"""Zero-learning full-graph equality of gated and always-open Pong dynamics."""
from dataclasses import replace
import json
from pathlib import Path
import time

import torch

from full_context_efficacy import (AlwaysOpenContextPrediction, MultiContextEfficacyNetwork,
                                   MultiContextTimedPrediction)
from fly_connectome.data import checksum
from fly_connectome.dynamics import NeuronConfig
from fly_connectome.graph import Graph
from fly_connectome.plasticity import LearningConfig
from fly_connectome.pong import Physics, Pong
from fly_connectome.sensor import EventCamera, Retina


def equal(left, right, name, frame, tick):
    if not torch.equal(left, right):
        raise AssertionError(f'{name} differs at frame {frame}, tick {tick}')


def main():
    torch.set_num_threads(4)
    source = Path('checkpoints/event-v1-combined-rate-initial.pt')
    source_sha = checksum(source)
    payload = torch.load(source, weights_only=True)
    metadata = payload['metadata']
    graph = Graph(**metadata['graph'], gain=metadata['gain'])
    retina = Retina(**metadata['retina'])
    kwargs = dict(batch=1, config=NeuronConfig(**metadata['neurons']),
                  cell_types=metadata['retina']['cell_types'],
                  target_mask=retina.injected)
    gated = MultiContextEfficacyNetwork(graph, metadata['delays'], metadata['pathways'], **kwargs)
    open_net = MultiContextEfficacyNetwork(graph, metadata['delays'], metadata['pathways'], **kwargs)
    weights = payload['state']['network']['magnitudes'].clone().clamp_(
        0, metadata['learning']['maximum_weight'])
    gated.set_weights(weights)
    open_net.set_weights(weights)
    config = replace(LearningConfig(**metadata['learning']), eta_prediction=0.,
                     eta_reward=0., homeostasis_rate=0.)
    rule_kwargs = dict(sensory_mask=retina.injected,
                       sensory_gain=metadata['config']['sensory_gain'])
    gated_rule = MultiContextTimedPrediction(gated, config, **rule_kwargs)
    open_rule = AlwaysOpenContextPrediction(open_net, config, **rule_kwargs)
    camera = EventCamera(1, retina.spec['height'], retina.spec['width'])
    pong = Pong([1101], Physics(**metadata['physics']))
    frames = 50
    closed_eligible = 0
    started = time.perf_counter()
    for frame in range(frames):
        image = pong.render(retina.spec['height'], retina.spec['width'])
        injection = retina.project(camera.observe(image))*metadata['config']['sensory_gain']
        for tick in range(8):
            sensory = injection if tick == 0 else torch.zeros_like(injection)
            old = gated.step(sensory, capture_increments=True)
            new = open_net.step(sensory, capture_increments=True)
            for name in ('spikes', 'observed', 'predicted', 'arrival_edges',
                         'arrival_environments', 'feedforward_arrivals'):
                equal(getattr(old, name), getattr(new, name), name, frame, tick)
            for name in ('voltage', 'sensory_state', 'predictive_current',
                         'excitatory_prediction', 'inhibitory_prediction',
                         'context_exc', 'context_inh', 'adaptation', 'current_context'):
                equal(getattr(gated, name), getattr(open_net, name), name, frame, tick)
            gated_rule.observe(old, torch.zeros(1))
            open_rule.observe(new, torch.zeros(1))
            for name in ('keys', 'values', 'arrival_trace', 'post_trace',
                         'expected', 'rates', 'reference', 'previous_state',
                         'gate', 'quiet_count', 'event_count'):
                equal(getattr(gated_rule, name), getattr(open_rule, name),
                      name, frame, tick)
            if tick == 0:
                old_issue, new_issue = gated_rule.last_issue, open_rule.last_issue
                for name in ('context', 'gate', 'sensory_state', 'raw_prediction',
                             'edges'):
                    equal(old_issue[name], new_issue[name], name, frame, tick)
                equal(new_issue['prediction'], new_issue['raw_prediction'],
                      'always-open expression', frame, tick)
                equal(old_issue['prediction'],
                      new_issue['prediction']*old_issue['gate'],
                      'gated expression', frame, tick)
                equal(gated_rule.forecast[0], open_rule.forecast[0],
                      'forecast edge keys', frame, tick)
                positions = gated.target_lookup[gated.post[old_issue['edges']]]
                gate = old_issue['gate'][positions].float()
                equal(gated_rule.forecast[1], open_rule.forecast[1]*gate,
                      'gated eligibility', frame, tick)
                equal(gated_rule.forecast[2], open_rule.forecast[2]*gate,
                      'gated forecast', frame, tick)
                closed_eligible += int(((gate == 0) &
                                        (open_rule.forecast[1] != 0)).sum())
        gated_rule.synchronize()
        open_rule.synchronize()
        equal(gated.components, open_net.components, 'context magnitudes', frame, 7)
        pong.step(-(pong.ball[:, 1]-pong.body.position)*10)
    if closed_eligible == 0:
        raise AssertionError('no closed timing windows had local eligibility')
    if checksum(source) != source_sha:
        raise AssertionError('source checkpoint changed')
    result = dict(source_sha256=source_sha, frames=frames, ticks=8*frames,
                  targets=len(gated.targets), selected_edges=len(gated.incoming),
                  closed_eligible_edge_issues=closed_eligible,
                  physical_and_local_state_exact=True, weights_exact=True,
                  only_forecast_and_eligibility_gate_removed=True,
                  seconds=time.perf_counter()-started,
                  script_sha256=checksum(Path(__file__)))
    path = Path('runs/full-context-gate-ablation-v1/parity.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
