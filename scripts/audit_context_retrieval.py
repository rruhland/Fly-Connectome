"""Inspect local neighbor selection at the failed outage boundary."""
import json
from pathlib import Path
import torch
from fly_connectome.vision import ProbabilisticVisualState
from camera_state_audit import crossing_scene
from run_zero_shot_visual_transfer import corrupt_events


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    model = ProbabilisticVisualState.load('checkpoints/m1a5/context-integration.pt')
    rows = []
    for axis, index in (('horizontal', 0), ('vertical', 6), ('diagonal', 12)):
        images, events, _ = crossing_scene(axis, -3, False)
        events = corrupt_events(events, seed=215000+index, dropout=.15, false_rate=.0005)
        noise = torch.Generator().manual_seed(216000+index)
        model.reset_state()
        for t in range(16):
            image = (.25+.35*images[t]+.02*torch.randn(images[t].shape, generator=noise)).clamp(0, 1)
            model.step(events[t], image)
        m = model.state.memory
        for key in model.state.keys.values():
            variance = m.keys.var(0, unbiased=False).clamp(min=1e-4)
            distance = ((m.keys-key).square()/variance).mean(1)
            nearby = distance.topk(32, largest=False).indices
            pre, post = m.keys[nearby], m.values[nearby]
            centered = pre-pre.mean(0)
            covariance = centered.T @ (post-post.mean(0))/len(nearby)
            metric = covariance.square().sum(1)/(centered.square().mean(0)+1e-4).square()
            metric /= metric.sum().clamp(min=1e-8)
            fitted = ((pre-key).square()*metric).sum(1)
            values, selected = fitted.topk(4, largest=False)
            rows.append(dict(axis=axis, prediction=m.predict(key).tolist(),
                nearest_sensory_indices=nearby[:4].tolist(),
                nearest_sensory_values=post[:4].tolist(),
                selected_indices=nearby[selected].tolist(),
                selected_values=post[selected].tolist(),
                selected_sensory_distances=distance[nearby[selected]].tolist(),
                nearest_sensory_distances=distance[nearby[:4]].tolist()))
    Path('docs/experiments/2026-09-27-context-retrieval-audit.json').write_text(json.dumps(rows, indent=2)+'\n')
    print(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
