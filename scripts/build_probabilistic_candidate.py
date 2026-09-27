"""Reproduce the opt-in candidate from observed experience and calibrated ranks."""

import copy
import hashlib
import json
from pathlib import Path

import torch

from probabilistic_visual_state import ProbabilisticVisualState
from spatial_belief import SpatialBelief
from streaming_visual_state import StreamingVisualState


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    source = StreamingVisualState.load('checkpoints/m1a5/streaming-candidate.pt')
    data = torch.load('checkpoints/m1a5/spatial-belief-data.pt', weights_only=True)
    ranks = torch.load('checkpoints/m1a5/spatial-belief-calibration.pt', weights_only=True)
    dynamics = {}
    for h, pairs in data['training'].items():
        model = SpatialBelief(horizon=h)
        for history, target in pairs:
            model.observe(history, target)
        model.calibration.ranks.extend(ranks[h]['learned'])
        dynamics[h] = model
    model = ProbabilisticVisualState(height=64, width=64, observer=copy.deepcopy(source.observer),
                                     memory=copy.deepcopy(source.state.memory), dynamics=dynamics)
    # No reidentification calibration is transferred from another architecture.
    # Empty outcome buffers explicitly expose a .5 prior with zero evidence.
    path = Path('checkpoints/m1a5/probabilistic-visual-candidate.pt')
    model.save(path)
    loaded = ProbabilisticVisualState.load(path)
    for h in dynamics:
        assert torch.equal(model.dynamics[h].values, loaded.dynamics[h].values)
        assert torch.equal(torch.stack(list(model.dynamics[h].calibration.ranks)),
                           torch.stack(list(loaded.dynamics[h].calibration.ranks)))
    report = dict(checkpoint=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        bytes=path.stat().st_size, sensor='every-camera-sample grayscale plus events',
        examples={h: m.seen for h, m in dynamics.items()},
        stored={h: len(m.keys) for h, m in dynamics.items()},
        rank_samples={h: len(m.calibration.ranks) for h, m in dynamics.items()},
        reidentification_calibration='unfitted; .5 prior, zero observed calibration outcomes',
        production_promoted=False)
    Path('docs/experiments/2026-09-26-probabilistic-candidate-manifest.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
