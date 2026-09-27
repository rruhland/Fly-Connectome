"""Regenerate dynamics/calibration after reviewed candidate integration fixes."""

import copy
import json
import shutil
from pathlib import Path

import torch

from run_local_motion_dynamics import KINDS, visual_episode
from streaming_visual_state import StreamingVisualState, LearnedVisualCandidate


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    path = Path('checkpoints/m1a5/streaming-candidate.pt')
    data = torch.load(path, weights_only=True)
    if data['version'] == 1:
        assert data['state_type'] == 'LearnedVisualCandidate'
        shutil.copy2(path, path.with_name('provisional-v1-candidate.pt'))
        data.update(version=2, memory_type='LocalMetricAssociation',
                    dynamics_types={h: 'ContextualMotionDynamics' for h in data['dynamics']})
        torch.save(data, path)
    model = StreamingVisualState.load(path)
    trained = LearnedVisualCandidate(height=64, width=64, observer=copy.deepcopy(model.observer))
    for kind in KINDS:
        for phase in (0, 2):
            for shape in ('dot', 'square'):
                case, _ = visual_episode(kind, phase, shape, background=bool(phase))
                trained.reset_state()
                for event, image in zip(case['events'], case['visible']):
                    trained.step(event, image, learn=True)
    same = all(torch.equal(model.dynamics[h].keys, trained.dynamics[h].keys) and
               torch.equal(model.dynamics[h].values, trained.dynamics[h].values)
               for h in model.dynamics)
    model.dynamics = trained.dynamics
    model.calibration = copy.deepcopy(trained.calibration)
    model.reset_state()
    model.save(path)
    loaded = StreamingVisualState.load(path)
    assert all(loaded.calibration[h].summary() == model.calibration[h].summary() for h in model.calibration)
    result = dict(dynamics_unchanged_by_raw_evidence_fix=same,
                   calibration={h: c.summary() for h, c in model.calibration.items()},
                   checkpoint_version=2, roundtrip_verified=True)
    Path('docs/experiments/2026-09-26-candidate-checkpoint-review-fixes.json').write_text(
        json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
