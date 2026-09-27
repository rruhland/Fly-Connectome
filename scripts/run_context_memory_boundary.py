"""Single boundary control after the integrated outage-support regression."""
import json
from pathlib import Path

import torch

from fly_connectome.vision import load_legacy_default, ProbabilisticVisualState
from context_retention_audit import score
from run_probabilistic_scenes import main as recovery


if __name__ == '__main__':
    torch.set_num_threads(1)
    model = ProbabilisticVisualState.load('checkpoints/m1a5/context-integration.pt')
    # Retain all trained dynamics and their calibration. Replace only context
    # associations with the original prior, for diagnosis, never for promotion.
    model.state.memory = load_legacy_default().state.memory
    prefix = 'docs/experiments/2026-09-27-context-memory-boundary'
    recovery(model=model, out=Path(prefix+'-recovery.json'))
    Path(prefix+'-results.json').write_text(json.dumps(dict(
        intervention='restore only original context memory; keep trained dynamics',
        context=score(model, model.state.memory)), indent=2)+'\n')
