"""Verify continued acquisition and retention with the consensus context rule."""
from pathlib import Path
import json
import torch
from fly_connectome.vision import load_legacy_default
from fly_connectome.vision.memory import ConsensusLocalMetricAssociation
from context_retention_audit import main


def fresh():
    model = load_legacy_default().upgrade_temporal_context()
    model.state.memory = ConsensusLocalMetricAssociation()
    return model


if __name__ == '__main__':
    torch.set_num_threads(1)
    out = Path('docs/experiments/2026-09-27-consensus-context-learning.json')
    main(factory=fresh, out=out)
    result = json.loads(out.read_text())
    final = result['stages'][-1]['retained']
    if (not all(row['hits'] == row['total'] and row['hits'] > result['before'][name]['hits']
                for name, row in final.items()) or
            not all(stage['keys_changed'] for stage in result['stages'])):
        raise SystemExit('Fresh context acquisition/retention failed.')
