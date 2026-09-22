"""Trace the live learner's slow-tempo sign failure to local edge currents."""
import json
from pathlib import Path

import numpy as np
import torch

from continuous_timing import challenge, record
from timing_transfer import load_model
from fly_connectome.data import checksum


def main():
    torch.set_num_threads(1)
    root = Path('runs/live-local-timing-v1')
    manifest = json.loads((root/'train.json').read_text())
    evaluation = json.loads((root/'evaluation.json').read_text())
    key = manifest['selected_eta']
    trained_path = root/f'train-eta-{key}.npz'
    assert checksum(trained_path) == manifest['artifacts'][str(trained_path)]
    train = np.load(trained_path)
    path = root/'continuous-trained-standard.npz'
    assert checksum(path) == evaluation['artifacts'][str(path)]
    live = np.load(path)
    crop, metadata, _ = load_model()
    frames, times, _ = challenge(False)
    fixed = record(crop, metadata, torch.from_numpy(train['weights_trained']), frames)
    for a,b in [('prediction','raw'),('events','target'),('state','state'),('spikes','spikes')]:
        np.testing.assert_array_equal(fixed[a], live[b])
    np.testing.assert_array_equal(fixed['presynaptic_bodies'], live['presynaptic_bodies'])
    weights = train['weights_trained'][train['incoming_edges']]
    reconstruction = fixed['eligibility']@weights
    error = float(abs(np.clip(reconstruction[32:], -1, 1)-fixed['prediction'][32:]).max())
    assert error < 1e-5
    y = live['target'][1:]
    issue = np.arange(len(y))
    tempo6 = (issue+1 >= times[48]) & (issue+1 < times[72])
    steady = tempo6 & (issue+1 >= times[52]) & ~((issue+1 >= times[60]) & (issue+1 < times[64]))
    groups = {}
    contributions = fixed['eligibility'][:-1]*weights
    for name, mask in [('ON', steady & (y < 0)),('OFF', steady & (y > 0)),
                       ('quiet', steady & (y == 0))]:
        idx = issue[mask]
        sign = y[mask] if name != 'quiet' else np.ones(len(idx))
        groups[name] = dict(count=len(idx),
            mean_edge_contribution=(contributions[idx]*sign[:,None]).mean(0).tolist(),
            mean_raw_signed=float(np.mean(fixed['prediction'][idx]*sign)),
            mean_issued_signed=float(np.mean(live['gated'][idx]*sign)),
            gate_open=int(live['gate'][idx].sum()))
    target = train['target'][1:]; issue_state = train['state'][:-1]
    deltas = train['delta'][1:]
    updates = {name:dict(count=int(mask.sum()),
        issue_positive=int((issue_state[mask]>0).sum()),
        issue_negative=int((issue_state[mask]<0).sum()),
        edge_delta_sum=deltas[mask].sum(0).tolist())
        for name, mask in [('ON',target<0),('OFF',target>0),('quiet',target==0)]}
    result = dict(trained_sha256=checksum(trained_path), live_sha256=checksum(path),
        source_graph=crop['graph'].identity(), reconstruction_max_error=error,
        incoming_bodies=fixed['presynaptic_bodies'].tolist(),
        incoming_weights=weights.tolist(), tempo6_steady=groups,
        training_updates=updates, script_sha256=checksum(Path(__file__)))
    Path('docs/experiments/2026-09-22-live-local-credit-audit.json').write_text(
        json.dumps(result, indent=2, allow_nan=False)+'\n')
    print('tempo6 ON', json.dumps(groups['ON']), flush=True)
    print('tempo6 OFF', json.dumps(groups['OFF']), flush=True)


if __name__ == '__main__':
    main()
