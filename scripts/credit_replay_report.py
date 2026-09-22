"""Verify and summarize saved passive credit records; no neural simulation."""
import json
from pathlib import Path

import numpy as np

from credit_replay import CreditRecorder, summarize
from fly_connectome.data import checksum


def main():
    root = Path('runs/credit-replay-v1')
    results = json.loads((root/'results.json').read_text())
    assert len(results) == 6
    output = {}
    for name, metadata in results.items():
        path = root/f'{name}.npz'
        assert checksum(path) == metadata['sha256']
        assert checksum(Path('scripts/credit_replay.py')) == metadata['runner_sha256']
        source = Path('runs/multitempo-v1/mixed-training.npz' if name.startswith('baseline')
                      else 'runs/short-term-v1/F-training.npz')
        assert checksum(source) == metadata['source_sha256']
        a = np.load(path)
        assert np.all(a['confirm_tick']-a['issue_tick'] == 8)
        assert np.all(a['accumulator_error'] == 0)
        for key in ['issue_state', 'confirm_state', 'eligibility', 'prediction', 'target', 'delta', 'weights_trained']:
            assert np.isfinite(a[key]).all()
        if 'frame_slot' in a:
            slot = a['frame_slot'][a['confirm_tick']//8]
            recurrent = (slot < 0) | (slot >= 2)
        else:
            old = np.load(source)
            blanks, dwells = old['blank_frames'][:24], old['dwells'][:24]
            lengths = blanks+8*dwells+1
            starts = np.r_[0, np.cumsum(lengths)[:-1]]
            frame = a['confirm_tick']//8
            trial = a['frame_trial'][frame]
            relative = frame-starts[trial]
            recurrent = (relative < blanks[trial]) | (relative >= blanks[trial]+2*dwells[trial])
        last = np.zeros(len(a['all_target']))
        value = 0.
        for i, y in enumerate(a['all_target'][:, 0]):
            if y != 0:
                value = y
            last[i] = value
        tag = np.sign(a['issue_state'][:, 0, 0])
        matches = tag == last[a['issue_tick']]
        assert matches.all()
        category_sums = {}
        for e, pre in enumerate([20655, 26550]):
            q, y = a['delta'][:, e].astype(float), a['target'][:, e]
            category_sums[str(pre)] = {str(s): {
                label: float(q[(tag == s) & mask].sum())
                for label, mask in [('ON', y < 0), ('OFF', y > 0), ('quiet', y == 0)]}
                for s in [-1, 0, 1]}
        output[name] = dict(last_polarity_matches=int(matches.sum()), issue_count=len(matches),
            recurrent=summarize(a, {'recurrent': recurrent}), context_update_sums=category_sums,
            confirmation_tag_change={label: dict(count=int(mask.sum()),
                changes=int((mask & (np.sign(a['confirm_state'][:, 0, 0]) != tag)).sum()))
                for label, mask in [('ON', a['target'][:, 0] < 0), ('OFF', a['target'][:, 0] > 0),
                                    ('quiet', a['target'][:, 0] == 0)]})
    pairs = {}
    for name in ['baseline', 'F']:
        ordinary = np.load(root/f'{name}-standard.npz')
        omitted = np.load(root/f'{name}-omitted.npz')
        frame = int(np.flatnonzero(ordinary['frame_slot'] == 2)[0])
        row = int(np.flatnonzero(ordinary['confirm_tick'] == frame*8)[0])
        for field in ['issue_state', 'eligibility', 'prediction', 'issue_weights']:
            np.testing.assert_array_equal(ordinary[field][row], omitted[field][row])
        for field in ['all_spikes', 'all_prediction', 'all_target']:
            np.testing.assert_array_equal(ordinary[field][:frame*8], omitted[field][:frame*8])
        pairs[name] = dict(issue_tick=int(ordinary['issue_tick'][row]), confirm_tick=frame*8,
            identical_prefix=True, identical_recorded_issue_state=True,
            prediction=ordinary['prediction'][row].tolist(),
            standard_target=ordinary['target'][row].tolist(), omitted_target=omitted['target'][row].tolist(),
            standard_delta=ordinary['delta'][row].tolist(), omitted_delta=omitted['delta'][row].tolist())
    published = dict(runs=results, analysis=output, first_omission_pairs=pairs,
        repeat_unavailable=json.loads((root/'repeat-unavailable.json').read_text()),
        state_columns=CreditRecorder.state_names+['presynaptic_voltage', 'presynaptic_adaptation'])
    path = Path('docs/experiments/2026-09-21-credit-replay-results.json')
    path.write_text(json.dumps(published, indent=2, allow_nan=False)+'\n')
    print('Verified six artifacts, sources, recorder identity, causal timing, finite states, accumulators and paired prefixes.')
    for name, row in output.items():
        print(name, 'last-polarity agreement', row['last_polarity_matches'], '/', row['issue_count'])
        for edge, groups in row['recurrent'].items():
            print(edge, {k: (v['correct'], v['count']) for k, v in groups['recurrent'].items()})


if __name__ == '__main__':
    main()
