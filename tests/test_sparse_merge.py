import torch
from fly_connectome.plasticity import _merge_arrivals


def test_sparse_merge_preserves_old_traces_and_counts_repeated_arrivals():
    keys, values, traces, arrivals = _merge_arrivals(
        torch.tensor([2, 8]), torch.tensor([-.5, .25]), torch.tensor([.2, .4]),
        torch.tensor([8, 1, 8, 11]))
    assert keys.tolist() == [1, 2, 8, 11]
    torch.testing.assert_close(values, torch.tensor([0., -.5, .25, 0.]), rtol=0, atol=0)
    torch.testing.assert_close(traces, torch.tensor([1., .2, 2.4, 1.]), rtol=0, atol=0)
    assert arrivals.tolist() == [1., 0., 2., 1.]


def test_sparse_merge_empty_inputs_preserve_dtype_and_values():
    empty = torch.empty(0, dtype=torch.long)
    for old, new in ((empty, empty), (torch.tensor([7]), empty), (empty, torch.tensor([3]))):
        keys, values, traces, arrivals = _merge_arrivals(
            old, torch.ones(len(old)), torch.ones(len(old)), new)
        assert keys.tolist() == ([7] if len(old) else new.tolist())
        assert values.tolist() == ([1.] if len(old) else [0.]*len(new))
        assert traces.tolist() == [1.]*len(keys)
        assert arrivals.tolist() == ([0.] if len(old) else [1.]*len(new))
