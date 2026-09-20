"""Verify the optional native merge against hand-derived and randomized cases."""
import argparse
import torch
from native_sparse_merge import load_merge
from fly_connectome.plasticity import _merge_arrivals

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('library')
args = parser.parse_args()
merge = load_merge(args.library)
for seed in range(20):
    generator = torch.Generator().manual_seed(seed)
    old = torch.randperm(100,generator=generator)[:seed*3].sort().values
    incoming = torch.randint(0,100,(seed*7,),generator=generator)
    values = torch.randn(len(old),generator=generator)
    traces = torch.rand(len(old),generator=generator)
    before = incoming.clone()
    actual = merge(old,values,traces,incoming)
    expected = _merge_arrivals(old,values,traces,incoming)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert torch.equal(incoming,before)
keys,values,traces,arrivals = merge(torch.tensor([2,8]),torch.tensor([-.5,.25]),
                                  torch.tensor([.2,.4]),torch.tensor([8,1,8,11]))
assert keys.tolist()==[1,2,8,11]
assert values.tolist()==[0.,-.5,.25,0.]
assert arrivals.tolist()==[1.,0.,2.,1.]
print('Native merge: hand-derived case and 20 randomized exact comparisons passed')


# Reject unsafe pointer reinterpretations before calling the native function.
for inputs in (
    (torch.tensor([1],dtype=torch.int32),torch.ones(1),torch.ones(1),torch.tensor([1])),
    (torch.tensor([1]),torch.ones(1,dtype=torch.float64),torch.ones(1),torch.tensor([1])),
    (torch.tensor([1]),torch.ones(2),torch.ones(1),torch.tensor([1])),
    (torch.tensor([1]),torch.ones(1,requires_grad=True),torch.ones(1),torch.tensor([1])),
    (torch.arange(4)[::2],torch.ones(2),torch.ones(2),torch.tensor([1])),
):
    try:
        merge(*inputs)
    except ValueError:
        pass
    else:
        raise AssertionError('unsafe input accepted')
print('Native merge input guards passed')
