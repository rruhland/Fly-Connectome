"""Exact reference checks for lower-cost promoted visual-state execution."""

import torch

from fly_connectome.vision.dynamics import mixture_quantile


def test_batched_intervals_match_scalar_search_for_varied_mixtures():
    from fly_connectome.vision.dynamics import batched_mixture_quantiles

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(92731)
    for batch, components in ((1, 1), (3, 5), (24, 32)):
        centers = torch.randn(batch, components, 2, generator=generator)
        weights = torch.softmax(torch.randn(batch, components, generator=generator), 1)
        probabilities = torch.rand(batch, 2, 2, generator=generator)
        probabilities[0, 0] = 0
        probabilities[0, 1] = 1
        expected = torch.stack([
            torch.stack([mixture_quantile(c, w, p) for p in bounds])
            for c, w, bounds in zip(centers, weights, probabilities)
        ])
        actual = batched_mixture_quantiles(centers, weights, probabilities)
        assert torch.equal(actual, expected)
