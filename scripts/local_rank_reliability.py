"""Opt-in direct local-error calibration of two visual forecasts."""

import torch


class LocalRankReliability:
    """Shared synapses learn whether each competitive visual site recurs."""

    def __init__(self, *, limit=64):
        self.limit = limit
        self.weight = torch.zeros((2, 2, limit))
        self.count = torch.zeros_like(self.weight)

    def ranked_sites(self, forecast):
        flat = forecast.flatten()
        indices = flat.topk(min(self.limit, flat.numel())).indices
        return indices[flat[indices] > 0]

    @torch.no_grad()
    def observe(self, fast, files, event):
        target = event.flatten() > 0
        spatial_size = event.shape[-2]*event.shape[-1]
        for source, forecast in enumerate((fast, files)):
            for rank, index in enumerate(self.ranked_sites(forecast)):
                polarity = int(index)//spatial_size
                error = float(target[index])-float(
                    self.weight[source, polarity, rank])
                self.count[source, polarity, rank] += 1
                self.weight[source, polarity, rank] += (
                    error/self.count[source, polarity, rank])

    @torch.no_grad()
    def forecast(self, fast, files):
        score = torch.zeros_like(fast)
        flat = score.flatten()
        spatial_size = fast.shape[-2]*fast.shape[-1]
        for source, forecast in enumerate((fast, files)):
            for rank, index in enumerate(self.ranked_sites(forecast)):
                polarity = int(index)//spatial_size
                flat[index] += self.weight[source, polarity, rank]
        return score


@torch.no_grad()
def fit_reliability(episodes, *, shuffled=False):
    model = LocalRankReliability()
    samples = [sample for events in episodes for sample in events]
    shift = max(len(samples)//3, 1) if shuffled else 0
    for index, (event, fast, files) in enumerate(samples):
        credited = samples[(index+shift) % len(samples)][0]
        model.observe(fast, files, credited)
    return model
