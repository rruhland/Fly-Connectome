"""Context-dependent covariance from the same bounded observed experience."""

import torch

from associative_patch_state import PatchAssociation


class LocalMetricAssociation(PatchAssociation):
    @torch.no_grad()
    def predict(self, key):
        if not len(self.keys):
            return torch.zeros(self.values.shape[1])
        variance = self.keys.var(0, unbiased=False).clamp(min=1e-4)
        sensory_distance = ((self.keys-key).square()/variance).mean(1)
        nearby = sensory_distance.topk(min(32, len(self.keys)), largest=False).indices
        pre, post = self.keys[nearby], self.values[nearby]
        centered = pre-pre.mean(0)
        covariance = centered.T @ (post-post.mean(0))/len(nearby)
        metric = covariance.square().sum(1)/(centered.square().mean(0)+1e-4).square()
        if metric.sum() > 1e-8:
            metric /= metric.sum()
            distance = ((pre-key).square()*metric).sum(1)
        else:
            distance = sensory_distance[nearby]
        values, indices = distance.topk(min(4, len(distance)), largest=False)
        weights = torch.softmax(-values/.1, 0)
        return (post[indices]*weights[:, None]).sum(0)
