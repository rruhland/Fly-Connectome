"""One local-consensus repair experiment; no default promotion here."""
import json
from pathlib import Path
import torch
from fly_connectome.vision import ProbabilisticVisualState
from fly_connectome.vision.memory import LocalMetricAssociation
from run_probabilistic_scenes import main as recovery
from context_retention_audit import score


class ConsensusMemory(LocalMetricAssociation):
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
        outcomes = post[indices]
        agreement = (outcomes.min(0).values >= 0) | (outcomes.max(0).values <= 0)
        return (outcomes*weights[:, None]).sum(0)*agreement


if __name__ == '__main__':
    torch.set_num_threads(1)
    model = ProbabilisticVisualState.load('checkpoints/m1a5/context-integration.pt')
    memory = ConsensusMemory()
    memory.__dict__.update(model.state.memory.__dict__)
    model.state.memory = memory
    prefix = 'docs/experiments/2026-09-27-context-consensus'
    recovery(model=model, out=Path(prefix+'-recovery.json'))
    result = dict(context=score(model, model.state.memory))
    Path(prefix+'-results.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result), flush=True)
