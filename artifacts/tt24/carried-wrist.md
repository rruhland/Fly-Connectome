# Carried wrist controller

The [single saved checkpoint](carried-wrist.npz) contains the five-joint feedback weights learned from rewards on three hinged donor arms, a source-derived familiar joint configuration, the degree of reference reuse selected in native Pong, and two wrist-camera calibrations. No recorded motor trajectories are stored. The mounted gripper holds the paddle. Each frame produces fresh rate commands for pan, shoulder, elbow, wrist pitch and wrist roll. [Inference code](../../scripts/experimental/tt24_carried_wrist.py)

From the repository root, an application with the same public 192 × 144 RGB and five-encoder interface can use:

```python
from scripts.experimental.tt24_carried_wrist import CarriedWristPolicy

policy = CarriedWristPolicy()
policy.begin_release(observation)
while playing:
    joint_rates = policy.act(observation)
    observation = next_observation(joint_rates)
```

The application supplies the loop, observations and release boundaries. The five normalized rate commands use the simulated SO-101 servo interface. Loading the checkpoint requires the repository's table-tennis Python environment. Calling `begin_release` starts a new optical history, retrieves the matching calibration from the current wrist image, and preserves every learned parameter. This inference checkpoint has frozen motor weights. Its runtime adaptation consists of visual tracking and recalling earlier body calibration. The separate online research runs report the costs of learning new settings.

On 32 mixed stock and changed-arm drops, loading this bundle once and using only the public observations reproduced **all 8,011 individual five-joint decisions** and each final rebound score exactly. On this CPU the median vision-plus-motor time was 1.43 ms per frame, the 95th percentile 2.05 ms, and the 99th percentile 3.42 ms. These timings exclude simulation and rendering. The control interval was 16.67 ms. The body-specific camera calibrations used eight earlier unlabelled stock probes and four altered-body probes with two reference cameras. Every frame in the reported play used the wrist camera alone with its encoders. [Physical rollout audit](carried-wrist-audit.json) · [Mixed-body results](ready.md#switching-physical-arms) · [Native coordinate-learning study](reference-transfer.md)
