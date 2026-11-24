# Learning when to trust augmented steering after the first return

Holding the opening controller fixed preserved exactly the same first returns while slightly improving later play. Reward-only practice increased later returns on practice, but its frozen memory did not beat the fixed switch on a new continuation challenge. [Paired physical results](joint-trust-results.json)

All conditions kept the same camera-to-joint law until the arm landed its first legal return. This isolates late steering and avoids the opening cost in the ungated native comparison. In the fixed switching control, the coherent native correction then runs on supported incoming observations. The RGB control never switches. The active treatment learns when to use either input law. It does not learn a new stroke tape or use a planner. [RGB feedback](joint-deep.md) · [Coherent candidate](joint-native.md)

The active policy chooses once per later approach from the current six camera-derived position and velocity components and its own prior reward outcomes. It keeps the direct sensor and encoder controller active every tick. A positive label is an actual legal near return. A terminal approach without a near return becomes a negative label only when the ball was visibly close enough to have been attempted. It gets no contact coordinate, target, simulator snapshot, release identifier or ground-truth state. Practice includes scheduled exploration so both steering laws can earn credit. The independent comparison disables exploration and freezes the carried outcome memory. [Controller](../../scripts/experimental/tt37_joint_trust.py)

The first 12 familiar measured openings in the frozen observation receipt each generate two practice reply streams and two independently redrawn challenge streams. Neither deck uses the preceding held reply seeds. The challenge was run only after practice sealed the memory. These are held continuations of familiar openings, not 24 novel launches. [Pinned continuation roster](joint-trust-plan.json)

On practice, both conditions landed 14 opening returns with a best rally of nine alternating legal strokes. Fixed switching landed nine later returns, and active practice landed 11. Active practice retained 25 observed outcomes, including two legal returns with RGB and nine with the coherent correction. The policy treated the remaining choice outcomes as failures to score. [Frozen reward memory](joint-trust-carry.json)

| Controller on 24 independent continuations | Legal first | Legal later | Later offers | Best near / alternating |
| --- | ---: | ---: | ---: | ---: |
| Unchanged RGB | 14 | 9 | 22 | 4 / 9 |
| Fixed switch after first legal return | 14 | 10 | 24 | 5 / 11 |
| Carried reward-trained switch | 14 | 10 | 24 | 5 / 11 |

The first legal returns were identical in contact geometry, landing and timing between the three controllers. The fixed switch beat RGB on three points and lost on two. The reward-trained switch tied fixed switching on all 24 points. Its 23 coherent choices and single RGB choice changed how often the native fit ran but did not change any final score. For this measured deck, the result supports a fixed late correction over the learned switch, not a claim of successful reward-driven transfer. [Challenge receipt](joint-trust-results.json)

The best carried point, measured release 7038 with reply seed 1041153001, reproduced all 11 alternating legal strokes in an action-only replay. The five legal near returns used the negative paddle face. The point ended after an unlanded rim touch. CAD found zero arm penetrations in 4,895 physics steps, with minimum clearance 15.8 mm. [Replay audit](joint-trust-audit.json)

## Reproduce

Use the same launch prerequisites as the [native augmentation](joint-native.md).

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt37_joint_trust plan
for arm in gate active; do
    python -m scripts.experimental.tt37_joint_trust assess /tmp/tt25-rallies.json practice "$arm" "artifacts/tt37/joint-trust-practice-$arm.json.gz"
done
for arm in rgb gate; do
    python -m scripts.experimental.tt37_joint_trust assess /tmp/tt25-rallies.json challenge "$arm" "artifacts/tt37/joint-trust-challenge-$arm.json.gz"
done
python -m scripts.experimental.tt37_joint_trust assess /tmp/tt25-rallies.json challenge carry artifacts/tt37/joint-trust-challenge-carry.json.gz --carry artifacts/tt37/joint-trust-carry.json
python -m scripts.experimental.tt37_joint_trust report
python -m scripts.experimental.tt37_joint_trust audit
```
