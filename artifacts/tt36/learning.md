# What the arm learned from its misses and hits

A simulated SO-101 learned to return three of six openings that its existing direct joint policy failed. On two of those, retaining an early adjustment and learning a second adjustment near the ball was necessary. A separate, more transferable improvement came from learning how the arm moves its paddle during scoreless play. That change increased legal returns from 20 to 26 across 64 unfamiliar openings with a physical opponent. Neither result establishes sustained rallies against arbitrary placement. [Three-stage replay of the same shot](learning-12373.mp4) · [Verification of the three command tapes](learning-12373.json)

## Learning to reach and then learning how to hit

The starting policy was distilled from 30 rewarded teacher strikes. Each subsequent action in these tests was a direct joint command based on the ball observed by calibrated stereo cameras and the measured joint angles. The training programs used five moving joints, estimated blade position from the body map, and used the legal-return signal for task success. The learner received no release ID, physical contact normal, simulator ball state or predicted interception point. The six practice openings were deliberately selected to include three clean misses and three contacts that failed to land. The same first delivery can be repeated during training. This is targeted simulator practice.

The first 11 attempts explored a joint adjustment on approach. The next eight preserved that reach while trying a second adjustment that blended in as the observed ball approached the blade. Two directions for that exploration came from fitting how past motor probes changed the outgoing velocity. This uses a non-backpropagating local regression and a singular-value decomposition in joint space. Legal scores and visual outcomes selected corrections stored against the observed approach. At evaluation, the checkpoints were frozen and the opponent could return the ball continuously. [Early training and reward controls](local.md) · [Outgoing response fit](inverse-prior.json) · [Later training](staged-practice.json.gz) · [Final motor checkpoint](staged-checkpoint.json)

| Physical evaluation | Original | Early adjustment only | Late adjustment only | Both |
| --- | ---: | ---: | ---: | ---: |
| Six practiced first deliveries, two new continuations each | 0 | 2 | 2 | **6** |
| Twenty perturbed first deliveries around those six | 2 | **7** | 6 | **7** |
| Six replacement first deliveries | **4** | 3 | 3 | 3 |

Each number counts all legal returns by the learner. On the first row each pair shares an opening. The six successes represent three distinct openings, and none produced a second legal return. Release 12373 first landed on practice attempt 19, release 3928 on attempt 13, and release 12403 on attempt 5. On 12373 and 3928 either component alone scored zero on both preserved openings and the combination scored on both. The video shows 12373 at real simulation speed and the opponent's initial contact and bounce matched across the three replays. The six sequences of 19 practice games took about 199 process CPU seconds in total, excluding prior calibration. [Practice learning curves and paired outcomes](learning-results.json) · [Independent CAD replay of each changed-score point](learning-audit.json)

The 20 perturbed incoming balls were selected from 24 proposals by whether the opponent delivered a valid physical first ball. The first bounce moved a median 15.7 cm (range 2.6–32.1 cm). Both adjustments together improved four points and worsened none relative to the original, but the early adjustment alone reached the same scores point for point. On the six replacement openings, selecting the late checkpoint solely by legal score preserved four returns. Selecting it with additional optical information scored three. Both selectors used the same explored actions. [Perturbations and selection](nearby-plan.json) · [Evaluation tapes](nearby-both.json.gz)

## Which observations transferred?

- **Body motion measured during misses.** Twenty scoreless games supplied a local joint-to-blade map. Relative ball and blade positions from rewarded demonstration frames specified examples of useful alignment. Recomputing the joint adjustment from the current camera state raised first contacts from 35 to 41 across two disjoint 32-opening courses. Legal returns increased from 11 to 14 on the first and from 9 to 12 on the second. Later returns increased from four to seven overall. On the first course, reversing the computed movement dropped first contacts from 19 to 12 and legal returns from 11 to 9. The newly reached balls still went out or hit the net. The legal gains came from changing the hit on balls the arm could already touch. [Body feedback and verification](reach.md)
- **Outgoing speed and direction.** A forecast trained on 47 camera-measured hits reduced held outgoing-velocity vector RMS error from 0.990 to 0.843 m/s when it also saw joint angles, joint movement and recent commands. Median directional error went from 8.97° to 5.74°. Only six held hits had readable stereo labels. Separately, a predictor of legal return from observed outgoing velocity improved leave-one-release-out Brier error from 0.262 to 0.220. Its motor selections tied score-only training on both tested courses. Predicting an outcome is easier than finding the joint change that produces it. [Prediction and sensory shaping assumptions](hit.md) · [Learned value of outgoing flight](effect-checkpoint.json)
- **Learning during continuous games.** A contextual motor memory updated during 32 physical games. It earned 19 returns on four familiar openings with four fresh continuations each, versus 12 after resetting and 18 with the inherited fixed adjustment. On 32 new releases it tied reset at 11. Retraining the selection from only camera outcomes or only legal rewards on identical experience produced identical held scores. This experiment gives no demonstrated advantage to combining the feedback signals in a single scalar reward. [Matched feedback controls](online-components.md)
- **Failed shortcuts.** A single global offset learned from the signed error on scoreless approaches reduced 11 returns to six on the first 32. Summing it with the hit adjustment scored nine. Selecting adjustments by practiced visual context scored 13, then tied the original nine on the second cohort. An experiment using the following return to select a different recovery movement found no training context where a tested change beat the original. [Miss analysis](miss.md) · [Matched alternatives](paired-results.json) · [Recovery experiment](recovery-checkpoint.json)
- **Weak touches.** Many faults produced no forward reversal. A camera detector using change of velocity at 60 Hz recognized 11 of 14 held contacts, including four of six nonforward deflections. The earlier 30 Hz reversal detector recognized seven of 14 and zero nonforward deflections. Both produced zero unmatched detections on this sample. Applying that feedback to two failed openings restored four out of four held first contacts, versus zero for the previous checkpoint. Eight more attempts per opening yielded zero legal balls. On one, vertical velocity changed from −0.34 to +0.53 m/s and the first bounce moved 27.6 cm closer to the net, still 49.2 cm short. [Observer and separate physical audit](impact.md) · [Subsequent motor learning](impact-refine.md)

## Combining the changes and checking retention

| Direct controller | New 32 | Independent 32 | Four familiar openings × four new reply seeds |
| --- | ---: | ---: | ---: |
| Original | 11 | 9 | 12 |
| Learned reach | **14** | **12** | 13 |
| Reach plus outgoing adjustment | 13 | 10 | 17 |
| Choose the outgoing adjustment by practice context | **14** | 10 | 17 |
| Limit outgoing adjustment to preserve learned blade position | 13 | 11 | 13 |
| Same calculation with scrambled joint-to-position axes | 10 | 8 | **24** |

The last two conditions test whether changing strike motion while preserving instantaneous paddle-center position helps. The projection through the empirical body map helped relative to scrambling on both unfamiliar courses. The scrambled control won on familiar openings, including a six-return point, and fell below the original on both unfamiliar courses. No combined controller improved on learned reach on both unfamiliar courses. On the earlier eight opponent continuations of release 5530, reach still returned all eight openings. It improved a two-return point to five, shortened the previous seven-return point to two, and tied the other six. Total legal arm returns fell from 16 to 14. Keep both checkpoints. [All per-point comparisons](learning-results.json) · [Earlier baseline](../tt35/replies-nominal.json.gz) · [Retention run](reach-retained-aligned.json.gz) · [Task-priority control reference](https://doi.org/10.1177/027836498700600201)

We isolated when the retention loss occurred by rerunning the two changed points and swapping controllers exactly as the opponent made its third stroke. The commands and incoming ball up to that switch matched the source replay. On the shortened point, starting with reach and then returning to the original policy restored all seven returns. Starting with the original policy and then enabling reach yielded three. On the improved point, starting with reach was enough to preserve all five returns after switching back, while enabling reach later produced three. The early movement made a useful difference on one trajectory and continuing to apply it later cost returns on the other. The switch time came from the evaluator and was used only to diagnose credit assignment. [Matched crossover tapes](retention-switch.json.gz)

## Reproduce

Use the measured launch file from the [regulation reproduction instructions](../tt29/endurance.md). Component reports above include their individual calibration and reproduction commands. The following uses the committed practice inputs and checkpoints. Rebuild both optical directions with `tt36_inverse fit`, or the local late checkpoint with `tt36_staged practice`. The grip and two-camera calibration are shared across these courses.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
feeds=/tmp/tt25-rallies.json
for mode in dense score late-only; do python -m scripts.experimental.tt36_staged "$mode" "$feeds"; done
for mode in fixed early late-only both; do python -m scripts.experimental.tt36_nearby "$mode" "$feeds"; done
python -m scripts.experimental.tt36_synthesis report
python -m scripts.experimental.tt36_synthesis audit
python -m scripts.experimental.tt36_learning_replay
```
