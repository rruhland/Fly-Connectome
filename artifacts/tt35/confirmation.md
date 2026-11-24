# Direct joint learning from scoreless experience

The clearest result is adaptation to delayed vision. A separate result on timely cameras shows that revisiting the states the arm encountered during misses can lengthen a rally. Varying the opponent's subsequent shots exposed how much of that motor gain depends on the trajectory.

## Camera adaptation repeated on new releases

Using the previously fitted observation models and the camera age learned during a single missed point, the raw pupil increased legal arm returns from **2 to 8 across 24 further releases**, including three later replies. Four point scores improved, 20 tied. The timely-camera baseline made seven. For these games the actor carried its estimated age and the regression for reconstructing present motion. No joint policy parameters changed. This confirms the earlier increase from one to five on a disjoint 16. Together these are 40 distinct held release IDs under a 100 ms image delay, with 3 legal arm returns for stale timing and 13 after adjustment. The separate eight-shot switching course tested continued adaptation during the games. [Paired confirmation](confirmation-results.json) · [Physical replays](confirm-lag-learned-audit.json) · [How timing was learned](../tt34/zero-reward.md)

The 24-release course used nine left, six middle and nine right first deliveries. Its roster excluded the preceding practice and test releases and used the neutral arm to screen for physical delivery. Every paired score checks the same first far contact and table bounce to absolute tolerance 1e−8. We count legal arm returns and exclude the synthetic opening feed. These cohorts cover reachable, screened regulation feeds. [Roster and exclusion list](confirmation-plan.json.gz)

## Learning joint commands on timely cameras

The original direct joint pupil played 14 practice points. We used only examples from points where it made zero legal returns. In five points the two RGB views both saw the ball pass within 3–22 cm of the paddle center at the paddle's depth. These five supplied 114 control ticks from up to 380 ms before the observed passage. The distance filter and window are training scaffolds. During play the pupil receives present optical position, velocity and encoder angles. Its inputs omit the gap label, contact time and future ball position. All refits use regularized least squares on these examples and the original 1,080 teacher samples. [Training points and optical labels](hindsight-practice.json) · [Implementation](../../scripts/experimental/tt35_hindsight.py)

| Training target or control | Earlier 8 points | Next 16 points | New 24 points |
|---|---:|---:|---:|
| Original direct joint pupil | 3 | 6 | 7 |
| Shift the observed ball toward the measured paddle passage, then imitate the issued command | 5 | 6 | 8 |
| Same issued commands with unshifted observations | 5 | 7 | 7 |
| Reverse the optical shift | 5 | 7 | — |
| Refit to the old policy's instantaneous target on the newly visited states | 6 | 6 | 11 |
| Same refit on previously demonstrated states, matched sample count and time weights | 4 | 6 | 7 |
| Adjust five shared joint biases by legal-return reward in physical practice | 4 | 6 | 9 |

The most effective refit queried the old policy at the visited states and recomputed the basis centers and output weights. Its targets came from its own existing readout before actuator feedback and slew limits. This gives the policy extra examples of where it has actually been. It supplies no new demonstration of a correct strike. Sampling the same number of familiar states did not reproduce the gain on the 24 new releases. The experiment does not establish that failures are unique as a source of useful states, since no equally novel successful-state bank was tested. Shifting the ball to turn a miss into a hypothetical hit did not reliably outperform its controls. [Model from visited states](hindsight-static.json.gz) · [Familiar-state control](hindsight-familiar.json.gz)

On the new course, the refit's 11 returns comprised six opening and five later returns versus six opening and one later for the original. Most of the difference came from release 5530, where it landed five consecutive replies and the original landed one. A different opening was gained and another lost. The reward-trained adjustment also lengthened 5530, to four arm returns after 48 practice episodes costing 88 process CPU seconds. The motor studies reuse the same release rosters and give no evidence yet of a broadly higher first-return rate. [Watch the five-reply arm](return-5530.mp4) · [CAD audit](confirm-static-audit.json) · [Reward-guided search](search-confirmation.md)

## Did the longer rally survive different subsequent shots?

We kept the entire first physical offer on 5530 and gave the opponent eight new continuations drawn from the same continuous placement distribution. We chose this opening after seeing its five-reply rally. The original pupil made 16 total legal arm returns, the gap-shifted refit made 15 and the visited-state refit made 13. All returned the first offer eight times. On the last continuation both the original and gap-shifted pupil made seven consecutive returns each in their respective real-time, 15-return games. Both opponents and arms contacted the physical ball for every counted return. This shows what direct joint control can execute. The changed replies failed to confirm an improved rally strategy. [Watch a 15-return direct-control point](varied-5530.mp4) · [Command replay](varied-5530.json) · [Matched outcomes](confirmation-results.json)

A separate correction used the sign of the optical miss to choose joint adjustments. It narrowed the independently measured ball-to-blade gap by about 3 mm on seven comparable held passes but scored six legal returns both before and after. Reward-free sensory learning is now useful for camera changes. For general rally learning, the next target is to distinguish an action that can reach the ball from an action that sends it back playably, and to verify the difference against freshly randomized continuations. [Optical miss study](error.md)

## Reproduction

Use the pinned launch data and raw teacher sample capture from the [baseline reproduction](../tt33/joint.md), its stereo calibrations and the archived course plans. The ordinary fit and the timing tests below have separate inputs. The `static` name denotes the old policy's unconstrained instantaneous targets.

```sh
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1
feeds=/tmp/tt25-rallies.json
out=artifacts/tt35
h=scripts.experimental.tt35_hindsight
j=scripts.experimental.tt33_joint
python -m "$h" collect /tmp/practice.npz --feeds "$feeds"
for mode in aligned as-observed reverse static familiar; do
  python -m "$h" "$mode" "/tmp/$mode.json.gz" --raw /tmp/tt33-raw.npz --source /tmp/practice.npz
  python -m "$j" evaluate "$feeds" "$out/confirmation-plan.json.gz" "/tmp/held-$mode.json.gz" --checkpoint "/tmp/$mode.json.gz"
done
for mode in nominal aligned static; do python -m scripts.experimental.tt35_replies "$mode"; done
python -m scripts.experimental.tt35_replies film
python -m scripts.experimental.tt35_confirm
```

For the camera confirmation use the [timing commands](../tt34/zero-reward.md) with `tt35/confirmation-plan.json.gz` as the course. Each archived point contains the per-tick commands and physical referee events.
