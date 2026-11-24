# Online motor decisions from hits and misses

A bounded learner collected experience during 32 physical simulation points, with no paired probes, backpropagation, inverse-kinematic solutions, or predicted interception. It carried 48 measured outcomes into assessment. On 16 points with new opponent continuation seeds, the carried state made **19 legal returns versus 12 with memory reset and 18 with the inherited bias applied to every approach**. On 32 new release IDs, carry and reset both made 11 returns. [Raw online practice](online-practice-acquire.json.gz) · [Retained memory](online-memory.json) · [Score receipt](online-results.json)

The raw stereo and encoder policy still issues each joint command. At the start of an observed approach, the adapter uses the current ball position, velocity and five joint encoders to select one of four small target offsets. The offsets are zero, the previous TT35 reward bias, its scaled inverse, and a small alternative span of the shoulder and wrist commands. No offset exceeds 0.07 rad per joint. During practice, a seeded exploration draw sometimes tries another offset. Only when the point ends does it add evidence from the issued offset to a kernel-weighted local memory. In assessment there are no exploratory choices or updates. The reset uses the identical actor and zero memory. [Controller and outcome bookkeeping](../../scripts/experimental/tt36_online.py) · [The TT35 bias](../tt35/search-checkpoint.json)

A legal return contributes a reward of one. An RGB-observed path already moving away after the near-arm approach earns partial credit, increasing with measured speed even when it does not score. A miss visible in both cameras gets a smaller reward that increases as the ball-to-blade gap narrows. That gap is interpolated after the fact between two actual frames bracketing the ball's passage. Invisible misses with no other evidence do not enter memory. The 32 practice points used 82.3 process CPU seconds and supplied 26 legal-return events, four observed outgoing paths without a legal return, and 18 visible misses. Each reward is attached to the sensory context where the chosen offset began. Neither the gap nor the legal-return pulse enters the per-tick motor policy. [Chronological feedback and command tapes](online-practice-acquire.json.gz) · [Mechanics tests](../../scripts/experimental/tt36_online_test.py)

After eight practice points the memory already chose a nonzero offset for 27 of 32 recorded new-release opening observations. After all 32 practice points, retaining only hit evidence changed eight of those decisions and retaining only miss evidence changed 31. Both signals influenced offset selection. These were decision checks on recorded observations, not additional physical matches or evidence of faster improvement in score. [Learning-dose and ablation counts](online-results.json)

In a separate physical ablation using the same practice experience, legal-only and optical-only grading each reproduced every held point score of the combined learner. Changing motor decisions did not establish that either feedback channel added legal returns beyond the other. [On-court component ablations](online-components.md)

## Matched points

| Course and controller | Opening returns | Later returns | All legal arm returns |
| --- | ---: | ---: | ---: |
| Practice, online learning | 19 | 7 | 26 |
| Practice, frozen with matched exploration draws | 18 | 8 | 26 |
| 32 unseen opening releases, reset | 8 | 3 | 11 |
| 32 unseen opening releases, carried | 9 | 2 | 11 |
| 16 new continuations, reset | 12 | 0 | 12 |
| 16 new continuations, inherited global offset | 16 | 2 | 18 |
| 16 new continuations, carried | 16 | 3 | 19 |

All seven additional returns against reset came from 8239, one of the four familiar openings. The original readout missed that opening on each attempt. Both the learned policy and the inherited global bias returned it under all four new continuation seeds. On the last seed, the carried policy landed four legal arm returns where the global bias landed three. Local learning added one shot over the stronger fixed control. It did not show a general improvement on this course. [Carried reply games](online-replies-carry.json.gz) · [Reset reply games](online-replies-reset.json.gz) · [Inherited global offset](online-replies-prior.json.gz)

On the unseen releases, carry gained openings on 9258, 4805 and 3888. It lost a two-return point on 13509 and a one-return point on 7776. The other 27 scores tied. The reset reproduced the nominal commands and referee events on all 48 held games. Action-only replay of all nine carried games whose scores changed found zero CAD penetrations across 13,829 solver samples and at least 18.6 mm clearance. [Carried unseen games](online-new-carry.json.gz) · [Reset unseen games](online-new-reset.json.gz) · [Replay checks](online-audits.json)

## Reproduction

Run from the repository root. The new-release and varied-reply rosters are the shared TT36 courses. Their fixed command tapes are used only to cross-check the reset, not for learning.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt36_online practice acquire
python -m scripts.experimental.tt36_online practice frozen
for stage in new replies; do
  python -m scripts.experimental.tt36_online "$stage" reset
  python -m scripts.experimental.tt36_online "$stage" carry
done
python -m scripts.experimental.tt36_online replies prior
python -m scripts.experimental.tt36_online summary
python -m scripts.experimental.tt36_online audit
python -m pytest -q scripts/experimental/tt36_online_test.py
```
