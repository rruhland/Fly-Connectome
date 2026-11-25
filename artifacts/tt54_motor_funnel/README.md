# Is the motor controller throwing away recoverable movements?

The direct arm controller forecasts 21 moving ticks at 60 Hz, with additional constant-reference time to judge the ensuing ball. Its stock short screen promotes approximately eight of 28 candidate paths. This audit replays actual photographed, encoder-controlled broad-random points and asks whether the discarded paths had any modeled legal nominations. It also tests whether a small set of candidates improves when motor perturbations are generated around the *issued* path instead of its often unreachable desired references.

In the independent 32-game TT48 course with the 63-return point, 207 returns scored and 30 measured bounced shots did not. The arm touched 19 of those 30. Late searches showed at least one legal native prediction on all 207 scores but only eight misses. Of 30 misses, four ever got a winning robust-voted forecast. The median last-search gap was three ticks on both scores and misses, measured to the actual contact or the available point endpoint. Good and failing observed centers separated much more strongly than the issuance gap. Median best virtual center-to-ball distance was 25 mm for scores, 69 mm for misses that touched, and 96 mm for misses that did not touch.

The misses were heavily concentrated on the same low receiving band. At a projected height below 1.01 m on the deep line, 25 of 52 offers scored. At 1.01–1.03 m, 36 of 39 scored. All 146 at or above 1.03 m scored in this course. These heights characterize actual deliveries, not inputs to the audit controller or thresholds for a new actor.

Python's reference conversion and the native forecast agree on the signed factor of .8, the absolute joint limits, and the same eight physical integration steps per action. No point exceeded the rated 5.45 rad/s measured shaft limit. During the final up-to-12 ticks after bounce, flex at the wrist used the command slew ceiling on 58% of scored windows and 39% of missed windows. Shoulder lift did so on fewer than 2% in both. These endpoint-based windows differ for a no-contact miss. This is observational and is not a reason to relax the rating. [Issued rates and outcomes](actual-rates.json)

The command pipeline does permit quick reversals. At least one arm axis issued adjacent opposing commands in 203 of 207 scored windows, with a magnitude product above .005. The question is whether differently timed, reachable motor candidates are ever generated for difficult arrivals, not whether the interface supports them at all.

## Audit design

- The first eight eligible distinct archived official releases were selected in their original order. Every case includes the last actual failed bounced offer, with its prior scored offer as a control where available. Archived commands, events, actual placements and traced search notes must match exactly again. Additional model trials are diagnostically separate and never influence the replayed actor. [Fixed native funnel roster](protocol.json)
- The later four releases also serve as a small paired check of executable motor centers. Stock generates the same desired seed paths at the same camera-derived state, using exactly the same initial memory and RNG. The alternate first applies the existing signed motor and absolute position saturation to each seed, then generates its mutations around the resulting feasible path. No extra proposal rows are available to the alternate. At each otherwise empty failing search, the stock actor still issues only the original command. If an alternate receives both independent native robustness votes, the earliest per case is checked as a separate physical fork with the same ball and unchanged physical opponent. [Fixed recentering roster](recenter-protocol.json)

This is a motor diagnostic, not a new held gameplay claim. The elbow preparation, older motion memory, outbound choice and perception remain unchanged.

## What survived the exact audit

All eight archived games reproduced every submitted action, placement, physics event and search note. On 16 searches before real scored controls, full-length native trials found 167 additional legal paths among 298 paths eliminated by the usual shortlist. Of those, 163 passed at least one perturbation. On 25 searches before real misses, the same exhaustive check found only one legal path among 609 omissions. That one received neither perturbation vote and occurred at a tick that already had eight native legal options. None of the 23 originally empty failed searches acquired a legal option even when every previously screened stock proposal was evaluated. This rules out shortage of finalist slots as the main bottleneck in this sample.

The median worst per-path difference between the desired reference and the signed, executable one was .142 rad for missed offers. The 90th percentile was .324 rad. Despite this, feasible-reference recentering yielded zero legal proposals across all 13 preassigned empty searches on four different failed offers. There was no candidate to physically promote. The nominal body position and the attached encoder measurements agreed to about `1.5e-11` rad throughout these checks, which also argues against ordinary precontact joint position initialization as the source of these particular misses. It does not rule out ball tracking errors, unreachable geometry or motion representations that cannot enter useful poses. [Exact shortlist and motor projection receipt](funnels.json.gz)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt54_motor_funnel archive
python -m scripts.experimental.tt54_motor_funnel run --with-recenter --out /tmp/tt54-replicate.json.gz
python -m pytest -q tests/test_tt54_motor_funnel.py
```
