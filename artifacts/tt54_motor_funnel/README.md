# Is the motor controller throwing away recoverable movements?

The direct arm controller forecasts 21 moving ticks at 60 Hz, with additional constant-reference time to judge the ensuing ball. Its stock short screen promotes approximately eight of 28 candidate paths. This audit replays actual photographed, encoder-controlled broad-random points and asks whether the discarded paths had any modeled legal nominations. It also tests whether a small set of candidates improves when motor perturbations are generated around the *issued* path instead of its often unreachable desired references.

In the independent 32-game TT48 course with the 63-return point, 207 returns scored and 30 measured bounced shots did not. The arm touched 19 of those 30. Late searches showed at least one legal native prediction on all 207 scores but only eight misses. Of 30 misses, four ever got a winning robust-voted forecast. The median last-search gap was three ticks on both scores and misses, measured to the actual contact or the available point endpoint. Good and failing observed centers separated much more strongly than the issuance gap. Median best virtual center-to-ball distance was 25 mm for scores, 69 mm for misses that touched, and 96 mm for misses that did not touch.

Python's reference conversion and the native forecast agree on the signed factor of .8, the absolute joint limits, and the same eight physical integration steps per action. No point exceeded the rated 5.45 rad/s measured shaft limit. During the final up-to-12 ticks after bounce, flex at the wrist used the command slew ceiling on 58% of scored windows and 39% of missed windows. Shoulder lift did so on fewer than 2% in both. These endpoint-based windows differ for a no-contact miss. This is observational and is not a reason to relax the rating. [Issued rates and outcomes](actual-rates.json)

## Audit design

- The first eight eligible distinct archived official releases were selected in their original order. Every case includes the last actual failed bounced offer, with its prior scored offer as a control where available. Archived commands, events, actual placements and traced search notes must match exactly again. Additional model trials are diagnostically separate and never influence the replayed actor. [Fixed native funnel roster](protocol.json)
- The later four releases also serve as a small paired check of executable motor centers. Stock generates the same desired seed paths at the same camera-derived state, using exactly the same initial memory and RNG. The alternate first applies the existing signed motor and absolute position saturation to each seed, then generates its mutations around the resulting feasible path. No extra proposal rows are available to the alternate. At each otherwise empty failing search, the stock actor still issues only the original command. If an alternate receives both independent native robustness votes, the earliest per case is checked as a separate physical fork with the same ball and unchanged physical opponent. [Fixed recentering roster](recenter-protocol.json)

This is a motor diagnostic, not a new held gameplay claim. The elbow preparation, older motion memory, outbound choice and perception remain unchanged.

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt54_motor_funnel archive
python -m scripts.experimental.tt54_motor_funnel run --with-recenter --out /tmp/tt54-replicate.json.gz
python -m pytest -q tests/test_tt54_motor_funnel.py
```
