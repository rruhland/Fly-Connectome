# The next ball and the arm's recovery

## Our shot changes the next physical flight, not its random aim

A matched census used seven frozen direct-policy action archives. In 179 comparisons across 25 game-leg keys, two different preceding near shots received the same seeded independent next placement goal and both produced a legal opponent reply. Despite identical requested targets, the next physical balls usually differed. These are comparisons of repeated policies on only 12 measured releases, not 179 independently drawn starts. A reply that could not be delivered is excluded, so the census does not estimate that failure rate. [Matched return changes](matched-replies.json)

| Absolute change between matching next targets | Median | 90th percentile | Maximum |
| --- | ---: | ---: | ---: |
| Our impact to opponent impact | 29 ms | 106 ms | 446 ms |
| Lateral position at the archived near receive line | 13 mm | 42 mm | 81 mm |
| Height there | 22 mm | 80 mm | 188 mm |
| Forward speed there | .071 m/s | .836 m/s | 1.617 m/s |

Lateral arrival velocity was more predictable than height, forward pace, or extra flight time. A paired-difference regression reduced its mean absolute error from .428 to .129 m/s holding out an entire opening release at a time. Its fitted slope associated a one-meter lateral shift of the preceding outgoing bounce with a −2.01 m/s shift in the next incoming lateral velocity, controlling other recorded outgoing changes. It also improved prediction of the small shifts in the next lateral position. It did not improve mean absolute prediction error for height, forward pace, vertical pace or recovery time. This supports a physical influence on cross-court momentum. It does not establish a learned shot that improves the score or a way to select the opponent's future location. No archived aim or simulator observation supplies the actor below. [Release-blocked influence check](reply-influence.json)

## Recovery time was not the bottleneck in these points

Action-only replays recovered the measured encoder history for 39 later balls on the direct `self-combined` pupil and 31 on the elastic joint-search learner, including their long points. At all 70 later opponent hits, the largest error on any joint was .000509 rad. This held on all 12 sampled later misses as well as the successful exchanges. The median time to come within .18 rad of home was about .30 seconds after the preceding own impact, against .93 seconds for the far hit. By the inbound bounce the arms had already moved substantially away from home again. Faster home recovery by itself buys little on these tapes. [Per-ball encoder witness](actual-readiness.json)

## A minimal common windup

From 519 physically scored training strokes, the pupil learned which early movements were shared across left, middle and right motor branches. A release-held score of the larger joint travel to the observed preparation states improved from .494 to .307 rad at the median and .782 to .619 rad at the 90th percentile. Only the elbow and wrist-flex axes moved in the preload. Neither lateral pan nor wrist roll committed to a direction. This score measures geometry of recorded successful postures, not how well the arm can hit from its new state. [Fit and held joint-travel proxy](preload.json)

The live controller begins preparing only after the existing public score increases for a real near return. It preserves the entire opening stroke and its outbound flight. It sends the learned ready target with the existing joint envelopes until the first reliable inbound stereo fit, then hands off to the unchanged direct joint field. It sees neither an opponent trajectory label nor an aim. The paired assessment includes every baseline opening that already scored, since the intervention cannot trigger on an unreturned opening. [Implementation](../../scripts/experimental/tt39_common_preparation.py) · [Unchanged runner replay](neutral-reproduction.json.gz)

Across the nine paired points, the unchanged pupil landed 39 returns, 30 later than the opening. The preload landed 20, including 11 later. It converted the first later reply in six points compared with five for the control, but shortened several existing long rallies. The longest fell from 11 own returns to 4. All first-later aims and far contacts matched. The first-later physical offer matched exactly on eight points and to solver roundoff on the ninth. Across 20 handoffs, median preparation duration was .317 s and median remaining error to the new target was .72 mrad. The arm reached the new posture, but the new posture was not a reliable replacement for the default. [Paired physical results](preload-result.json) · [Preload action tape](preload-held.json.gz)

Release 11886 improved from one to four own returns. Its independent action-only replay reproduced nine alternating returns with zero penetrations in 4,108 full CAD samples and 15.9 mm minimum separation. This was a genuine local improvement despite the aggregate loss. [Score and replay receipt](preload-result.json)

## Reproduce

Use the original measured launches and unchanged TT37 broad-opponent plans.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
out=artifacts/tt39-recovery
python -m scripts.experimental.tt39_recovery_audit matched "$out/matched-replies.json" artifacts/tt37/{long-rgb,rollin-rgb,self-combined,matched-teacher-rgb,matched-rollin-rgb}-held_openings.json.gz artifacts/tt38-body-relative/{position,motion}-held_openings.json.gz
python -m scripts.experimental.tt39_recovery_audit influence "$out/reply-influence.json" "$out/matched-replies.json"
python -m scripts.experimental.tt39_recovery_audit encoders "$out/actual-readiness.json" artifacts/tt37/self-combined-held_openings.json.gz artifacts/tt38/joint-mpc-extended.json.gz
python -m scripts.experimental.tt39_common_preparation fit "$out/preload.json"
python -m scripts.experimental.tt39_common_preparation test "$out/preload-held.json.gz"
python -m scripts.experimental.tt39_common_preparation report "$out/preload-result.json" "$out/preload-held.json.gz" --replay
python -m pytest -q tests/test_tt39_common_preparation.py
```
