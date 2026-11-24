# Learning from the ball's first visible departure

A direct joint policy converted three of four previously missed practice openings by learning how earlier physically executed action changes affected the first outgoing ball. It did not replay a recorded stroke or ask a nominal court whether the new correction would score. On eight initially unseen opening releases, the ungated policy scored five versus six for the stock pupil. After a physical exposure to those openings, using only the camera-measured misses to gate and anchor the correction scored seven on repetitions with redrawn continuation seeds. This is local learning from experience, not a general solution to the broad rally. [Complete physical results and replay](physical-results.json)

## What was measured and learned

The archive holds exact action-matched prefixes from the earlier direct-joint repair comparison. The earlier researchers used a separate nominal court to produce alternatives, but these labels come only from alternatives that were actually issued on the original simulated table. Before-and-after stereo and single-eye rays estimate the signed change in first-contact velocity without reading simulator contact locations or velocities. Native contact and separation records are retained only as a separate diagnostic. In the 17 auditable camera detections the median departure error was .129 m/s in Euclidean velocity, and median impact-timing error was 1.22 ms. [Raw camera and encoder replay](paired-camera.json.gz) · [Independent native check](validation.json)

Eleven pairs across four releases had usable observed departures on both sides. Their issued reference differences up to the first impact were compressed into two smooth control coefficients per joint. The actor's current camera motion, measured joint angles and reference lag weight a local ridge estimate of how those motor changes move the first outgoing ball. A damped inverse proposes a bounded angle correction to move toward an optically measured legal departure. The usual raw camera-to-joint feedback still runs every tick. Neither a known launch ID, a whole archived stroke, a selected intercept, nor any private simulator tests the issued joint correction. Held out by entire release, the velocity-change forecast was worse than assuming no change, 1.51 vs 1.24 m/s mean error. That warning was borne out in the broad cold test. [Learned contrast fit](inverse.json) · [Implementation](../../scripts/experimental/tt40_contact_inverse.py)

## Physical comparisons

| Assessment | Stock | Changed | Scope |
| --- | ---: | ---: | --- |
| Four examined practice openings | 0 | 3 | New direct corrections |
| Same four practice openings | 0 | 1 | Invert every correction's sign |
| Eight novel measured releases | 6 | 5 | No prior experience on those releases |
| Repeat those eight first deliveries | 6 | 7 | Only own measured missed departures can authorize an adjustment |
| Four practice starts with redrawn replies | 0 | 4 | Learned first correction, then ordinary stock joint feedback |

Every first physical delivery matched exactly. The reversed correction still landed release 15270 but did not convert 5864 or 8857. On the new releases the ungated correction converted 3849 and spoiled two previously legal openings. The observed-miss policy used the genuinely measured failed baseline departure in place of the model's guessed baseline; it touched none of the six successful command streams. On 3849 the measured lateral velocity fell from .927 to .223 m/s and upward velocity rose from 3.386 to 3.748 m/s. The two rare misses on the new roster were learned from before this repeat. This is an anchored carry test, not a second cold generalization result. [Matched outcomes, signatures and trajectories](physical-results.json) · [Observed-miss tape](observed-miss-carry.json.gz)

On the four continued points, commands reproduced the shortened first-strike runs exactly through their final recorded ticks. Three first returns landed under fresh reply seeds, and release 8857 proceeded to two own legal returns and five alternating returns. The three selected changed tapes replayed exactly from actions with zero penetrations in 4,725 full CAD samples. [Physical continuations](practice-continuations.json.gz) · [CAD verification](physical-results.json)

## Learning from a failure that still scored zero

At 7038 the first attempted inverse changed measured departure only slightly, by −.064 m/s forward and +.029 m/s upward, and still scored zero. A local rank-one correction used that actually executed command contrast and its signed optical response, without a positive reward. The new readout substantially changed the next physical shot. Its lateral velocity went from +.474 to −.420 m/s and forward velocity from 3.434 to 3.688 m/s, but it still did not land legally. This establishes that a measured failure can change the next command and its actual effect, not that it reliably supplies the right new stroke. [Scoreless update](own-failure-updated.json) · [Physical recheck](own-failure-recheck.json.gz)

## Reproduce

Use the unchanged broad opponent and original measured launch feeds. The first-strike checks stop only after the actual score or physical termination. The continuation test uses the ordinary whole-point cap.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
out=artifacts/tt40-contact-effect
python -m scripts.experimental.tt40_real_contact "$out/paired-camera.json.gz"
python -m scripts.experimental.tt40_contact_inverse fit "$out/paired-camera.json.gz" "$out/inverse.json"
for mode in inverse reversed; do
  python -m scripts.experimental.tt40_contact_inverse physical "$out/inverse.json" practice "$mode" "$out/practice-$mode.json.gz"
done
for mode in fixed inverse; do
  python -m scripts.experimental.tt40_contact_inverse physical "$out/inverse.json" fresh "$mode" "$out/fresh-$mode.json.gz"
done
python -m scripts.experimental.tt40_contact_inverse physical "$out/inverse.json" fresh carry "$out/observed-miss-carry.json.gz" --witness "$out/fresh-fixed.json.gz"
python -m scripts.experimental.tt40_contact_inverse physical "$out/inverse.json" practice inverse "$out/practice-continuations.json.gz" --continue-point
python -m scripts.experimental.tt40_contact_effect_report validate
python -m scripts.experimental.tt40_contact_effect_report physical --replay
python -m scripts.experimental.tt40_contact_failure_credit update "$out/own-failure-updated.json"
python -m scripts.experimental.tt40_contact_failure_credit recheck "$out/own-failure-recheck.json.gz"
python -m pytest -q tests/test_tt40_contact_inverse.py
```
