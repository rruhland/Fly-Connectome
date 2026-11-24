# Earlier causal recognition of a rising table rebound

The new optical observer can confirm a table bounce on the first informative stereo exposure. It improved phase timing more reliably than position or velocity. In three small full-random paired live trials, changing only this readout increased legal near returns from 6 to 11. The independent 30 Hz sensor panel did **not** show a velocity-accuracy gain, so this is a promising control result and not a general detector upgrade.

## What changed

The legacy learned camera readout already predicts a table launch from the observed descent and 140 calibration bounces. Its p/v can be correct while the `Flight.bounced` flag still waits for the generic raw-ray fit. The new observer fits a continuous two-piece trajectory using an already observed descending segment and the first meaningful upward deviation. Candidate contact times must be earlier than the current measurement. A calibrated restitution prior helps infer early rising velocity, and both phases share one contact center. Once the stock positive-rise readout agrees, the patch returns to stock p/v and retains only the confirmed phase flag.

The estimator receives no referee timestamp, actual bounce state, partner goal, future image or actuation tape. Scorer-only truth is loaded after processing both camera streams. Earlier practice rebound priors may use completed optical segments. That is distinct from reconstructing a bounce retrospectively at decision time.

## Sensor receipts

| Camera panel | Unique actual offers | Bounce flag on first physically flagged exposure | First-exposure vertical velocity MAE |
| --- | ---: | ---: | ---: |
| 60 Hz stored held camera streams | 32 | 5 → 24 | .597 → .503 m/s |
| 30 Hz new independent first-offer replays | 9 | 4 → 7 | .245 → .293 m/s |
| 30 Hz selected early-rising examples | 3 | 2 → 3 | .114 → .085 m/s |

There were no false active exposures in those receipts. The first physically flagged exposure is not always after the short contact impulse has finished. The claimed numbers measure behavior against that physical marker, not subframe knowledge from the first photons. On the 60 Hz set, the number of first exposures within 0.35 m/s of the actual positive vertical velocity was 14 → 16. On the independent 30 Hz set it was 7 → 6. Subsequent reported velocity errors were stock-identical after hand-back. A complete velocity overlay instead worsened later readings in the exploratory 60 Hz comparison.

The three early-rising tapes represent three distinct physical incoming offers. The repeated 14843 flight from another motion trace was deduplicated. Each had a real contact 108–131 ms after the legal near-side bounce. On 14843, stock learned p/v already forecast the scorer-only strike plane to .18 ms in time and about 4.85 mm in height on the first post-bounce exposure, but its flag arrived 33 ms later. The patch supplied the flag immediately without changing those useful readings. On 4322, it reduced first-exposure forecast height error from 16.0 to 9.2 mm but increased time error from .35 to 2.32 ms. On 8302 the first forecast was unchanged. The physical strike plane is used only to score these retrospective forecasts. [Strike-plane errors](early-optics-rising-contact.json) · [Independent physical stereo replay](early-optics-independent30-errors.json) · [60 Hz lead time and p/v](early-optics-held60-gated.json)

## Frozen live policy check

The same elastic MPC, same motor bank, same frozen memory, same randomly varying live opponent, and same measured starts were run under each optical readout. No action search, motor or opponent setting changed. The small preassigned slice used three openings disjoint from the optical calibration roster and capped both arms at 24 alternating legal strokes per point. No point reached the cap.

| Measured release | Stock near / alternating legal | Earlier optics near / alternating legal | First changed action |
| --- | ---: | ---: | ---: |
| 8772 | 3 / 7 | 4 / 9 | 168 |
| 9965 | 2 / 5 | 6 / 13 | 195 |
| 3905 | 1 / 3 | 1 / 3 | None, exact complete tapes |

Both changed points matched physical histories before the first different action, and the first affected incoming offer was actually returned by both arms. Later physical replies diverged. This is a small rally outcome comparison, not evidence of two newly saved first offers. The inserted readout took 74.6 ms of additional CPU across 661 altered-arm camera exposures, about 0.113 ms/exposure. That number excludes the already present renderer, stock filter and motor search. These results don't establish broad transfer or continuous 100-return play. [Paired physical runs](early-optics-live.json.gz)

## Stronger carried online comparison

The optical-only player did not beat the 109-return carried controller. I reused all twelve replication cards in the original order, 47 initially credited physical practice strokes, the exact 321-row starting digest, a 768-row memory limit, intersected camera credit, the original random opponent and the 200-stroke cap. Every game continued to update memory. The archived stock column and the alternate column are whole physical courses. They do not hold the learned state fixed after the first game.

| Release | Stock own legal | Earlier optics own legal |
| --- | ---: | ---: |
| 12109 | 8 | 6 |
| 9812 | 6 | 20 |
| 5098 | 17 | 9 |
| 3173 | 5 | 7 |
| 12259 | 1 | 16 |
| 8510 | 0 | 0 |
| 9171 | 13 | 8 |
| 4570 | 9 | 1 |
| 5481 | 8 | 8 |
| 11023 | 6 | 6 |
| 10873 | 1 | 6 |
| 14390 | 35 | 17 |
| **Total** | **109** | **104** |

There were four gains, five losses and three ties. Total alternating strokes were 230 versus 220, and the longest point fell from 35 near legal returns and 71 alternating strokes to 20 near legal returns and 41 alternating strokes. No game reached the cap. Independent rescoring verified all twelve physical prefixes through the first changed command. One game's entire tapes matched. Of the eleven changed games, the first common incoming opportunity was returned by both players ten times. The alternate earned one local rescue on 12259. On that moderate-low shared offer, stock contacted the ball but sent it out toward the near player. The alternate produced a legal strike and the rally continued for 16 own returns. The memory had already diverged from previous games, so this is not an isolated proof that phase alone caused the rescue.

The inserted estimator cost 0.733 seconds of CPU across 5,593 actual camera exposures in the twelve alternate games, or 0.131 ms/exposure on average. The whole alternate pipeline took 335.2 seconds of CPU and the archived stock pipeline took 374.9. Different action tapes and point lengths prevent treating that as a speedup. The independent 30 Hz stereo replay had a .89 ms 99th percentile added CPU cost, while the simulation's full motor-planning and rendering pipeline remains much more expensive. The final independently hashed memory checkpoint contains 768 actual credited segments. This observer is therefore retained as an experimental option, not recommended as a new default for the stronger controller. [Whole-course commands](early-optics-carry.json.gz) · [Paired scoring and timing](early-optics-carry-score.json) · [Final online bank](early-optics-carry.npz) · [Observer CPU profile](early-optics-cpu.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt41_early_optics artifacts/tt41/early-optics-held60-gated.json
python -m scripts.experimental.tt41_early_optics_capture known artifacts/tt41/early-optics-rising30.npz
python -m scripts.experimental.tt41_early_optics_capture held artifacts/tt41/early-optics-independent30.npz --source artifacts/tt40/initial-second.json.gz --limit 10
python -m scripts.experimental.tt41_early_optics artifacts/tt41/early-optics-independent30-errors.json --capture artifacts/tt41/early-optics-independent30.npz
python -m scripts.experimental.tt41_early_optics_contact artifacts/tt41/early-optics-rising-contact.json
python -m scripts.experimental.tt41_early_optics_live artifacts/tt41/early-optics-live.json.gz
python -m scripts.experimental.tt41_early_optics_carry artifacts/tt41/early-optics-carry.json.gz
python -m scripts.experimental.tt41_early_optics_carry_report artifacts/tt41/early-optics-carry.json.gz artifacts/tt41/early-optics-carry-score.json
python -m pytest -q tests/test_tt41_early_optics.py
```
