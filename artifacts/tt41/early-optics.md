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
python -m pytest -q tests/test_tt41_early_optics.py
```
