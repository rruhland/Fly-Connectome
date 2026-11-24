# Can stereo RGB choose the long low-ball preparation in time?

Better tracking after the far hit is not enough to reproduce the timing of the three verified early-rising strokes. The earliest visible-exposure estimate with inward longitudinal speed below −1.15 m/s arrived **0.73–1.10 seconds after** the successful arm movement had already begun. It left just 0.15–0.32 seconds of the tested preparation window. This uses the camera estimates recorded from the actual moving arm in the physical witnesses, not a privately time-aligned hindsight trajectory. Actual contact times serve only to audit the timeline. These were successful examples, not trials of whether a different motor skill could work with shorter notice. [Timing and source digests](teacher-timing.json)

| Practice release | Physically issued preparation | First inward estimate after prep start | Time left to tested preparation end |
| --- | ---: | ---: | ---: |
| 14843 | 0.883 s | 0.733 s | 0.150 s |
| 4322 | 1.333 s | 1.100 s | 0.233 s |
| 8302 | 1.167 s | 0.850 s | 0.317 s |

## The outgoing shot carries an early clue

The severe receive persisted when only the upcoming wide random aim changed. All six fresh draws were below 0.95 m on the same three real outgoing motions. The player replayed its earlier actions and receipts exactly, then let the stock finite-mass opponent plan toward a fresh goal from the original full distribution. This is a counterfactual sensitivity assay, not a new benchmark or an actor with access to the draw. The heights in this assay are native evaluation measurements at the same projected depth. [Full paired timing and pixel receipts](draw-visibility.json)

| Practice release | Original height | Native heights after two fresh draws | First short-bounce RGB cue after prep start |
| --- | ---: | ---: | ---: |
| 14843 | .935 m | .932 / .933 m | 67 ms |
| 4322 | .914 m | .911 / .935 m | 67 ms |
| 8302 | .892 m | .902 / .914 m | 50 ms |

In 4322 the far camera regions were pixel-identical across both redraws through at least 0.367 s after the normal enabled preparation start. In the other two releases a few pixels changed at most 67 ms before that start, but ten pixels of material contrast changed only 17–33 ms after it. The short-bounce cue is a drag-and-gravity projection of the **outgoing** observed ball hitting the far surface at y < 0.65 m. It was visible while the ball was still near y = −1.04 to −1.16 m on all three cases, about half a second to a second sooner than the inward estimate. Three known positives do not establish its precision. They suggest a short, risk-based initial move or a shared starting move before trusting the inferred far-face pose.

## The arm can still act after seeing the cue

All three practice balls were returned and physically answered when the arm waited for the causal shortness cue. The camera, running live during exact replay, triggered the delayed paths at ticks 96, 456 and 300. The controller time-compressed each **assigned** physically earned preparation to its already tested ready tick and preserved the reference suffix under the standard rate and slew limits. All three achieved the identical actual far hit and near bounce as before. CAD checked all 3,400 integration states from the changed force onward with no penetrations. Minimum audited clearances were 18.3, 2.4 and 14.8 mm. This establishes a useful timing window for this sensor. It does not establish a causal way to select or retime the right motor route on an unseen point. [Original practice retiming, 14843](identified-preparation.json.gz) · [Strict-CAD retiming, 4322 and 8302](strict-preparation.json.gz)

The first two 4322 and 8302 preparations in the older original archive still involved table rim penetrations in the delayed replay. They are diagnostic only. The two later archived strict-CAD variants are the controls used to count clean results above.

## Causal observer study

The observer uses the existing fixed 384×288 stereo cameras at 30 Hz with no changes to physics or opponents. It can mark the first outward passage of y = −.70, −.10, +.45 and +.95 m, the first credible optical reversal, four inward lines, and the optically inferred near rebound. Those gates run continuously over unchanged, already-issued articulated arm actions. They receive no referee event, seed, release ID, clock relative to actual contact or hidden geometry. Raw optical clocking and the camera fit must earn the gate. A separate scoring step mates its calls to real far native contacts and excludes a mislabeled phase. Missed or late calls reduce deadline recall. This separates pose/motion visibility from the ability to turn it into a useful early decision.

The learning target is projected ball height at y = −1.53 m. After a real far hit, a later stereo arc detects the near-side rebound based on a local height turnaround and fits its following motion with gravity and damping. Referee markers audit the detected bounce and stop measurements at the near racket to prevent a different stroke from becoming the label. They do not give the label coordinates or enter a prediction. As a check on coverage and selection, the fit report compares these optical labels to a separate native crossing extrapolation that is never used for training or model selection.

A release-held practice split compares unconditional prevalence, outgoing or inward ball fit alone, far blade motion and moments alone, the combination, and the generic rebound projection where available. Ridge penalties and operating thresholds are estimated on training releases only. We report a same-sample comparison where both cues are visible, as well as operational recall over all optically labelled receives including missing gates. Two low cuts are 0.99 m and the more rare 0.95 m. The nominal deadlines are 0.88, 1.0, 1.2 and 1.33 seconds before both the projected receive line and the actual near bounce, so a good late classifier cannot be mistaken for available early control.

### Held timing and discrimination

The 36 exact articulated replays used the first 12 assignments in each source. The held source contributed 33 earned later deliveries across eight releases. Stereo recovered 22 usable later height labels, including all three below 0.99 m and the only one below 0.95 m according to the separate evaluator. The optical and evaluator labels agreed on both cutoffs whenever both were available. Mean absolute height disagreement was 5.7 mm. Training used 80 optical labels from 105 earned later deliveries after removing one release shared with holdout. [Optical receipts](receipts.json.gz) · [Precision–recall curves, deadline counts, and held calibration](forecast.json) · [Frozen ridge fits](observer.json)

| Causal gate | Median before actual bounce | Held with both cues | Prevalence AP | Ball AP | Paddle AP | Combined AP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Outbound −.70 m | 1.018 s | 22 | .136 | .487 | .545 | .550 |
| Outbound −.10 m | .830 s | 22 | .136 | .389 | .389 | .553 |
| Outbound +.45 m | .608 s | 21 | .143 | .437 | .350 | .300 |
| First credible inward fit | .277 s | 19 | .158 | .792 | .533 | .567 |

Average precision (AP) ranks sub-0.99 m arrivals. The earliest combined model ranked these few lows above baseline, but its AP gain versus the ball alone was not consistent under release resampling (95% interval −.017 to +.208). No model selected by requiring at least 60% precision on practice folds called an actual held low with 0.88 s still left before the near bounce. The fixed geometric rule used in the timing demonstrations called two of three held lows at the earliest comparable gate, but made 14 calls total. Only one of those two arrived at least 0.88 s before the bounce. The only severe held positive was one of the early calls, which is too little to estimate reliable severe-ball precision. The evidence supports a timely but weak risk cue for a small reversible preparation, not a trigger for committing to a long, exclusive motor route.

## Reproduce

Run the following commands with the project environment and MuJoCo available. The replay saves completed matches incrementally and can be resumed with `--resume`.

```sh
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 python -m scripts.experimental.tt41_anticipate --limit 12
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 python -m scripts.experimental.tt41_anticipate_fit --models artifacts/tt41_anticipate/observer.json
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 python -m scripts.experimental.tt41_anticipate_fit --teachers --out artifacts/tt41_anticipate/teacher-timing.json
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 python -m scripts.experimental.tt41_anticipate_reveal
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 python -m scripts.experimental.tt41_anticipate_prepare
PYTHONPATH=.:src OPENBLAS_NUM_THREADS=1 python -m scripts.experimental.tt41_anticipate_prepare --witness artifacts/tt40/long-low-alternate-strict-optical.json.gz --output artifacts/tt41_anticipate/strict-preparation.json.gz
```
