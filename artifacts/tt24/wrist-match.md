# Carrying the rally pupil to a single wrist camera

The experiment freezes the combined motor readout that reached eleven near-arm returns in the prior S/T assessment. It registers a wrist ball silhouette and the actual joint readings into the reference coordinates of that pupil. A separate kinematic buffer uses the nominal robot fixture and lens calibration. Four synchronized clips provide optical labels for ball range, bearing offset and the grip-to-paddle position. A short causal range update smooths the wrist estimate, and the resulting reference observations feed the existing direct joint feedback on every frame. The wrist branches render one actual wrist camera at 320×240 during play. Every near-arm command comes from the policy, using wrist RGB and its own joint and command histories. [Wrist transfer code](../../scripts/experimental/tt24_wrist_match.py) · [Frozen source pupil](rally-improve.md)

The four clips use 96 consecutive frames each from earlier verified games, for 384 frames or 12.8 simulated seconds. They yield 284 usable synchronized ball labels and 384 paddle labels. The clips add no motor training labels to the frozen motor law. Its earlier motor demonstrations and corrective teaching remain charged in the source report. Leaving each clip out in turn gives median ball-position errors of 23–41 mm and 90th-percentile errors of 121–262 mm. The fitted size correction's in-sample median position error was 28 mm. A simpler one-scale radius conversion with the same nominal kinematics scored 23 mm on those frames, so the case-level control results assess the full observer and do not isolate the benefit of that regression. [Paired sensor data](wrist-match-calibration.npz) · [Calibration exposure](wrist-match-calibration.json.gz) · [Optical folds](wrist-match-observation-audit.json.gz)

## Twelve held points against the active physical partner

The same fixed motor law plays every scheduled start with live stereo, the wrist observer, or the wrist observer with its registered visual input delayed six frames. The hold branch issues zero rates.

| Outcome across 12 releases | Stereo | Wrist | Delayed wrist | Hold |
| --- | ---: | ---: | ---: | ---: |
| First legal near return | 6 | 5 | 1 | 0 |
| Second legal near return | 1 | 4 | 0 | 0 |
| Third legal near return | 1 | 3 | 0 | 0 |
| Fifth legal near return | 1 | 0 | 0 | 0 |
| Total near returns | 13 | 12 | 1 | 0 |
| Longest alternating exchange | 16 | 6 | 1 | 0 |

Wrist play produced three near returns and three far replies on each of `2112730568`, `205721030` and `1102959212`. The stereo pupil returned once, once and zero times on those three points. Stereo reached eight near returns on a different launch, where wrist play scored zero and delayed wrist scored its sole first return. The wrist controller's first second return came on point two after 110 evaluation frames. All five active joint rates changed in the wrist-delay comparison, and all 48 complete physical tapes reproduced the referee's exact events and terminal results. [Physical game tapes](wrist-match-held.json.gz) · [Wrist-only replay with onboard inset](wrist-match-held.mp4) · [Video verification](wrist-match-held.json)

The prior one-clip depth fit reached a second return on three of 16 different held launches and once reached eight alternating returns. The present paired test establishes wrist-only access to second and third turns with the stronger motor readout. Its six-return peak and zero fifth returns leave the source pupil's extended rally capability unresolved for this wrist transfer. [Prior wrist experiment](rally-eye-depth.json.gz)

## Reproduce

Use the existing [table-tennis simulator environment](../tt23/tt23_sequence.md#reproduce) and [source rally artifacts](rally-improve.md#reproduce).

```sh
python -m scripts.experimental.tt24_wrist_match --workers 2
python -m scripts.experimental.tt24_wrist_match --audit
python -m scripts.experimental.tt24_wrist_match_movie
```
