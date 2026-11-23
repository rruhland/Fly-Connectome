# Estimating paddle orientation from one wrist camera

Four ordinary practice drops on the altered MuJoCo arm produced a small correction to the carried policy’s paddle normal. On six fresh four-second drops, the correction raised the mean longest streak from 7.83 to 10.00 and the minimum from one to nine. It reduced the number reaching ten rebounds from three to two. The result supports a short-horizon reliability gain on this course. The four-second cap and six starts leave the effect on longer rallies unresolved.

The carried controller supplied the existing RGB ball tracker, wrist calibration, encoder-to-link map, and frozen motor weights. Some of that earlier calibration used fixed cameras. This fit and its live games used only the 192×144 wrist camera and encoders. The fit subtracted an acceleration estimated from tracked free-flight segments and looked for an abrupt upward change in ball velocity near the estimated blade. The main assumption was that, when tangential slip is small, the ball’s velocity change approximates the blade’s upward normal. Each qualifying drop contributed its median angular residual with equal weight. The learner received neither referee contact events nor simulator state. The unchanged physics referee scored each game and reproduced results from the recorded action tapes.

## Hidden-tilt controls

The first scout tried to explain world-y tilt by rotating the calibrated normal within its local xy plane. This discarded the relevant component because the local z axis pointed nearly along world −y. The first scout failed with two accepted events in each tilted condition and almost no recovered world-y separation. The raw impulses still had distinct normalized world-y residuals of +0.124, +0.019, and −0.069 for −6°, ordinary, and +6° fixture settings. That observation motivated a fresh registered replication with the correction taken along local z, perpendicular to the calibrated normal. [Initial plan](wrist-impulse-plan.json) · [Initial scout](wrist-impulse-scout.json.gz) · [Replication plan](wrist-impulse-local-z-plan.json)

The replication crossed four fresh stock launches with all three fixture settings. Physical tilt was hidden from the estimator. All four releases in each condition supplied optical witnesses, and each estimated separation exceeded its registered threshold. [Replication scout](wrist-impulse-local-z-scout.json.gz)

| Physical fixture tilt about world x | Accepted optical events | Per-drop median world-y component of inferred minus original normal |
|---|---:|---:|
| −6° | 4 | +0.1163 |
| Ordinary | 25 | +0.0154 |
| +6° | 4 | −0.0815 |

The four altered-arm practice drops supplied 32 optical events across three starts. Their median local residual was −4.54°. Subtracting the matched ordinary stock residual of −0.88° and shrinking by 3/4 froze a −2.74° correction. It was applied when the original wrist-image classifier chose the altered calibration. A sign-reversed correction used the same magnitude and controller as a directional control. [Frozen fit](wrist-impulse-local-z-fit.json.gz)

## Fresh paired drops

| Altered observer | Longest streaks by paired release | Mean streak | Eight or more | Ten or more | At four-second cap |
|---|---|---:|---:|---:|---:|
| Frozen | 2, 12, 11, 7, 14, 1 | 7.83 | 3/6 | 3/6 | 4/6 |
| Learned correction | 9, 9, 12, 12, 9, 9 | 10.00 | 6/6 | 2/6 | 5/6 |
| Sign reversed | 2, 15, 12, 1, 11, 1 | 7.00 | 3/6 | 3/6 | 3/6 |

The learned correction won four of six streak comparisons against each control and increased mean survival to 3.83 seconds from 2.98 for frozen and 2.40 for sign reversed. Eight rebounds summarizes the observed lower tail. Two independent stock launches produced exactly identical action tapes and referee scores with the correction available, at zero and twelve rebounds, because the wrist-image classifier selected the stock calibration. [Held action tapes and scores](wrist-impulse-local-z.json.gz) · [Runnable study](../../scripts/experimental/tt24_wrist_impulse.py)
