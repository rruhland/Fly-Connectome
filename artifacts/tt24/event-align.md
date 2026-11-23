# Learning motor effects around a visible trajectory reversal

A wrist-observed reversal provides a time reference for relating recent commands to the outgoing ball. On wide resets, a first hit could send the ball more than a paddle width sideways before the next descent. An earlier fixed-time fit using 48 donor flights did not predict the tested optical action effects. The new tests paired signed command interventions at identical releases and measured the difference in the subsequent ball flight. [Earlier fixed-clock result](flight-effects.md) · [Expanded-posture physical audit](startup-acquisition.md)

## Single-encounter fit

Twenty-four paired donor releases supplied 48 single-encounter tapes and 1,645 wrist frames, about 27.4 seconds of simulated acquisition. Every branch added a signed, independently drawn sequence of short five-joint rate variations to live frozen donor feedback. The sequence was drawn before the release. The learner saw wrist RGB, current joint encoders and commanded rates. It received no physical contact time, teacher action or source rewards for this fit. The inherited donor motor and public sensor calibration are charged separately. [Donor motor history](tt24_donor.md) · [Stock metric calibration](public-body.md) · [Carried camera and link adjustment](wrist-egomotion.md)

An optical detector fits short segments before and after a localized change in trajectory. It requires visible descent, subsequent upward motion and continuity within the observed range error. The label is horizontal velocity from the rising optical segment. This label uses later observations during learning. An independent held fit compares the signed velocity difference of the plus and minus branches on an expanded-posture arm with unseen link dimensions, actuator parameters and ball dynamics. Sixteen donor pairs and ten held pairs met the paired optical and early-input requirements. A hidden scorer after replay put the median optical reversal error at one 60 Hz tick over 24 available held single-branch comparisons. [Source and held wrist traces](event-align-source.json.gz) · [First effect model](event-align-fit.json.gz)

| Action-window reference on the first held body roster | Optical lateral effect RMS | Physical lateral effect RMS |
| --- | ---: | ---: |
| Predict no difference | 0.292 m/s | 0.251 m/s |
| First ten ticks | 0.282 m/s | 0.240 m/s |
| Fixed donor median, tick 12 | 0.304 m/s | 0.268 m/s |
| Measured reversal, shuffled action pairs | 0.292 m/s | 0.248 m/s |
| Measured optical reversal | 0.259 m/s | 0.208 m/s |
| Learned early event forecast | 0.306 m/s | 0.238 m/s |

Retrospective alignment improved the velocity estimate, while the unconstrained model using an event forecast alone trailed the optical baseline. That forecast used five pre-intervention images and had mean absolute timing error of 1.05 ticks on the eligible paired held tracks.

## Prospective geometry-constrained effects

The second model first predicts encoder changes from accumulated issued commands at five delays. A public geometric readout converts the predicted joint change to a blade displacement and normal change. Three shared fitted coefficients then map these effects to horizontal outgoing motion. The model estimates event time from the early images, and evaluates the command history only through that predicted tick. It does not read a branch's later trajectory to choose the input window. The same 16 eligible donor pairs provide the fit, including 74 pre-event encoder samples. Only optical velocity differences train the outcome coefficients. [Geometry model](event-align-geometry.npz) · [Frozen independent gate](event-align-geometry-fit.json.gz)

Twelve additional paired releases on the novel wide body supplied 908 wrist frames. The detector found both optical reversals on all twelve pairs. Eleven had the early RGB support required for prospective evaluation. Before measurement, the gate required at least an 8% reduction in both optical and physical RMS relative to zero-change, shuffled geometry and first-ten controls. [Independent command and wrist tapes](event-align-geometry-held.json.gz)

| Independent prospective comparison | Optical lateral effect RMS | Physical lateral effect RMS |
| --- | ---: | ---: |
| Predict no difference | 0.184 m/s | 0.193 m/s |
| First ten ticks | 0.167 m/s | 0.168 m/s |
| Shuffled geometry fit | 0.191 m/s | 0.203 m/s |
| Measured optical reversal | 0.138 m/s | 0.124 m/s |
| Early forecast with the geometry fit | **0.131 m/s** | **0.123 m/s** |

The geometry model passed the gate. Its predicted effect pointed in the measured optical direction on nine of eleven pairs. The effect comparison knows which joint commands each intervention actually issued through the expected event. The live test below requires a controller to choose those commands from current wrist input.

## Reward-only controller test

The carried donor policy continued to issue all five commands from live wrist and encoder state. A correction used the early visual event forecast, current ball motion and the measured motor response to reduce predicted lateral drift. It requested bounded joint-rate changes, then retired after the first encounter estimate. The diagnostic leveling comparison used the same sensors, timing, encoder predictor and actuator limits with a fixed normal-only response. No stored action sequence or contact label entered play. [Closed-loop controller](../../scripts/experimental/tt24_event_align_live.py)

Four separate development releases supplied 40 reward comparisons and 1,218 simulated control ticks across all candidates. For each method the selector maximized the same capped rebound-count reward with a bonus for reaching a second rebound. It retained the frozen controller on ties with zero gain. Among tied nonzero settings it selected the smallest. The chosen gain was 0.45 for the learned response and leveling comparison and zero for the shuffled fit. The learned controller then changed rates on 113 control ticks across 16 entirely new wide-body starts. [Development choices and held motor tapes](event-align-live.json.gz)

| Controller on 16 new wide starts | Total rebounds | First | Second | Best consecutive |
| --- | ---: | ---: | ---: | ---: |
| Frozen donor | 13 | 12/16 | 1/16 | 2 |
| Learned response | **25** | 12/16 | 2/16 | **11** |
| Shuffled response, zero gain selected | 13 | 12/16 | 1/16 | 2 |
| Normal-only leveling | 23 | 13/16 | 2/16 | 7 |
| Zero command | 16 | 13/16 | 2/16 | 3 |

The learned correction improved two matched starts and tied fourteen against the frozen controller. Seed `371401` yielded 11 consecutive rebounds with eight changed control ticks. It yielded two with the frozen policy, seven with leveling and three with holding. The other improvement was four rebounds versus one. The small total margin over leveling and the second-rebound rates leave the overall game advantage uncertain. All 40 development and 80 held command tapes reproduced the exact game referee record. [Matched wrist-camera replay](event-align-live.mp4)

A separate instrumented replay measured lateral flight speed after the first rebound. The learned correction reduced that speed on all 12 matched first hits, with the median falling from 0.766 to 0.645 m/s. The normal-only comparison had a median of 0.624 m/s across its 13 first hits. On the eleven-rebound witness, the second descent went from 13 mm inside the rim under the frozen donor to 36 mm inside with the learned correction and 32 mm inside with leveling. These physical readings were used only after the learned and comparison commands were sealed. [Instrumented exact replay](event-align-live-audit.json.gz)
