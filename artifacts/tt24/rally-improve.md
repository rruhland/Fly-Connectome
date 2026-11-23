# Direct rate feedback, observed reversal, and corrective practice

Every assessment near-arm command comes from live stereo RGB at 240×180 pixels, measured joints and locally issued joint commands. The same MuJoCo opponent and strict referee score complete points. Each actor calculates one set of five active joint rates per frame. The policy has a closed gripper. All teacher data are training inputs. The follow-up reserves separate two-stream cohorts and checks 144 distinct proposed launch IDs against the original TT24 cohorts, the teacher bank, and 265 earlier TT22/23 JSON receipts. [Assessment roster audit](rally-improve-schedules.json.gz)

## Direct rate and relative posture

Five candidate readouts share the 48 demonstrated release tapes and their 5,124 individual samples. Four variants register the current and earlier camera observations using the fixed camera calibration to estimate three-dimensional visual position and velocity. Relative outputs estimate a change from the measured joint posture, with a second learned rate output. An ambiguity control weights nearby examples by consistency of their demonstrated actuator responses. The rate-heavy retina policy keeps the original two-view pixel representation, uses 36 neighbors, and gives its learned rate output a weight of 0.72. Its remaining weight goes to a mixture of absolute and relative posture correction. All fits use local least squares. [Policy implementation](../../scripts/experimental/tt24_rally_improve.py) · [Original teacher exposure](rally-bank-demonstrations.json.gz)

Selection ranks candidates by counts of fifth and second near returns, then total and first near returns on twelve E/F development releases. The original broad fit remains the independent comparator. The rate-heavy retina candidate scored seven near returns and seven far replies on development seed `1652822702`. The relative-posture consistency fit reached its second near return on five of twelve development launches and reached four near returns on its longest point. All 60 development tapes passed referee replay. [Development comparison](rally-improve-development.json.gz)

| Fresh G/H outcome across 24 launches | Broad | Rate-heavy retina | Delayed RGB | Hold |
| --- | ---: | ---: | ---: | ---: |
| First near return | 13 | 12 | 0 | 0 |
| Second near return | 6 | 7 | 0 | 0 |
| Fifth near return | 0 | 2 | 0 | 0 |
| Total near returns | 19 | 29 | 0 | 0 |
| Points with at least ten alternating returns | 0 | 2 | 0 | 0 |
| Longest alternating exchange | 4 | 14 | 0 | 0 |

Fresh seed `955566762` reached five near and five far returns. Seed `76454876` reached seven near and seven far returns, with legal near landings at frames 37, 78, 108, 147, 191, 226 and 259. The broad policy earned two near returns on each seed. The first qualifying fifth return came on the eleventh point after 875 cumulative control frames, or 29.17 seconds. In overlap with the broad treatment, the rate-heavy fit changed shoulder pan, shoulder lift, elbow flex, wrist flex and wrist roll on 1,102, 1,091, 1,088, 1,020 and 1,074 frames. All 96 independent score tapes passed strict physical replay. [G/H score tapes](rally-improve-held.json.gz) · [Fourteen-return held video](rally-improve-held.mp4) · [Video verification](rally-improve-held.json)

## Visually phased feedback

The broad fit controls the opening from release. When the camera history shows the ball approaching on the near half and then moving away there, continuous feedback weights shift toward a rate-heavy or relative-posture fit over four frames. The two conditions retain fresh observation-to-rate feedback on every frame. A visual delay shifts both the retained motion and the camera-driven phase detector. No demonstration is added for this step. [Visual reversal and continuous mixing](../../scripts/experimental/tt24_rally_improve_phase.py)

| Development outcome across 12 E/F launches | Broad | Rate blend | Relative blend |
| --- | ---: | ---: | ---: |
| First near return | 7 | 7 | 7 |
| Second near return | 4 | 6 | 5 |
| Fifth near return | 2 | 3 | 1 |
| Tenth near return | 0 | 1 | 0 |
| Total near returns | 20 | 36 | 21 |
| Longest alternating exchange | 10 | 24 | 12 |

The rate blend scored nine, seven and twelve near returns on seeds `146477391`, `1176636968` and `1682479719`, respectively. The same points gave the broad pupil three, five and five near returns. The longest point contained 12 legal near returns and 12 legal far replies. Its first command change followed the observed opening reversal and its tenth near landing occurred at frame 349. All 36 development tapes passed physical replay. The independent M/N comparison freezes the rate blend before playing any M/N release. [Development tapes](rally-improve-phase-development.json.gz) · [Twenty-four-return development video](rally-improve-phase-development.mp4) · [Video verification](rally-improve-phase-development.json)

| Fresh M/N outcome across 24 launches | Broad | Rate blend | Delayed RGB | Hold |
| --- | ---: | ---: | ---: | ---: |
| First near return | 13 | 13 | 0 | 0 |
| Second near return | 6 | 7 | 0 | 0 |
| Fifth near return | 1 | 2 | 0 | 0 |
| Total near returns | 26 | 33 | 0 | 0 |
| Points with at least ten alternating returns | 1 | 2 | 0 | 0 |
| Longest alternating exchange | 12 | 11 | 0 | 0 |

The frozen rate blend reached six near returns on seed `1776254840` and five on `1782030410`, where the broad fit scored one each. The broad fit reached six on seed `96293307`, where the rate blend reached three. The twelve-near development result did not recur on this cohort. All 96 M/N tapes passed strict replay. [M/N score tapes](rally-improve-phase-held.json.gz)

## Active practice and optical measurements

The scalar-only residual experiment compares mirrored changes to a 5×5 sensor-to-rate term. Its entire score is the number of legal near returns on three development starts, capped at eight per point. In the all-frame variant, the original residual received a score of six. After 45 actual practice points and 2,979 control frames, it remained the selection. The two live held audits reproduced the same trajectories for the original and unchanged practiced parameters. The rest of its proposed independent schedule remains unused. A second variant restricts smaller residual changes to visually detected continuation, protecting the independently issued opening. [Scalar reward code](../../scripts/experimental/tt24_rally_improve_practice.py) · [All-frame practice result](rally-improve-practice.json.gz)

The extent and interaction variants learn apparent size versus depth using paired RGB observations on the same demonstrated releases. A short causal track estimates velocity through partial occlusion. The interaction variant also reads the observed ball-to-blade displacement. A separate mixed-initiative collection charges practice in which the pupil starts the point and an explicitly privileged teacher takes over after its first legal landing. Any fits from those extra observations are assessed on completely separate starts. [Optical readout](../../scripts/experimental/tt24_rally_improve_optical.py) · [Recovery protocol](../../scripts/experimental/tt24_rally_improve_recovery.py)

The optical fit reconstructs 5,706 camera frames from the same 48 verified tapes and keeps 5,124 motor examples. Its size calibration uses visible paired ball observations. Across twelve E/F development launches, the broad, extent-only and interaction pupils reached a second near return on four, two and two points and reached a fifth on two, zero and two. Their longest alternating exchanges were ten, four and ten. The two interaction fifths happened on different releases than the two broad fifths. [Optical data](rally-improve-optical-demos.npz) · [Optical development tapes](rally-improve-optical-development.json.gz)

| Fresh K/L outcome across 24 launches | Broad | Optical interaction | Delayed RGB | Hold |
| --- | ---: | ---: | ---: | ---: |
| First near return | 12 | 12 | 0 | 0 |
| Second near return | 4 | 7 | 0 | 0 |
| Fifth near return | 0 | 3 | 0 | 0 |
| Total near returns | 18 | 33 | 0 | 0 |
| Points with at least ten alternating returns | 0 | 3 | 0 | 0 |
| Longest alternating exchange | 6 | 17 | 0 | 0 |

The interaction fit scored five near and five far returns on both `1035008827` and `2049605462`. It scored nine near and eight far returns on `1287979324`, where the broad fit scored one near return. That point's near landings were at frames 35, 66, 108, 143, 178, 211, 246, 281 and 317. The first fifth-return point in the frozen course was point 16 after 1,149 cumulative frames, or 38.3 seconds. All 96 physical tapes passed strict replay. [K/L assessed physical tapes](rally-improve-optical-held.json.gz) · [Seventeen-return held video](rally-improve-optical-held.mp4) · [Video verification](rally-improve-optical-held.json)

The corrective teacher took over on seven of the twelve development points. The other launches remained pupil-controlled because the first legal return never arrived. All twelve pupil prefixes reproduced the broad policy's near and far commands and their referee events through the switch or terminal stop. The extra exposure was 446 pupil frames and 1,486 teacher frames, including teaching that failed to score another return. The fit keeps 1,255 correction examples and retains the earlier 5,124 teacher examples. Evaluation uses the broad opening and phases into the corrected feedback law after the RGB-observed reversal. The teacher's longest intervention added 19 near returns on one point, which belongs to teacher practice. [Charged corrective tapes](rally-improve-recovery-practice.json.gz) · [Corrective fit data](rally-improve-recovery-demos.npz)

## Reproduce

Use the existing [table-tennis simulator and environment](../tt23/tt23_sequence.md#reproduce) and the [48-tape training bank](rally-bank.md#reproduce). The rate and phase candidates reuse the complete archived demonstrations without extra teacher interaction. The optical pass recalculates observations from the same physical tapes.

```sh
python -m scripts.experimental.tt24_rally_improve_trial --audit
python -m scripts.experimental.tt24_rally_improve_trial --workers 2
python -m scripts.experimental.tt24_rally_improve_trial --confirm --workers 1
python -m scripts.experimental.tt24_rally_improve_phase --workers 1
python -m scripts.experimental.tt24_rally_improve_phase --confirm --workers 2
python -m scripts.experimental.tt24_rally_improve_optical --workers 1
python -m scripts.experimental.tt24_rally_improve_optical --confirm --workers 1
python -m scripts.experimental.tt24_rally_improve_practice --workers 1
python -m scripts.experimental.tt24_rally_improve_practice --continuation --workers 1
python -m scripts.experimental.tt24_rally_improve_recovery --workers 1
python -m scripts.experimental.tt24_rally_improve_movie
python -m scripts.experimental.tt24_rally_improve_movie --source artifacts/tt24/rally-improve-phase-development.json.gz --mode rate_blend
python -m scripts.experimental.tt24_rally_improve_movie --source artifacts/tt24/rally-improve-optical-held.json.gz --mode interaction
```
