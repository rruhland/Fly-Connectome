# Learning paddle orientation by watching real simulated impacts

The SO-101 bounced balls with its existing wrist-camera controller while a calibrated fixed camera pair observed the practice. In each return, the change in the ball's flight direction gives information about which way the blade faces. Rotating this change into the last arm link's coordinate system lets the learner carry a tool estimate into a game in which it holds the paddle at a different angle. The method uses an explicit ball detector, calibrated cameras, known robot link geometry, and an engineered impact detector. It learns the change in the tool orientation from observations. [Implementation](../../scripts/experimental/tt26_bounce_optics.py)

The estimator fitted smooth incoming and outgoing trajectories to visible positions around a velocity reversal close to the paddle. It estimated free-flight acceleration from the images. It used encoder angles to express the velocity change in the local gripper frame and compared the result with experience on the nominal grip. Each of three changed tools was practiced for 16 drops against the same 16 nominal drops. Relative face-angle errors were **0.80°, 0.62° and 0.32°**, respectively, when simulator geometry was revealed after fitting. Deterministic replay placed all 63 optically detected first impacts within half a control tick of physical contact. The learner can infer a face direction. Rotation about that face normal is ambiguous. [Fitted state and practice observations](optical-face.json.gz) · [Physical checks](optical-face-held.json.gz)

## Carrying the same estimate into both games

The player used its existing kinematic calibration to find joint postures that approximately preserve the blade's old location and face direction under the altered grip. This is a generic tool-pose adjustment. Neither the solve nor the camera policy predicts or plans a table-tennis stroke. The SO-101 issued bounded joint rates during play. A control inverted the learned rotation, and a second left the original target unchanged. All had the same earlier six receiving demonstrations.

| Physical mounting change | Fresh beginner returns without new solo practice | With the observed tool change | With the direction reversed | Wrist drops with at least ten rebounds, learned / cold |
|---|---:|---:|---:|---:|
| +0.22 rad about the handle | 16/48 | 28/48 | 0/48 | 8/12 vs 0/12 |
| −0.22 rad about the handle | 5/48 | 26/48 | 0/48 | 12/12 vs 0/12 |
| +0.26 rad about a different axis | 12/48 | 17/48 | 3/48 | 0/12 vs 0/12 |
| Unchanged nominal grip | 24/48 | 24/48 | 24/48 | — |

All conditions used the same 48 distinct perturbed openings. There were zero new practice shots at the table before this test. On 48 more difficult original nearby recordings the corresponding totals across the three altered grips were 9/144 optical versus 1/144 cold. One of 32 unrestricted recordings was returned on the positive handle twist and none on the other two. In solo play the eight-second cap right-censors long streaks. Correcting the inferred face direction alone, without changing the motor target, produced 18 rebounds versus 19 cold on a separate short set. Targeting and sensing must use the learned tool correction together. [Receiving outcomes and corrected targets](optical-transfer.json.gz) · [Wrist-only returns and exact physical replays](optical-bouncing.json.gz)

The canted paddle exposed a different control problem. The correction initially tried to preserve both blade location and angle. On six practice releases the learner tried three degrees of positional flexibility. Rebound rewards favored the option that allowed the blade to shift further while restoring its attitude. On 12 fresh wrist releases this raised median streak **from 1 to 18**, with at least ten bounces on 11/12. Carrying that exact tolerance into a new table receiving set reduced the score to **16/48**, versus 20/48 with the original tolerance and 19/48 cold. A common estimate of the tool helped across tasks. The two tasks favored different tolerances for positional error. [Source-selected tradeoff and paired target outcomes](canted-tool.json.gz)

## How much source practice was necessary?

For a more natural training schedule we removed paired releases. Eight old nominal drops supplied a reference. One, four, or eight successive altered-grip drops supplied the new evidence. Unlike the initial fit, the acceleration estimate was frozen from the nominal subset before the changed drops. We repeated the test with the nominal and altered seed halves exchanged. The learner had no target-table reward, hidden contact time, new-grip angle or simulator ball state.

| Altered-grip drops before playing | Legal returns in the 192 paired target evaluations |
|---:|---:|
| 0 | 62 |
| 1 | 97 |
| 4 | 99 |
| 8 | 98 |

The same 32 distinct held beginner variations were evaluated under each of the three changed mountings and two independent source/nominal split assignments. One observation was useful on this problem. Additional source samples gave a similar average. These counts measure adaptation of an established skill to a changed tool. [Per-drop fit, source seeds and physical target outcomes](few-drops.json.gz)

## Wider coverage and the next return

A fresh confirmatory recorded roster excluded all 192 prior reach-screening shots and all nearby and unrestricted TT25 test IDs. We drew 72 unseen recordings from the next nearest available neighborhoods and 64 unrestricted ones. With the original six-posture motor repertoire, the optical learner returned only **one distinct unrestricted shot**, #8402, under both handle twists. The old uncorrected receiver returned none of the 64 on the three changed grips. Only one of the 72 nearby recordings was returned using optical compensation, and a cold canted-grip configuration returned the same one. The return-rate advantage on beginner feeds does not establish broad recorded match coverage. [Fresh recorded outcomes and independent replays](fresh-recordings.json.gz)

Once an opponent returned a first block, a separate simulation teacher searched for usable direct joint motion. It found nine legal second returns on 14 development variants in 629 physical attempts. A stereo-driven pupil returned a second ball in **one of 14 held reply opportunities**, and the opponent returned it again. That point was a fresh perturbation of a donor trajectory used in practice. It replayed the first-return commands before the camera-selected continuation took control. [Development and held rally evidence](match-continuation.md) · [Watch the matched points](bounce-to-table.mp4)

To broaden the existing range of movements we admitted 13 additional postures after actual legal returns on 64 separate practice feeds. On the same 136 fresh, unaltered recordings, the larger library raised nominal returns from one to four and optically adjusted positive-twist returns from one to five. Both reached 2/64 on the unrestricted subset. These improvements cost 1,238 extra physical practice shots and 13 successful replays. The learner still has little coverage on the full incoming distribution. [Broader library, exact results and checkpoint](match-broader.md)

With the 19-posture library fixed on the same positive-twist grip, dropping the bounce-derived optical correction reduced legal returns from **5/136 to 1/136**. This comparison changes only the shared tool estimate at deployment. [Matched expanded-bank control](expanded-no-bounce.json.gz)
