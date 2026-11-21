# Reacquiring the ball after a physical reply

The comparison used the unmarked clamp SO-101, one native wrist RGB stream at 480 × 360 and 30 Hz, the strict referee, and the frozen TT20 factory opener. A return after the opening required a real opponent strike and a legal bounce on the learner’s half. The trial counted a second learner return only when the physical referee recorded a second legal landing on the opponent’s half.

## Where TT20 rallies stopped

The audit combined the archived single-eye camera-carry games. Points with the same eye, seed, opponent contact position and first legal landing were counted once. The old planning depth covers −0.81 to −0.24 m. A follow-up motor command exceeded .025 of rated motion after the fifth frame following the opponent’s strike.

| Camera | Legal opponent replies | At least two sightings in planning depth | No sighting in planning depth | No follow-up motor command | Next learner contacts | Second legal returns |
|---|---:|---:|---:|---:|---:|---:|
| Outside | 25 | 25 | 0 | 16 | 9 | 1 |
| Wrist | 23 | 10 | 7 | 20 | 2 | 0 |

Eighteen of the 23 wrist replies bounced over 0.20 m from the centerline, compared with 10 of 25 outside. On three chosen wrist development replies, the ordinary wrist estimator left 19 gaps that the continuation could fill. These fills had a median error of 76 mm at four frames since the last source sighting. An evaluator-only check ran the fixed-catalogue ranker on exact ball states at 14 pre-contact frames. It produced seven predicted contacts and no predicted legal return. The archived outside histories show that visible approach alone often failed to elicit a stroke. [Source games](../tt20/camera-carry.md) · [Archive counts and hashes](reacquire-archive.json) · [Development action tapes](reacquire-wrist-development.json.gz)

## Fresh points against the strong rival

After observing a prior stroke travel out, the receiver looks for a persistent inward reversal. It fits the returning arc from that same camera’s registered ball bearings. The flight and bounce model can supply up to eight missing control frames and yields when a live sighting disagrees. A qualified incoming arc can also open planning sooner. The dynamic option solves wrist poses and motor tapes in the declared CAD arm for possible blocks behind the end line, then ranks the tapes through the dry contact solver. The player receives images, known geometry, its own commands and the point clock. World states in the receipts were sampled after decisions for evaluation.

| Receiver | First legal returns | Legal opponent replies | Points with a second plan | Second legal returns | Points won |
|---|---:|---:|---:|---:|---:|
| Frozen factory | 6/14 | 4/14 | 1/14 | 0/14 | 2/14 |
| Reacquisition and earlier CAD planning | 6/14 | 4/14 | 1/14 | 0/14 | 2/14 |

The new receiver supplied two missing positions with median error 5.5 mm. It moved the sole second plan from tick 48 to tick 44. Both variants contacted the reply and landed short on the learner’s half. The accepted earlier stroke came from the existing catalogue. The inverse-kinematic block search supplied no selected stroke in this sample. The scorer reproduced all 28 full point tapes and event traces on replay. [Held wrist receipt](reacquire-wrist-block.json.gz)

## Physical coaching pilot

Eight distinct wrist starts used the existing centered launch course and the cooperative opponent. The launched ball still had to traverse the table and the rival drove its own joints to strike. Both wrist policies made five first returns and received four legal coach replies. Both made zero second plans and zero second legal returns. The continuation filled nine missing positions with a median error of 103 mm at three frames since the last sighting. The four coach replies reached the usual interception depth at lateral offsets of approximately 0.24–0.40 m. None crossed the pilot’s central window of |x| < 0.14 m, −0.99 < y < −0.77 m and 0.07 < z < 0.32 m after a legal bounce. These points supplied too few usable examples to train a reply selector. All 16 physical point tapes reproduced the recorded event traces and results. [Coaching pilot receipt](reacquire-wrist-block-coach.json.gz)

A narrower teaching distribution would require the physical teacher to return through |x| ≤ 0.12 m near y = −0.90 m, at 0.10–0.30 m above the table and 1.8–3.0 m/s inbound. Accept only physical replies with a legal bounce and an independently verified feasible counterstroke. Centering the first learner return as well gives the teacher a better angle to deliver that distribution. The current wrist learner still needs an early visible reversal and at least several preparation frames before contact.
