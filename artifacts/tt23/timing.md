# Does Pong experience tell the SO-101 when to start its stroke?

Acting on the first eligible stroke produced the longest physical rallies in this comparison. The Pong-selected timing gate outperformed the permuted gate, while both deferred feasible openings on the two longest points.

The experiment carried two numbers selected by native Pong duration. Both games could defer a proposed movement until the remaining time was below a weighted sum of actuator travel and response lag. The other policies used the same proposal and motor controller. One used a gate chosen after permuting the Pong scores, and one accepted the first eligible proposal. The source and target draws, rule bank, and outcomes were fixed in advance. [Design](timing-design.md) · [Runner](../../scripts/experimental/tt23_timing.py)

## Native Pong

Eight rules played the same 16 launches under palette, rotation, reflection, and action-ID randomization. The longest mean capped duration selected `2.0 × travel + 0.5 × lag`. The fixed permutation selected `0.65 × travel + 0.5 × lag`. Each condition estimated the paddle's speed and response from the same rendered actuator probes and shared the existing lateral controller. The duration selector used elapsed ticks as its only outcome.

| Rule on 16 new native Pong starts | Mean duration | Episodes reaching 10 seconds | Paddle returns |
|---|---:|---:|---:|
| Source-selected | 4.69 s | 2 | 12 |
| Permuted | 3.34 s | 0 | 5 |
| Eager | 5.10 s | 0 | 13 |

The source-selected gate lasted longer than the permuted gate on seven launches, tied on eight, and ended sooner on one. Against eagerness it lasted longer on four launches, tied on seven, and ended sooner on five. An independent rerun of one launch per policy reproduced the native action digest and outcomes. [Source receipt](timing-pong.json.gz)

## Physical SO-101 points

All policies used the same RGB stereo observations, optical flight fit, factory motor search, correction from four prior arm practice points, stock serve and cooperative physical partner. The eligible time was the nominal collision time of the stock planner's chosen stroke. The body twin estimated the distance from its own joint positions to the commanded contact positions in full-rate control frames. A fixed probe on the nominal arm gave a median tracking lag of 0.038 control frames. The corresponding median Pong probe estimate was 8.54 camera frames.

| Rule on the same 12 starts | Arm legal returns | Partner legal returns | Total legal alternations | Points with at least 10 | Longest rally |
|---|---:|---:|---:|---:|---:|
| Source-selected | 6 | 12 | 18 | 0/12 | 4 |
| Permuted | 0 | 6 | 6 | 0/12 | 1 |
| Eager | 34 | 40 | 74 | 2/12 | 24 |

The learned timing gate extended four points relative to the permuted gate and tied eight. Relative to eager play it extended one point, tied five and shortened six. It deferred 23 proposals and changed the actual arm command on 11 starts. The permuted gate deferred 25 proposals and changed the actual arm command on all 12. Within each matched start, the optical positions and partner commands agreed up to the first changed learner command. Every arm served legally six times, and the partner won all 12 points in each condition. Independent replay reproduced the duration, strict referee events and score credit on all 36 points. [Full physical receipt](timing-live.json.gz)

On ordinary receive seed `1621105536`, the eager arm and partner made 12 legal returns each over 470 control frames. The source-selected and permuted gates both vetoed the opening plans at ticks 16 and 19 and made zero returns. The first plan had 8.19 nominal frames to contact and 2.20 full-rate frames of required travel. The later plan had 4.63 frames to contact and required 2.01 frames of travel. Every finite rule in the source bank would have deferred both plans. The eager arm lost this point when its final shot went to the floor. An independent replay of both motor streams reproduced the duration, referee events, score, and fixed stereo RGB digest. [Paired native command tapes](timing-live.json.gz)

The same opening failure occurred on receive seed `935904631`. Eager made 20 alternating legal returns, and both gates deferred two candidate strokes and made zero returns. For the later of those candidates, the ratio of nominal time to contact and minimum joint travel was 2.99. The largest multiplier available to the source selector was 2.0, and the arm lag allowance added less than 0.04 frames. A short predicted movement can still need an early commitment when a later decision falls outside a feasible stroke window. [Paired native command tapes](timing-live.json.gz)

Reproduce with `python -m scripts.experimental.tt23_timing both` in the repository's MuJoCo EGL environment. The command can also run `source` or resume the fixed `target` schedule independently from the frozen receipt.
