# Anticipating where to wait for the next stroke

[Watch the 9.8-second comparison](ready-position-rally.mp4) · [Live camera, control and scoring checks](ready-position-rally.json)

The learner fitted the lateral position of the next incoming ball from 15 previously recorded real exchanges, using the RGB tracks from two fixed cameras. Its teaching target was a later optical observation projected to the strike depth. The simulation's true ball positions, opponent intentions, and referee labels were excluded. It used a three-coefficient ridge fit with a bias. One version predicted from the outgoing stroke. A second waited for the first few frames of the opponent's actual reply and predicted from its observed track. [Optical training samples and controller](../../scripts/experimental/tt21_ready_position.py) · [Original practice exposures](recovery-practice.json.gz)

On leave-one-point-out evaluation of those 15 tracks, median lateral error was 94.8 mm from the outgoing ball, 12.9 mm from the incoming ball, and 220.1 mm for always predicting the center. The body physically panned the shoulder only if the predicted offset exceeded 105 mm. Its existing stroke shortlist was rotated about the shoulder; actual candidate ranking used the separate nominal arm driven by its issued controls. Both versions kept the original receiving and serving software on the opening.

| On the 15 practice openings | All legal learner returns | Legal answers after a rival return | Points with two learner returns | Points won |
|---|---:|---:|---:|---:|
| Stay centered | 11 | 3 | 2 | 1 |
| Anticipate from outgoing ball | 10 | 2 | 1 | 1 |
| Position from early incoming ball | 12 | 4 | 2 | 1 |

Waiting for the optical reply preserved the two multiple-return points and gained a continuation after a learner serve. Moving early on a mistaken outgoing prediction cost a second return. We advanced only the incoming cue to fresh starts. [Outgoing-cue tapes](ready-position-development.json.gz) · [Incoming-cue tapes](ready-position-incoming-development.json.gz)

The fresh comparison used 32 unselected points with alternating service roles, two unmarked fixed cameras and the stronger physical rival. It began with ordinary resets. The same starts and scorer were used for centered and shifted play. Where the learned motor offset was exactly zero, both deterministic programs followed the same optical and motor path and each branch separately replayed the physical tape. [Fresh comparison](ready-position-incoming-held.json.gz)

On 32 fresh starts, positioning produced 19 legal returns versus 16 when centered, 5 legal answers after a rival return versus 2, and 12 wins versus 9. Each arm made 14/16 opening returns on rival service and 13/16 legal own serves. Points containing two legal learner returns rose from two to three. The learner chose a nonzero shift on 18 starts. All 64 scored branches reproduced their referee events. The extra answers include continuations after own serve; two legal learner returns in one point is a stricter measure.

## What required target experience?

A stronger control took the same RGB cues and panned the same physical shoulder. It predicted the incoming location by extending the observed lateral slope in a straight line, with zero target examples. On the same 32 starts it landed 4 continuations and won 11 points (centered 2 and 9, fitted 5 and 12). The fitted and straight-line policies produced different motor commands on three starts. Only seed `1132089956` changed the score: the fitted offset answered the reply after a learner serve and won. The other two gains over center, including the point with two learner returns, also worked with the zero-practice extrapolator. On the 15 teaching examples the untrained optical predictor's median error was 14.3 mm compared with 12.9 mm for the leave-one-out fit. Much of the benefit on this table comes from preparing with the observed trajectory. [Untrained optical predictor and 32 replays](ready-position-prior-held.json.gz)

## Continuing to learn during matches

Two disjoint full games compared staying centered, using the frozen fit and continuing to update the fit after each point with a later observed optical crossing. The continuously learning branch began with 15 examples, left the first game with 22 and loaded that checkpoint for its next game. It ended the second with 32. No game action used that point's later observation. All three branches lost 2–11 and 4–11. They each landed six legal returns in each game, with one multiple-return point in the first game and none in the second. The two prepared branches made five and ten nonzero shoulder shifts respectively across the two schedules. The three-point gain in the 32-start screen did not translate to a match gain here. [First online match](ready-game-online-1427016321.json.gz) · [Continued checkpoint and second game](ready-game-online-541340360.json.gz) · [Centered control](ready-game-center-541340360.json.gz) · [Frozen fit](ready-game-frozen-541340360.json.gz)
