# From ordinary Pong to a 3D racket

The earlier button experiment received a ball detector, named paddles, colored button identities, marking conventions, a reflection equation and candidate shots. It learned which predictions and shots worked, how joints moved, and which switches controlled the paddle. The new experiment asks whether a learner can discover which visible objects matter and which relation between them pays.

## Evidence to collect

- Practice in native Pong from images, actions and score. Save numeric learning state. Do not transfer scene track IDs, colors, a camera-axis labeling, an old motor inverse, a signed paddle command or object names.
- Expose the receiver to images of a genuinely three-dimensional table, arm and opponent. Audit joint commands through contact, net clearance and landing. Evaluate scoring, physical returns, repeated returns, early costs and wall-clock efficiency. Include same-rule learners without prior practice and with disrupted source associations, stationary/random controls and a moving visual distractor.
- Use a fresh cohort, count opening misses, and track first-eight-point behavior. Report when transfer has a useful effect and when receiving-domain practice catches up. Keep decisive versus inconclusive outcomes visible. Check for a return on a range of x and z coordinates. Expose any simplifications to striking and serving.
- Change only the publicly paid outcome on an independent cohort and check whether a numeric relation found in Pong can be rejected. The learner must never receive a task flag selecting the desired behavior. A policy that blindly tracks a single object cannot show task discovery.
- Take a copy for paired Pong evaluation before table tennis, then evaluate the actual state that emerged from table practice. Require evidence of genuine pre-training including earned points and later returns. Report the same metrics after table tennis, relative to pre-practice results. Investigate interference before claiming retention.
- Separately run the previous button player with physical button travel and camera delay. Keep these results distinct from table tennis and from physical hardware.

Generic connected-region tracking, action-effect estimation, model fitting, candidate relational behaviors and exploration are permissible inductive biases. Document every supplied prior that materially narrows what the agent can discover. A general learner for arbitrary visual environments is a longer-term goal.

## Confirmation after removing game cues

Run the full source search with the generically fitted reward-time classifier and its 60 random-action native episodes. Refine the spatial proposals using native score alone. Give the receiving player the fitted visual clock, fresh joint identification, and unrestricted visual meeting times. The late timing objective also includes observed continuation. Compare with the same clock and reset roles/geometry, plus retained roles with reset geometry. Use eight new 20-point courses beginning at seed 93714422; use match zero's actual post-table state for 48 paired five-point Pong matches from seed 81752634. These seeds were set before running that qualification. Record all first misses and later opportunities. The opening and repeated-return effects, plus the earlier native retention gates, are the intended checks.

For persistence, re-create the first new table course, verify a complete point-boundary save/reload on the next fresh serve, and extend that same course from 20 to 64 points. Recheck the resulting native Pong state on the same 240 points. This addresses the deterioration previously measured after 48 points with a single shared utility estimate.
