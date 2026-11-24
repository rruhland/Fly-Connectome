# Teaching the SO-101 to return shots on both sides

On a previously unseen opening, the table-clamped SO-101 landed seven consecutive shots across its body. Its contacts went 21 cm left, 24 right, 17 left, 24 right, 21 left, 24 right and 19 left of its shoulder. The physical opponent landed eight shots, making 15 legal returns in 11.6 simulated seconds. All seven arm strikes used one paddle face. The arm observed two external RGB cameras and its joint encoders. The video replays its motor commands at simulation speed. The full CAD audit found zero penetrations across 5,581 physics states and a minimum clearance of 18.3 mm between checked parts. [Watch the 15-return point](alternating-15.mp4) · [Contact and clearance measurements](alternating-15.json)

## What the previous rally practiced

On exact replay of the earlier 200-return cooperative point, all 100 legal near-arm strikes occurred 19.5–26.6 cm right of the shoulder and used the same physical paddle face. All 15 arm strikes in the wider 30-return point also used that face, 14.0–31.5 cm to the right. In this study an independent observer measured the signed alignment between the solver’s sphere-to-blade contact normal and the blade's cylinder axis. Absolute alignments below 0.8 count as rim contacts. Court side, blade face and legal landing are measured independently. [Cooperative point](previous-cooperative-faces.json) · [Wide point](previous-wide-faces.json) · [Side and face observer](../../scripts/experimental/tt31_contacts.py)

## Identical practice in two orders

The earlier player began with 223 rewarded movement sequences and 24 optical shot-error samples. Each of the 30 lessons supplied a physical incoming ball from a powered opponent. The staged player received eight fixed placements, ten alternating placements and twelve drawn from continuous left and right bands. The mixed player received the identical cards in shuffled order. Both carried their motor memory and error estimates across the lessons. The opponent alone knew the placement request. The near player could store movements after legal landings and measure shot errors from its own stereo observations, without backpropagation. Both players continued to use the calibrated cameras, ball detector, court geometry, stroke generator and simulation-based movement checks. Initial positions and joint targets were restored between single-ball lessons. A separate set of live games below trained consecutive transitions without restoring the arm inside a point. Prior robot-table-tennis research found that adjusting the shot distribution during training could elicit multiple stroke styles. Here we directly tested whether the order helped our learner. [Course design, exact matched deliveries and learning states](course.md) · [Google Robotics curriculum study](https://research.google/pubs/robotic-table-tennis-with-model-free-reinforcement-learning/)

| Assessment | Original checkpoint | After staged practice | After mixed practice |
| --- | ---: | ---: | ---: |
| Ten new two-band games, arm’s own legal returns | 7 | 9 | 10 |
| Eight new games including the center, arm’s own legal returns | 4 | 4 | 4 |
| Sixteen measured incoming releases, legal first returns | 6 | 6 | 6 |

Both schedules landed all 19 right-side practice balls and missed all 11 left-side balls without paddle contact. The gain in the two-band games came from one longer right-side rally. The full-interval opponent drew independent target coordinates across −28 to +28 cm, including the crossover, and also varied receiving depth, height and pace. Its subsequent shots physically depended on the arm’s preceding return. The separate receiving exam used actual measured DeepMind releases. It selected two balls in each of eight lateral bins using an unimpeded flight projection and a bounded height and speed range, before scoring a learner. It tests a reachable region of a regulation table. Neither benchmark represents full-width human competitive play. [Measured launches](measured-openings.json) · [Full-interval course](full-plan.json.gz) · [Results and per-game CPU](results.json)

## Why the old arm stayed still

On the 16 measured releases, the original player issued meaningful motor commands on all eight right-side balls and only two of eight left-side balls. A camera replay measured the unobstructed physical crossing of a plane 1.10 m on the near side. On the seven measurable left approaches, forecasts available 200 ms before crossing had median absolute lateral error of 8.1 mm. At 100 ms it was 1.6 mm. The remaining left release had no measurable crossing. Shifting remembered trajectories using the local arm Jacobian kept the same six legal returns and increased computation. Short camera-guided movements after a failed plan also failed to convert the two tested left opportunities. [Camera and command traces](measured-tracking.json) · [Spatially adjusted replay](measured-spatial.json.gz) · [Approach tests](approach.md)

We used the body model to screen fixed ready postures for clearance and joint travel. The posture calibrated toward the left gave three legal returns on four development cards, including both left offers. A centered candidate gave two, both on the right. Each candidate used the same starting posture for either side throughout a game. On the independent 16 measured releases, changing the starting posture increased left-side returns from 1/8 to 5/8 and reduced right-side returns from 5/8 to 2/8. After 16 additional lessons the new posture scored 4/8 left and 1/8 right on that exam. The default posture remains available. [Posture calibration](../../scripts/experimental/tt31_ready.py) · [Selected joint configuration](left-ready.json) · [Scores by release](results.json)

## What the short lessons transferred

From the new posture, both staged and mixed players landed 15 of 16 practice balls, including all six left offers. Each had four fixed, six alternating and six variable deliveries. With the complete planner enabled, the eight full-interval games still yielded four legal arm returns per player before and after practice. When only previously rewarded movements could be retrieved and verified, the unpracticed checkpoint landed zero and either practiced checkpoint landed one new left return. A learned movement replaced some of the expensive searches. This was a small, measurable transfer. [Staged checkpoint](ready-staged-checkpoint.json.gz) · [Mixed checkpoint](ready-mixed-checkpoint.json.gz) · [Frozen controls](results.json)

Four further live practice games exercised the recovery from one stroke into the next. After these, the trained checkpoint made four arm returns versus three in the first four held alternating games. On opening 12451 it retrieved a different first left stroke and then planned a legal right response. The earlier checkpoint failed to find a movement for the right reply. The first stroke changed the subsequent physical ball. In contrast, on the practiced point 6810 the original checkpoint could already return left, right and left. Its recorded switch establishes physical capability. [Watch the held left-to-right return](held-switch.mp4) · [Held contact audit](held-switch-replay.json) · [Transition study and exact decision traces](transition.md)

## Preparing for the next stroke

We tested a second change on the same four alternating games. After a candidate’s predicted contact, the planner waited 65 ms and tested moving the joints back toward the calibrated starting posture during the remainder of the stroke. It retained this change only if a new physics rollout preserved the predicted hit and legal landing. No target-side label or future opponent placement entered this computation. This recovery rule is engineered. The existing movement sequences and optical corrections are learned.

| Frozen controller, four alternating games | Arm’s legal returns | Longest point, both players |
| --- | ---: | ---: |
| New posture, original movement and error memories | 3 | 3 |
| New posture, after 16 lessons | 3 | 3 |
| Original memories with recovery | 4 | 5 |
| After 16 lessons with recovery | 6 | 9 |

All three additional returns over the practiced controller came on the same opening. The first stroke and far player's next contact matched. The commands first differed on tick 90, after the arm's first hit on tick 86. The same second ball reached the receive line 21.3 cm left of the shoulder. With recovery, the arm returned it and the next two. [Measured movement paths](recovery-path.png) · [Action tape](recovery-periodic.json.gz) · [Matched control](ready-periodic-staged.json.gz)

On a second set of four previously unused openings, practice and recovery each contributed to the result. The 16 lessons had no effect on the total until the controller could recover. With recovery, their gain was seven additional arm returns across two points. Two openings remained outright misses. The first opponent contact and table bounce matched for every controller on each opening. Later balls depend on how both players strike. The independent tests below used frozen checkpoints, with learning enabled during practice.

| Held test, arm’s own legal returns | Earlier memories | After 16 lessons | Earlier memories + recovery | After 16 lessons + recovery |
| --- | ---: | ---: | ---: | ---: |
| Four new alternating points | 2 | 2 | 4 | 11 |
| Eight continuously random points | 4 | 4 | 6 | 6 |

On the four new alternating openings, the final arm’s results were 0, 7, 4 and 0. It landed nine of eleven balls after its opening return across these games. Four additional live practice games without recovery left the original two returns unchanged. On the continuously random test, recovery added two legal returns in one point. The arm hit right, left and left before missing the next left. Its longest point reached seven legal returns between both players. Practice had no additional effect on that course’s scores. The eight practiced points used 122 process CPU seconds with recovery and 136 without. [Independent scores and matched deliveries](results.json) · [Alternating replay](alternating-15.mp4) · [Continuously varied replay](varied-game.mp4)

## Paddle management and retention

The wrist saw the first rebound and attempted to flip the paddle before the ball came back. With the coupled motor limits it completed zero opposite-face rebounds on eight high drops. The ball was airborne for a median 0.69 seconds, while the wrist needed 1.08 seconds to turn within 15° of the other face. The same releases with the older independent torque and speed limits yielded eight successes. The coupled motor made 24 rebounds holding one face on those eight drops. Its exploratory motions did teach a two-tick command delay that transferred to one selected group of delayed table games. [Motor and transfer evidence](flip-practice.md)

Both physical blade faces are usable in table play. A separate controller returned with the positive and negative faces in one rally, both to the right of the shoulder. A cold controller reproduced that point exactly, and the alternative controller shortened rallies overall. A forehand or backhand stroke classification would also require the arm's swing and body configuration. [Watch the face change](face-switch.mp4) · [Full control study](faces.md)

The familiar cooperative opening was retested to a cap of 24 alternating returns with the same partner seed. With the earlier posture and original memories it reached 24. With the new posture it scored zero both before and after the 16 staged lessons. Restoring the earlier posture while retaining the newly trained checkpoint restored all 24. This check shows that the learned movements remain available and that one fixed readiness configuration carries a performance cost. [Original](cooperative-original.json.gz) · [New posture](cooperative-trained.json.gz) · [Restored posture, new memories](cooperative-restored.json.gz)

## Next learning target

Keep the original cooperative posture as a retention comparison. Give the same learner short uninterrupted bouts where the opponent can repeat a side, cross the center or switch after any shot. Use what happens on the following ball to learn where and when to recover. Starting tests should compare that learned choice with always recovering to the same posture, on identical new openings and an equal practice budget. Score the arm’s legal returns after the first ball, coverage through the center and performance on the old cooperative opening. The staged and shuffled lesson results gave no consistent advantage to a fixed lesson order. The constrained flip drill has yet to teach a usable stroke on the coupled motors.
