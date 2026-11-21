# Physical rally rival

The rival uses the simulated far-side robot and the strict compact-court referee. After each learner strike it uses actual ball and joint state in a separate prediction copy to find where the ball can be reached after a legal bounce. It searches joint-rate-limited swings over several depths and verifies candidate contacts and landings against the same referee. The live point continues from its original release. Each side gets legal-return credit only after a physical paddle contact and a far-side bounce. The cap is 1,400 control frames, or 46.7 simulated seconds. [Rival implementation](../../scripts/experimental/tt22_opponent.py)

The stock option adds this controller to the existing four-axis rival. The steerable option also drives the existing far wrist's yaw joint through a real fifth actuator and moves its flange 4 cm toward the net. The learner, court, ball and scoring stay common to both robot configurations. The fifth joint lets the far arm steer the blade independently of base yaw. The central target is near `(0, −0.91, 0.19)` m in tabletop coordinates at 2.2 m/s. The varied mode rotates through five lateral, depth, height and pace targets within the available stroke region. [Trial and replay](../../scripts/experimental/tt22_opponent_trial.py)

The archived opening assay reuses eight authenticated first learner strikes, balanced by service role. After each first strike the learner holds its motor requests. It measures legal rival replies and the delivered incoming ball at the learner strike plane. The native assay uses the same fixed optical learner across rival modes, two fixed native RGB views, and four shared, fresh seeds from a narrower opening distribution. It records both arms' actual rate requests. An independent replay applies both tapes directly and checks the terminal score and all referee events.

| Rival | Legal replies on 8 shared openings | Median absolute lateral offset at strike plane | Forward speed range at strike plane |
| --- | ---: | ---: | ---: |
| TT21 cooperative coach | 7 | 22.8 cm | 1.76–3.38 m/s |
| TT22 planner, stock four-axis robot | 7 | 22.6 cm | 1.81–2.73 m/s |
| TT22 planner, actively steered wrist | 8 | 3.7 cm | 2.18–2.50 m/s |

The sample nearest y = −0.91 m for the steerable rival was between 5.0 cm left and 10.7 cm right, with heights of 13.2–22.3 cm above the tabletop. The new body and controller therefore materially tightened feed placement while preserving legal scoring. [Archived openings and replays](tt22_opponent-archived.json.gz)

The steerable varied mode also legally returned all eight openings. At the strike plane those feeds covered 9.5 cm left to 12.8 cm right, 15.4–21.5 cm above the table and 1.84–2.57 m/s. All eight reproduced under two-tape replay. [Varied replies](tt22_opponent-varied.json.gz)

| Rival | Alternating legal returns on the four shared native starts | Legal returns, learner / rival | Ten-return rallies |
| --- | --- | ---: | ---: |
| TT21 cooperative coach | 3, 1, 0, 4 | 5 / 3 | 0 |
| TT22 planner, stock four-axis robot | 2, 6, 0, 4 | 6 / 6 | 0 |
| TT22 planner, actively steered wrist | 4, 2, 0, 10 | 8 / 8 | 1 |

Seed `1690317037` produced ten strict alternating returns with the steerable rival and fixed-stereo learner. The two robots made eleven actual paddle contacts across 211 ticks. Both landed five legal returns. The final learner strike landed on its own side. The first serve and entire rally remain in the receipt. The 211-tick point exceeds the former 190-frame limit. The new rival spent 29.2 seconds planning six shots in this host run. [Native rally tapes](tt22_opponent-native.json.gz)

## Learning a feed target from earlier rallies

The learner's previous fixed-stereo traces supplied 42 legal rival reply episodes from nine point seeds. Twenty-eight led to a legal learner continuation. The first reliable optical motion estimate in a fixed approach band yielded anticipated position and pace at four candidate strike depths. Forty-one episodes supplied a feature. A seed-balanced local model fit subsequent legal credit and selected its bandwidth by leaving out whole seeds. That grouped check gave a Brier score of 0.243, compared with 0.257 for a seed-balanced constant outcome rate. The fit selected `(0, −0.90, 0.20)` m at 2.4 m/s from 100 candidate targets. [Learning and scoring code](../../scripts/experimental/tt22_opponent_learn.py)

The fixed and learned profiles both use the same steerable far robot, controller, and frozen native-stereo learner. The only change is the far planner's requested depth, height, and pace. Six fresh starts use the earlier release generator with nominal spread scaled to 0.55 and pace to 0.90. The four-start comparison used 0.30 and 0.82. The matched schedule includes every draw, and each point runs uninterrupted after its single release. [Fit, held schedule and replay tapes](tt22_opponent-learned.json.gz)

| Feed profile | Legal alternating returns across six held points | Legal returns, learner / rival | Real paddle contacts | Ten-return points | Learner points won |
| --- | --- | ---: | ---: | ---: | ---: |
| Fixed center, −0.91 m depth, 0.19 m height, 2.2 m/s | 4, 2, 10, 6, 10, 7 | 20 / 19 | 44 | 2/6 | 1/6 |
| Learned center, −0.90 m depth, 0.20 m height, 2.4 m/s | 4, 2, 10, 10, 10, 4 | 20 / 20 | 46 | 3/6 | 0/6 |

Both treatments reproduced the exact first learner contact and opening actions on all six matched starts. The learned profile changed a six-return point into a ten-return point and shortened the final seven-return point to four. Total learner return credit stayed at 20. All twelve point outcomes and referee traces independently replayed from the two motor tapes.

## Incoming flight phase

Height alone leaves the timing of the receiving stroke unspecified. In the archive, 3 of 8 feeds with an optically forecast vertical velocity below −0.7 m/s led to a legal learner return, compared with 21 of 29 rising feeds at the same reference depth. Adding vertical and lateral velocities changed the grouped same-depth Brier score from 0.242 to 0.240. The model selected a +0.8 m/s upward crossing with zero lateral drift. The physical planner penalizes both velocity errors in its dry ranking and exact confirmation while keeping the original central target and 2.2 m/s forward pace. This comparison uses a separate unfiltered schedule at the same wider jitter. [Flight phase fit and replay tapes](tt22_opponent-phase.json.gz)

| Feed profile | Legal alternating returns across four held points | Legal returns, learner / rival | Real paddle contacts | Mean absolute error to +0.8 m/s upward speed |
| --- | --- | ---: | ---: | ---: |
| Fixed target | 10, 2, 4, 4 | 10 / 10 | 24 | 1.06 m/s (9 samples) |
| Learned upward phase | 8, 2, 6, 4 | 10 / 10 | 24 | 0.61 m/s (7 samples) |

The vertical error uses evaluator-only post-bounce positions and velocities sampled nearest the learner strike plane. Mean absolute lateral speed moved from 0.08 to 0.14 m/s on those samples. Both arms landed the same total number of legal returns. All four openings and first learner strikes matched exactly, and all eight points replayed from the two physical command tapes.

## Running the comparison

Use `install_rally_opponent(arena, lane='anchor')` for the existing TT20 arena, with `max_steps=RALLY_STEPS`. Use `RallyArena` from the same module for the steered model. Choose `lane='varied'` for the rotating feed targets.

```bash
python -m scripts.experimental.tt22_opponent_trial --suite archived --output /tmp/tt22-opponent-openings.json.gz
python -m scripts.experimental.tt22_opponent_trial --suite native --output /tmp/tt22-opponent-native.json.gz
python -m scripts.experimental.tt22_opponent_learn --output /tmp/tt22-opponent-learned.json.gz
python -m scripts.experimental.tt22_opponent_learn --phase --output /tmp/tt22-opponent-phase.json.gz
```
