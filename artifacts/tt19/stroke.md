# SO-101 learned strokes from RGB and score

This player builds a small library of approaches and sweeps. The command candidates move the physical shoulder, elbow and wrist. Each short movement probe measures the blade's response from RGB. From that catalogue the player tries motions in full points, varying the preparation, swing and the amount of time it leads the ball. During a point it chooses when to swing from the current ball and blade tracks. The controller gets these tracks, the serving turn, its own commands and the point reward. During practice it also uses changes in ball direction and observed table bounces. Physics contact events are kept in the evaluator and the replay receipt.

Both versions used 18 five-command movement probes and 64 scored practice points on a 0.848 × 1.60 m table. The gantry-informed version carried pitch, blade velocity and launch slope from eight receiving strokes and two serving strokes inferred from prior gantry images and positive point scores. The cold version used the same motion directions, search budget, seeded points, image tracker and scoring, with no gantry examples. The follow-through replay holds its last command target after one swing in a point. There is no further swing planned when the rival returns the ball.

## Initial games with the legacy end-line rule

| Player | Score | Legal serves | Legal returns | Scored receive points | Replayed points including practice |
|---|---:|---:|---:|---:|---:|
| Gantry-informed | 6–11 | 0/9 | 2 on 9 receive points | 2 | 82 |
| Cold | 5–11 | 0/9 | 1 on 8 receive points | 1 | 81 |

The two versions each made six legal practice serves in 32 attempts. The initial simulator awarded both players four service points when the scripted rival struck the ball beyond its end line before a required bounce. These were long-ball faults under the corrected rule. The displayed legacy scores reproduce the old rule. The one-point difference between the two players provides little evidence of an advantage from the gantry examples. [Gantry-informed receipt](stroke-gantry.json.gz) · [Cold receipt](stroke-cold.json.gz)

## Serve selection from visible placement under the legacy rule

A separate held game selected the serving stroke using the recorded practice images of bounces on both sides of the net. The player reused the trained SO-101 archive and took zero new movement probes or practice points. A real simulated wrist camera and two fixed cameras provided synchronized native RGB data, with fresh verification against the surveyed hall marks. The pilot used preview geometry for rendering the robot meshes with the same contact physics as the detailed model. Commands ran at 30 Hz and the observer received images at 15 Hz.

The gantry-informed player made five legal own serves in six attempts and two legal returns on seven receive points, losing 2–11. The cold player made four legal serves in five attempts and zero legal returns on six receive points, losing 0–11. On the common five own-serve starts, the controller chose the same motions in both games. The wrist had a verified pose in 289 of 375 captured frames for the gantry-informed game and contributed a ray to 171 ball triangulations. Five blade center triangulations included a wrist ray. These counts measure use within the combined camera rig. Wrist dependence would require scoring an otherwise matched game with that feed withheld. The placement games changed the selected serve, camera configuration and point schedule relative to the first test. [Gantry-informed placement replay](stroke-placement-wrist.json.gz) · [Cold placement replay](stroke-placement-cold.json.gz) · [Placement player checkpoint](stroke-placement-wrist-player.json.gz)

The same retained strokes played 18 additional points on a regulation 1.525 × 2.740 m table, with nine attempts in each role. They produced no legal serves, legal returns, or scored points. The result bounds the current one-swing library to the short fixture. [Regulation replay](stroke-regulation.json.gz)

## Receiving and rearming with corrected end-line scoring

The retained placement serve was frozen. The same controller practiced 48 more points exclusively as receiver under corrected scoring and recorded nine legal returns. Using 42 paired observations of the pan command and the blade's visible lateral displacement, it estimated 28.6 mm of lateral movement per cumulative command unit. The controller used that estimate for a bounded adjustment toward the next incoming ball. After a swing, the rearming condition used the integral of its own commands to recover the receiving preparation. It waited for an observed outgoing ball and then a new incoming approach before executing another sweep. All test policies used the same serving and receiving motions, timing estimates and lateral adjustment. All four games per condition had fresh starts, alternating service, a target score of 11 with a two-point lead and a 2.9-second point cap. [Receiver practice](stroke-receive-practice-strict.json.gz)

| Test condition | Four scores | Legal serves | Legal returns | Legal serves followed by a legal return | Points with a planned second swing | Points with at least two legal returns |
|---|---|---:|---:|---:|---:|---:|
| Rearm, wrist and fixed cameras | 1–11, 2–11, 0–11, 2–11 | 15/24 | 8 | 2 | 16 | 0 |
| Single swing, wrist and fixed cameras | 0–11, 2–11, 0–11, 2–11 | 12/22 | 5 | 0 | 0 | 0 |
| Rearm, fixed cameras | 0–11, 2–11, 0–11, 2–11 | 14/23 | 7 | 1 | 15 | 0 |

The four-game tests yielded zero wins in every condition. Rearming let the player return the ball after its own legal serve. No point contained two legal learner returns. The rearming policy with all three cameras had a verified wrist pose in 1,025 of its 1,382 captured frames, with a wrist ray used in 539 ball and 28 blade-center triangulations. Fixed cameras alone supported seven legal returns, including one after a legal serve, on the same game-start schedule. The wrist contributes measurements, and these games provide limited evidence of a score benefit. All 147 held point tapes replayed. [Rearming games](stroke-strict-rearm.json.gz) · [Rearming player](stroke-strict-rearm-player.json.gz) · [Single-swing comparison](stroke-strict-single.json.gz) · [Fixed-camera comparison](stroke-strict-fixed.json.gz)

Every point in the receipts includes its sequence of commanded and applied rates, score, evaluator contact events and chronological optical tracks. The track columns give the ball position, blade position and blade pitch. Positions are in meters from the tabletop center and the ticks are 1/30 s apart. Missing sightings and skipped captures are explicit. All action tapes replayed through MuJoCo. The gantry checkpoint is carried in memory through training and the SO-101 archive is appended separately, so the gantry controller remains available for a fresh return test. [Gantry-informed player checkpoint](stroke-gantry-player.json.gz)

```bash
python -m scripts.experimental.tt19_stroke_trial artifacts/tt19/stroke-gantry.json.gz gantry --practice 64 --test 45 --game --width 256 --height 192
python -m scripts.experimental.tt19_stroke_trial artifacts/tt19/stroke-cold.json.gz cold --practice 64 --test 45 --game --width 256 --height 192
python -m scripts.experimental.tt19_stroke_trial artifacts/tt19/stroke-receive-practice-strict.json.gz gantry --resume artifacts/tt19/stroke-placement-wrist-player.json.gz --practice 48 --test 0 --receive-only --rearm --strict-endline --seed 37642519 --width 256 --height 192
python -m scripts.experimental.tt19_stroke_trial artifacts/tt19/stroke-strict-rearm.json.gz gantry --resume artifacts/tt19/stroke-receive-practice-strict-player.json.gz --pan-from artifacts/tt19/stroke-receive-practice-strict.json.gz --strict-endline --practice 0 --game --games 4 --test 48 --seed 39924471 --cameras wrist+fixed --width 320 --height 240 --rearm
python -m pytest -q tests/test_tt19_stroke.py
```
