# Later-stroke practice on the learner’s own rallies

Adding outcomes at cues reached by the frozen learner reduced the number of held points with a second legal near return from four of eight to one of eight. Both policies made all eight first returns. In each pair, commands and referee events agree before the first later-rally decision, and the stereo observation at that decision agrees as well. [Frozen baseline](branch-motor-student-baseline.json.gz) · [Expanded practice](branch-motor-student-held.json.gz)

The learner collected 12 points on schedule `402687021`, using the teacher-trained policy with the signed Pong lateral coefficients and the wider .55 spread and .90 pace release family. Both arms’ command tapes reproduced each point’s full referee event sequence. Nine points supplied later practice cues. The original opening head stayed fixed. Each added cue followed a physical far-arm hit, and practice reconstructed the cue from the recorded stereo positions and commands before trying each feasible stroke with the real colliders and strict referee. [Collection](branch-motor-student-native.json.gz) · [Original practice](branch-motor-practice.json.gz) · [Expanded checkpoint](branch-motor-student-practice.json.gz)

| Added practice | Count |
| --- | ---: |
| Native collection frames | 1,102 |
| Later cues with a reachable stroke | 17 |
| Later cues with a strictly legal stroke | 14 |
| Evaluated motor alternatives | 1,764 |
| Alternatives with physical blade contact | 1,693 |
| Alternatives with a strictly legal return | 360 |
| Counterfactual control frames | 36,943 |

The alternative fit a kernel ridge surface for each stroke from the same expanded practice. It estimates a residual from the stroke’s stage prior using a Gaussian kernel over the existing optical and commanded-pose features. The squared-loss systems use only the cues where the stroke was physically attempted. The frozen first-return head, stroke menu, command generator, source gains and placement credit remain common. Grouped practice validation with each donor held out scored 37 of 62 feasible selections for the ridge score and 30 of 62 for the local score. The kernel width and ridge constant were chosen using this archive, so the grouped result is a model-selection diagnostic. [Donor validation](tt23-motor-surface-validation.json.gz)

The native point comparison uses schedule `1731922005`, disjoint from the original donors and new collection. Each condition uses both live motors from release through referee termination. The ordinary game uses independent schedule `853884001`, standard opening pace and spread, and the shared optical toss-adjusted near serve. Both sides alternate service every two points, switching each point at deuce. A later stroke is selected only after a stereo trajectory is available. [Match runner](../../scripts/experimental/tt23_motor_game.py)

| Eight paired native points | First legal | Second legal | Third legal | Near legal returns | Alternating legal returns | Longest rally | Point wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Original local score | 8 | 4 | 1 | 13 | 26 | 6 | 1 |
| Expanded local score | 8 | 1 | 0 | 9 | 18 | 4 | 0 |
| Expanded ridge score | 8 | 5 | 1 | 14 | 28 | 6 | 0 |

The ridge score produced a second near return on every point where the original local score did, plus one more. Its six-return rally occurred on a different point. All three policies used identical commands and observations through the first near return and received the same next ball. Both command tapes and the full referee sequence independently replayed for all 24 points. [Ridge points](tt23-motor-surface-held.json.gz)

| Ordinary first-to-11 game | Final score, near–far | Near / far legal returns | Longest rally in game | Near legal returns in shared 11 starts | Longest rally in shared 11 starts |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original local score | 0–11 | 7 / 11 | 7 | 7 | 7 |
| Expanded local score | 2–11 | 13 / 17 | 10 | 6 | 7 |
| Expanded ridge score | 0–11 | 8 / 12 | 9 | 8 | 9 |

On the same near serve and point start, the rallies lasted seven alternating returns with the original score, one with the expanded local score and nine with the expanded ridge score. Both arms’ commands agree before the first learner decision at frame 52, and the stereo observation at that frame agrees. The expanded local score’s ten-return rally came on point 13, after the other games had ended. In the 11 starts common to all policies, each returned two of six far serves legally. All 35 game points replayed with the same terminal outcomes, event sequences and scores. [Original game](tt23-motor-game-baseline.json.gz) · [Expanded local game](tt23-motor-game-student.json.gz) · [Expanded ridge game](tt23-motor-game-ridge.json.gz)

Run `python -m scripts.experimental.tt22_branch_motor_student` to record the 12 student points, extend practice and evaluate the held schedule. Use `--baseline-only` for the paired teacher-trained evaluation. Run `python -m scripts.experimental.tt23_motor_surface` to regenerate the ridge validation and held evaluation. Reproduce the ordinary games using the project table-tennis environment and the existing branch-practice checkpoints.

```sh
python -m scripts.experimental.tt23_motor_game
python -m scripts.experimental.tt23_motor_game \
  --checkpoint artifacts/tt22/branch-motor-student-practice.json.gz \
  --output artifacts/tt22/tt23-motor-game-student.json.gz --label student
python -m scripts.experimental.tt23_motor_surface \
  --game artifacts/tt22/tt23-motor-game-ridge.json.gz
```
