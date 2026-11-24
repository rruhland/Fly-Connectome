# Learning from missed table-tennis returns

The optional `--outcome-feedback learn` mode keeps recent legal returns and misses, paired with the model description of the stroke it actually issued. Candidates receive a penalty when the nearest miss is more similar than the nearest success. The description uses the camera-derived incoming trajectory and rollout estimates of landing position, net clearance, departure speed and distance from the blade center. The simulator referee supplies the outcome labels. `--outcome-feedback record` logs the same labels while leaving the model costs intact. [Experiment code](../../scripts/experimental/tt29_outcome_feedback.py) · [Commands, matched seeds and receipt](tt29_outcome_feedback.json)

| Matched course | Return cap | Control | Outcome feedback | Control / feedback CPU |
| --- | ---: | --- | --- | ---: |
| Stock physics, varied opponent targets | 32 | 22 / 2 / 8 / 6 / 4 / 4 | 22 / 2 / 8 / 6 / 4 / 4 | 103.3 / 108.2 s |
| Combined material, arm load, camera remount, command noise and delay | 24 | 4 / 2 / 3 / 2 / 6 / 2 | 4 / 2 / 6 / 6 / 6 / 4 | 48.9 / 116.7 s |
| Four repeats of 7345 with one opponent seed and a fixed stroke archive | 16 | 16 / 16 / 16 / 16 | 16 / 16 / 16 / 16 | 61.0 / 62.8 s |

The combined change used two registered cameras, a body fit, and the same two-tick timing estimate carried from solo practice in both runs. Feedback first changed a command at tick 96 on the second point. It changed the top-ranked native rollout in eight batches across the six points. Both courses continued updating the existing stroke archive and optical flight model, so effects could carry between points. The learner landed 14 returns versus 10 in the control. All six points still ended by six total returns, and CPU rose from 2.57 to 4.17 seconds per alternating return. [Compound case and attribution](tt29_outcome_feedback.json)

The varied-opponent courses produced identical command tapes on all six points. The learner recorded four attributable misses, all with paddle contact. The two terminal no-contact failures had no associated verified command tape. The capped repeats also produced identical tapes and supplied no misses to learn from. [Varied and repeat cases](tt29_outcome_feedback.json)
