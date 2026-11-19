# Rapid recovery from a hidden control change during Pong

The rapid controller passed every prospective criterion on 30 held-out starts. It returned 27 of 30 first balls after a concealed mid-rally change, up from 12 of 30 with the shorter-probe controller. It returned all 95 later balls it faced. [Portable receipts and analysis](../../artifacts/m1b/2026-11-19-rapid-recovery-qualification.zip)

The qualification compares controllers on independent native games. Each game starts from the acquired visual-button checkpoint and runs against the opponent at speed 0.65 until a score or 20 simulated seconds. Thirty planned starts cross every nonidentity permutation of the three button requests with switches at 2, 4, or 7 seconds. Each combination has one start in each serve direction. A second, independently sampled batch follows the same design. The preparation step excludes seeds in the available prior schedules, pins both batches before play, and does not select starts based on the trajectories that result.

Each acquired, frozen comparator plays the same start. The unchanged arm always uses the original execution. The frozen arm receives the change and keeps the acquired order. The shorter-probe arm receives the same change and can adapt with its existing twelve-camera procedure. The calibrated-at-switch arm gets the exact adjusted order at the switch through its frozen learner. Its physical requests, observations and outcomes must match unchanged play in every game. This last comparison measures what the inherited policy can do with instantaneous relabeling. It supplies no data to the candidate.

The candidate plays each start once with the hidden change and once with unchanged execution. Every controller uses the same frozen camera tracker, visual relation and neutral fallback. The candidate's only inputs are the public visual entities, its own requests and hit, score, terminal and elapsed-camera feedback. The runner applies remaps outside the controller and uses the simulator's contact geometry to score returns. A prefix hash covers requests and observations up to the designated switch. The first post-switch contact clock should match across arms when those prefixes match.

## Criteria fixed before play

- Return the first post-switch ball on at least 80% of planned starts and at least half the starts for each permutation. An early ending or no first opportunity remains in the planned denominator.
- Improve first returns over the matched shorter-probe control by at least 20 percentage points, with a one-sided exact paired sign-test probability of at most 0.05. Counts treat the start as the unit of comparison.
- Return at least 90% of later balls on at least 30 later opportunities and achieve two or more post-switch returns on at least half the planned starts. Later opportunity counts depend on what each controller does, so both counts and rates are reported.
- Add no pre-switch misses or early endings against the same-start unchanged control. Across full unchanged games, return at least 90% of first balls and 95% of later balls, with neither rate more than five percentage points below the unchanged control.
- Separately confirm the full request order within 0.75 simulated seconds on at least 60% of planned starts. The runner credits confirmation at the tenth consecutive camera decision reporting the correct order. It also counts gained first returns whose full order was still unconfirmed at contact.

Passing the return and unchanged-play gates qualifies the behavior. The full qualification also requires the order-speed gate. The first-return improvement, later-return performance, and order confirmation are measured separately so a successful interception made with partial evidence still counts as a return.

## Held-out results

| Controller | First return after switch | Later returns | Correct order confirmed within 0.75 s |
|---|---:|---:|---:|
| Shorter probes | 12/30 | 4/16 | 0/30 |
| Acquired frozen | 13/30 | 5/18 | 0/30 |
| Rapid | **27/30** | **95/95** | **22/30** |
| Calibrated at switch | 30/30 | 107/107 | 30/30 |

Rapid gained first returns on 16 paired starts and lost one against shorter probes, a 50-percentage-point improvement with one-sided exact paired p = 0.0001373. It achieved at least two post-switch returns on 27 starts. Its first-return counts ranged from 5/6 to 6/6 across the five permutations and were 10/10, 8/10 and 9/10 at the three switch times. All arms entered all switches and shared the same pre-switch requests, observations, and first post-switch contact clocks. The rapid controller completed 173/173 returns under unchanged execution and matched the original reference’s physical trajectory on every start.

The rapid controller confirmed the correct full order on 28/30 starts, with a median confirmation time of 0.617 simulated seconds among those 28. It confirmed the order before the first contact on 21/30. Two of its 16 gained first returns came before the ten-camera confirmation. Exact replays showed that the controller made four and three correct translations from partial evidence before it first selected the correct full order. The first such translations came 0.233 and 0.258 seconds after the switch. A replay that used only the last selected full order for execution missed both contacts at the same contact clocks. The replays cover those two rescues, which were selected because they occurred before stable confirmation.

The two primary phases used 142,198 camera transitions and 180 initial cameras. The four baseline arms took 167.829 seconds, and the two rapid arms took 89.344 seconds. The reserve batch remains untouched.

## Reproduction

Use the Python environment for this checkout and prepare an unused receipt directory. The acquisition checkpoint is `acquired.json` from the normal button scout, alongside its `live/game-05.json` receipt in the motor-learning archive. The preparation command emits the SHA-256 of the locked spec. Run the primary baseline before supplying the frozen candidate source. The harness uses four worker processes and one Torch thread each.

```bash
export PYTHONPATH=src:scripts/experimental OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/experimental/run_pong_rapid_recovery_qualification.py prepare \
  "$receipt_directory" "$button_scout/acquired.json" "$prior_experiments"
python scripts/experimental/run_pong_rapid_recovery_qualification.py run \
  "$receipt_directory" --spec-sha256 "$spec_sha256" --batch primary --phase baseline
python scripts/experimental/run_pong_rapid_recovery_qualification.py run \
  "$receipt_directory" --spec-sha256 "$spec_sha256" --batch primary --phase candidate \
  --candidate "$candidate_file" --candidate-class "$candidate_class"
```

The preparation step records source, asset, checkpoint, schedule and dependency provenance. Each run records every complete game, the candidate source hash, both phases' wall time and the aggregate results. The reserve batch remains available for an independently frozen follow-up.

The archive contains all primary starts and receipts, the acquired checkpoint, the exact candidate source, and the two-case replay tool. The registration copy omits the sampling entropy to keep the reserve starts blind. The locked spec SHA-256 is `2234c475270db14dd8164cd44da852164a8071fc02b116990f341b3c93b03067`. The archive manifest supplies individual receipt and source hashes.
