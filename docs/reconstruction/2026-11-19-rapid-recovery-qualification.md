# Rapid recovery from a hidden control change during Pong

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
