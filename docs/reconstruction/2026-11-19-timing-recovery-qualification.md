# Hidden request delays during native Pong

The timing experiment starts from the motor belief saved after an actual Stage 1 remap and restoration, with the original control order and a zero-delay estimate. Each controller plays against the opponent at speed 0.65. The timing-frozen controller continues to learn the motor order and holds its delay estimate at zero. An undelayed Stage 1 controller provides a reference with the same initial motor memory, also carried between games.

The prospective schedule has 16 independent three-game streams. Streams balance 12- and 21-camera delays, switches at three and five simulated seconds, serve direction, and two permutations used in the restoration game. The arms receive the same starts and initial state within each stream. The environment applies the delay to its camera-command queue outside the controller. If the queue contains fewer requests than the delay requires, it executes neutral. The controller receives visual entities, its own request, elapsed camera time and inherited hit, score and terminal feedback.

1. The first game starts without a delay. It receives a hidden delay during play and retains that delay to its end. The prefix hash covers public requests and observations up to the intervention.
2. The second game starts with the same delay and a fresh command queue. The controller carries its public motor and timing memory from the preceding game.
3. The third game starts with the delay removed. It carries the preceding memory and receives a hidden button remap after four simulated seconds. Opening return performance measures play after delay restoration. The post-remap return performance measures whether motor adaptation still works in that state.

The undelayed reference also receives the last game's remap. All games end at a score or 20 seconds. Every planned game contributes to the first-return and sustained-play denominators, including games that end before the intervention or never present a post-change opportunity.

## Prospective timing criteria

- Return at least 65% of first affected balls across the two delay phases. Improve the matched timing-frozen first-return rate by at least 20 percentage points, with positive first-return count gains at both delays. The one-sided paired sign test uses each three-game stream as an independent unit and requires p ≤ 0.05.
- Return at least 70% of later balls on at least 24 opportunities. Make at least two returns after the intervention in at least half of the delayed games.
- Add no pre-switch misses or pre-switch endings compared with either reference.

The separate restoration criteria require at least 75% opening and 80% later returns across the restored games. After their hidden motor remap, the controller must return at least 70% of first balls and at least 80% of later balls. The first-return fraction must be within 20 percentage points of Stage 1 on the same starts. The runner records delay estimates and response timing for diagnosis. The return gates judge whether the delay learner recovers useful control.

## Reproduction

Prepare an unused receipt directory using the Stage 1 acquisition checkpoint and the native Stage 1 carry summary. The preparation command emits the locked spec SHA-256. The Python environment must make the Stage 1 rapid controller importable. Supply the frozen timing controller source to the run command.

```bash
export PYTHONPATH=src:scripts/experimental OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/experimental/run_pong_timing_qualification.py prepare \
  "$receipt_directory" "$acquired_checkpoint" "$prior_experiments" "$motor_carry_summary"
python scripts/experimental/run_pong_timing_qualification.py run \
  "$receipt_directory" "$timing_controller_file" --spec-sha256 "$spec_sha256"
```
