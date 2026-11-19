# Hidden request delays during native Pong

The revised timing learner passed the full qualification on 16 fresh paired streams. It returned 21/32 first and 71/75 later balls under delay, up from 6/32 and 3/8 with timing frozen. After delay removal it returned all 16 openings and all 16 first balls after a subsequent remap. [Portable confirmation receipts and carried states](../../artifacts/m1b/2026-11-19-timing-recovery-confirmation.zip)

The timing learner increased first returns under delay from 9/32 to 20/32 and later returns from 2/11 to 62/70. It fell one return short of the prospective first-return floor. Opening play after delay removal also fell short of its restoration criterion. [Portable qualification receipts and carried states](../../artifacts/m1b/2026-11-19-timing-recovery-qualification.zip)

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

## Held-out v2 results

| Delayed play | Timing live first | Timing frozen first | Timing live later | Timing frozen later |
|---|---:|---:|---:|---:|
| Hidden switch during the game | 11/16 | 6/16 | 34/37 | 0/6 |
| New game with the same delay | 9/16 | 3/16 | 28/33 | 2/5 |
| Both delay phases | **20/32** | **9/32** | **62/70** | **2/11** |

The live arm gained three first returns at the 12-camera delay and eight at the 21-camera delay. Across independent streams the improvement was 34.4 percentage points, with ten streams improving, one worsening and five tying. The one-sided exact paired sign-test p-value was 0.00586. The live arm made at least two affected returns on 18/32 games, compared with 2/32 for the timing-frozen control. All 16 mid-rally streams had matching prefixes and no pre-switch misses. The undelayed Stage 1 reference returned 32/32 first and 132/132 later balls over the two corresponding games.

The live arm selected the exact delay during 15/16 mid-rally episodes, with a 0.817-second median among those 15. It selected the exact delay by the end of the second game in all 16 streams. When the delay was removed, live play returned 11/16 openings and 44/45 later balls. Five starts ended before the fresh remap. It returned 10/16 planned first balls after the remap, including 10/11 that became available, and 34/34 later balls. Stage 1 returned 16/16 restored openings and 13/16 first balls after the same remap. The live controller passed the sustained and matched delayed-control gates, and fell short of the absolute first-return and restoration-opening gates.

An environment-side reach check applied neutral until the first fresh request could take effect, then held one of the two extreme drives until the first incoming contact. Five of the 16 carried-delay openings fell outside the resulting paddle reach interval, including the contact margin. Inside that interval, live returned 9/11 and timing frozen returned 3/11. All 16 starts still count toward the primary criterion. [Opening reach diagnostic](../../scripts/experimental/check_pong_delayed_opening_reach.py)

The complete comparison used 98,677 transition cameras and 144 initial cameras in 146.811 seconds. The archive includes the exact carried state from the lowest eligible stream for each held delay and for restored original controls. The locked spec SHA-256 is `c9b86a7635a50c9470a5f170957b184747273ce97cd7f228083acf48efb6aad0`.

## Fresh v3 confirmation

The revision reconsiders a carried delay when the same visually tracked paddle moves before any nonneutral command in the fresh queue could have taken effect. It provisionally selects the immediate model and continues to compare delay models using public motion. The new controller ran once on a wholly fresh schedule with the same design, starting checkpoint, paired controls, and criteria.

| Delayed play | Timing live first | Timing frozen first | Timing live later | Timing frozen later |
|---|---:|---:|---:|---:|
| Hidden switch during the game | 10/16 | 5/16 | 36/36 | 3/7 |
| New game with the same delay | 11/16 | 1/16 | 35/39 | 0/1 |
| Both delay phases | **21/32** | **6/32** | **71/75** | **3/8** |

The paired improvement was 46.875 percentage points. First returns increased by seven at a 12-camera delay and eight at a 21-camera delay. Eleven streams improved, none worsened, and five tied, giving one-sided exact paired p = 0.000488. The live controller made at least two affected returns on 18/32 games versus 2/32 with timing frozen. All 16 mid-rally prefixes matched exactly with zero pre-switch misses. Five carried openings were physically outside the range under an empty request queue. The timing learner returned all 11 inside that range, compared with 1/11 for the timing-frozen control. All 16 remain in the planned denominator.

The timing learner selected the exact delay in 15/16 mid-rally games, taking a median 1.033 seconds among those 15. By the end of carried delayed play it had selected the exact delay in all 16 streams. After delay removal it selected zero delay on every start in a median 0.217 seconds. It returned 16/16 openings and 74/75 later balls across the restored games. After the fresh hidden remap it returned 16/16 first and 56/57 later balls. The undelayed Stage 1 reference returned 16/16 first and 57/57 later balls after the same remap.

The complete comparison used 104,497 transition cameras and 144 initial cameras in 135.723 seconds. The archive includes the actual v3 timing checkpoint selected after restoration and before the fresh remap. This checkpoint is the timing input for the next stage. The confirmation spec SHA-256 is `819772b0f68487b535acf50cda76a48039f99efa29a8f543763fd8889dd7ca8f`.

## Reproduction

Prepare an unused receipt directory using the Stage 1 acquisition checkpoint and the native Stage 1 carry summary. The preparation command emits the locked spec SHA-256. The Python environment must make the Stage 1 rapid controller importable. Supply the frozen timing controller source to the run command.

```bash
export PYTHONPATH=src:scripts/experimental OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python scripts/experimental/run_pong_timing_qualification.py prepare \
  "$receipt_directory" "$acquired_checkpoint" "$prior_experiments" "$motor_carry_summary"
python scripts/experimental/run_pong_timing_qualification.py run \
  "$receipt_directory" "$timing_controller_file" --spec-sha256 "$spec_sha256"
```
