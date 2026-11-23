# Completing a five-axis wrist controller after half a second of native Pong

The native Pong actor already learned from actual game outcomes. Its archived physical joystick bridge returned eight balls across eight source points while the stationary comparison returned zero. The ordinary Pong control interface issues one direction through the physical SO-101 and reads a projection of its encoders into the native screen paddle. All five encoder channels are logged during this play, so the first 30 arm ticks estimate one five-component motor response. Resulting commands have no information about the complementary four input dimensions. [Earlier native Pong evidence and trained actor provenance](played-pong-prefix.json.gz) · [Native prefix estimator](../../scripts/experimental/tt24_played_pong_prefix.py)

The physical test begins a new fixed-length native point on the actual receiving arm, using the same frozen reward-trained actor and the first 0.5 seconds of its ordinary live loop. The learned source column for each start uses its own 30 new command and encoder pairs. The earlier archived columns are retained in the receipt for comparison. The prefix sends no scripted motor probes. The joint-angle filter is chosen by chronological prediction within that prefix. This stage charges 60 native frames, or 30 simultaneously executed physical encoder transitions, once per trial. It does not use any scalar Pong payout to fit the motor map. [Joint command ledger](pong-dense-setup.json.gz)

## Matched physical comparison

Ten fresh stock-arm starts use independent wrist launches and new dense five-dimensional command mixes. The first eight hold the source-stimulated action response fixed across the game switch while other couplings differ by physical start. In two separate trials the wiring reverses that one unmeasured direction after the native point and preserves the four other effects. The fit never reads the evaluator's wiring matrix. Continuity is imposed as a test condition. The complementary directions alone cannot validate it.

After Pong the same arm visits small balanced command offsets in a randomly rotated basis of the four-dimensional orthogonal complement. The corresponding settled changes in actual encoders identify the remaining map. The carried, stale and time-shuffled versions share that exact physical prefix and setup, and the same projector for visual preparation. Their released joints, joint velocities, clock and wrist image hash match on each pair. An independent cold receiver skips Pong and measures all five directions. It uses its own full fitted map while the launcher holds the ball. Both routes play six-second physical evaluations with the same frozen learned wrist actor and native rebound judge. The wrist actor and readiness phase use only wrist RGB plus encoders. [Continuous fixture, calibration, and replay code](../../scripts/experimental/tt24_pong_dense_setup.py)

A seven-start development pilot used widened body postures and a stationary visual preparation interval. Median scores were two carried, one stale, zero shuffled and two cold. The fresh ten-start schedule restored the ordinary stock posture and moved during visual preparation. The carried, stale and shuffled preparation still uses only the shared four known directions. Both the pilot and final test froze the game actors and motor estimators. [Development pilot](pong-dense-setup-pilot.json.gz)

The active carried route re-estimates the response from each current point. A second controlled route instead takes the exact coefficient from the first archived point of the original experiment, with no refitting of that direction. It reconstructs and charges the same full receiving setup as the current-point route. These routes change only the first response column before the held ball is released. The archived donor Pong checkpoint used nine reward-driven updates across 990 native points and achieved at least one return on 23 of 24 held points versus seven for the stationary head. Across the ten newly recorded 60-frame prefixes, the controller issued real gameplay actions before the first return and received zero hits or point outcomes within that window. [Earlier native learning budget and held outcomes](tt24_joint_pong.md)

## Results with continued wiring

| Fit on eight new starts | Qualified rebounds | Survived 5 s | Mean five-axis map error | Motor setup | Total time before release | Rebounds per elapsed simulation minute |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Current Pong estimate plus four measured directions | 131 | 7/8 | 0.31% | 1.375 s | 2.300 s | 128.6 |
| Exact archived Pong estimate plus four measured directions | 130 | 7/8 | 0.33% | 1.375 s | 2.300 s | 127.6 |
| Stale direction plus same four measured directions | 57 | 3/8 | 33.20% | 1.375 s | 2.300 s | 85.4 |
| Time-shuffled Pong direction plus same four measured directions | 8 | 0/8 | 63.74% | 1.375 s | 2.300 s | 20.6 |
| Cold measurement of all five directions | 148 | 8/8 | 0.19% | 1.529 s | 1.954 s | 139.5 |

The carried map saved one measured direction and 0.154 s of receiver motor setup. Once the 0.5-s Pong prefix was counted, the cold route released the ball 0.346 s sooner and scored more. The two carry estimates each reached five seconds on seven common starts. Their advantage over the stale and shuffled estimates is measured from identical physical release states. For example, seed `382592233` produced 18 and 19 rebounds from current and archived Pong estimates, one with stale, zero with shuffled and 18 with cold. Time per minute includes the required setup and the observed live time to termination or cap. The relative map errors are calculated by the referee after scoring and are never read by the controller.

## Deliberate wiring replacement after Pong

The current and exact archival Pong estimates each made one rebound across two starts after a sign reversal in the single skipped direction. The stale branch also made one, the five-direction cold fit made 35, and the shuffle made 20. The time shuffle happened to reverse the direction and partially approximate the replacement in these two cases. The four-direction examination produces no signal for determining whether the omitted mapping is still valid. There is no automatic validation or repair when that condition fails.

Every recorded physical tape reproduces its full judge state, elapsed ticks, and gravity events, including the source motion, setup, and timed handoff before the ball is physically released. The carried and cold branches each progress without a state reset between their own initial start and the wrist game. The evaluation uses ten independent target starts and 50 verified tapes, with zero target-reward updates. [Replay data and estimates](pong-dense-setup.json.gz) · [Downloadable motor banks](pong-dense-setup-bank.npz)

## Reproduce

```sh
export PYTHONPATH=src MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt24_pong_dense_setup \
  --output /tmp/pong-dense-setup.json.gz --bank /tmp/pong-dense-setup-bank.npz
```
