# Denser stereo for the private ball state

This experiment adds a separate, same-pose 768×576 stereo exposure on each 60 Hz control tick during optically observed incoming flight. The ordinary 384×288 stereo continues at 30 Hz and still determines the raw joint field, ball direction, bounce phase, when searches run, and which executed references receive credit. Only the private rollout's position and velocity may be replaced at a search sync. The treatment never changes court, robot, actuator limits, opponent, target draws, or plant integration.

The supplemental camera fits a bounded suffix of actual RGB ball rays with drag and gravity. It needs fresh observations in both views and rejects poor fits or a large disagreement with ordinary optics. All disagreements and rejections fall back to the original p/v without touching its tracker. There are no special keyframes, case-specific labels, simulator poses, or future exposures inside the actor. The denser stream increases the sensor budget and has a separate measured render-and-detection CPU cost.

## Paired camera measurements

For this diagnostic both camera streams watched exact replays of complete physical action tapes. Positions and velocities were scored against separate native evaluator records. These records are not read by the camera estimator. A 67 or 100 ms endpoint uses the same free-flight equation and is scored only if the identical native arc still exists at that endpoint. Thus a native collision is not scored as an optical extrapolation mistake. Endpoints include model error shared by both arms, such as unobserved spin. The two extra ablations use the same fit at 384×288 and 30 Hz or 768×576 and 30 Hz.

On 405 quiet post-bounce practice frames, mean Euclidean position error fell from 2.73 to 1.93 mm, and velocity error from .144 to .132 m/s. Mean endpoint errors fell from 14.68 to 13.23 mm at 67 ms and 23.76 to 21.94 mm at 100 ms. The new fit was applied on 335 of those frames. A physics-only fit, without fitting an extra force or using longer image windows, was frozen before evaluating the release-disjoint sensor holdout. The four practice roles and eight held positions in the action archives were fixed by the capture manifest, not by measured camera error. [Practice fit and frozen selection](practice-frozen.json)

On 1,070 quiet post-bounce frames across the eight held releases, the frozen fit reduced mean position error from 2.90 to 2.20 mm, velocity error from 0.141 to 0.131 m/s, and mean endpoint error from 15.24 to 14.23 mm at 67 ms and 24.99 to 23.78 mm at 100 ms. It applied on 881 frames. The release-weighted bootstrap intervals for the endpoint differences were −1.38 to −0.47 mm at 67 ms and −1.81 to −0.53 mm at 100 ms. At 30 Hz, the 384×288 fit did not improve either mean endpoint and the 768×576 fit improved both by less than 0.3 mm. This is a gain in ball-state estimation on the stock action tapes, not a gain in simulated contact fidelity. [Held exposures and scores](accuracy.json)

## Matched games

The prospective comparison keeps the carried-credit learner's initial physical readiness and motor memory, training bank, scoring rules, random draws for each release and search seed, search parameters, and naturally accumulated credit path. Its six ordered assignments are the first six of the complete TT39 credit-carry archive. The first stock game is freshly run to test exact archive compatibility, with automatic fresh control for the rest if it diverges. Each physical before/after pair is checked through its first different issued command. Once actuated changes alter the incoming trajectory or rewards, later activity is real, but no longer a matched same-ball intervention.

All six pairs verified to the first different issued command, and the fresh first stock run exactly reproduced the archive. Stock scored **37** legal near returns and the denser private state scored **31**. The treatment improved two games, lost ground on three, and tied one. Alternating legal returns were 80 vs 68, and the longest run was 17 vs 11 legal near returns.

| Release | Stock | Dense sync | First differing command |
| --- | ---: | ---: | ---: |
| 12109 | 8 | 1 | 80 |
| 9812 | 6 | 2 | 70 |
| 5098 | 17 | 9 | 71 |
| 3173 | 5 | 8 | 188 |
| 12259 | 1 | 11 | 79 |
| 8510 | 0 | 0 | None |

Every changed command first affected an identical incoming offer. Both arms contacted and legally returned all five of those offers. They nevertheless produced different outgoing landings, including a 22.8 cm depth difference on 3173. Only the first game started with the same carried memory. The other five measure the combined effect of private synchronization and the resulting credit path, not an isolated estimate swap. The supplemental sensor applied on 98 of 155 syncs and used 51.1 CPU seconds across the six treatment games. The final point-boundary checkpoint verified at serial 6 with 456 stored fragments. [Paired outcomes and native replay checks](outcomes.json) · [Action archive](live.json.gz) · [Checkpoint](live.npz)

Strict CAD replays of the two treatment gains reproduced the native events with zero penetration samples across 16,383 physics steps. Those two replay checks do not establish clearance for every game.

The direct question for a follow-up is whether both estimates support the *same motion*. On matched, frozen-memory offers, run the same candidates through stock and dense sync on the same decision tick. Only replace the incumbent if the proposed motion improves a predeclared contact and outbound margin under both estimates. Measure the first affected legal return and the next two returns separately. The observed gain in endpoint accuracy and unchanged first-hit legality do not establish that switching actuated motion is beneficial.

## Reproduce

Run from the repository root with the project environment and MuJoCo available.

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt42_camera_precision practice
python -m scripts.experimental.tt42_camera_precision_report --output artifacts/tt42_camera_precision/practice-frozen.json
python -m scripts.experimental.tt42_camera_precision held
python -m scripts.experimental.tt42_camera_precision_report --held artifacts/tt42_camera_precision/held.npz --frozen artifacts/tt42_camera_precision/practice-frozen.json
python -m scripts.experimental.tt42_camera_precision_live --sensor artifacts/tt42_camera_precision/practice-frozen.json
python -m scripts.experimental.tt42_camera_precision_audit --cad
pytest -q tests/test_tt42_camera_precision.py
```
