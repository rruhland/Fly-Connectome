# TT75 — incoming state from causal cropped stereo

The original unmarked orange ball, fixed camera mounts, ordinary uniform physical opponent, and original centered arm remain unchanged. An electronically aimed 192×192 crop per eye is **rendered at its actual narrower field of view**, not enlarged after capture. Previous RGB detections determine crop centers. Exposures run at 120 Hz; ordinary full-camera RGB reacquisition runs at 30 Hz. The native renderer has instantaneous exposure already; no new shutter or blur model is claimed.

An RGB-only ROI translation calibration measured magnification 3× horizontally and 6× vertically (the requested renderer zoom is 6). The ball's ellipse is evaluated in calibrated source-pixel coordinates. Calibration moves only the optical ROI on a fixed startup scene; no true ball coordinates enter it. See `sensor-calibration.json`.

`DenseFlight` adapts TT39's joint translation/spin linearized physical least-squares fit to physical-time windows and the higher-resolution center noise. It fits the causal recent inbound arc or an observed rebound and estimates p/v and spin together. No backpropagation, markings, live state, opponent intent labels, targets, release IDs, or future observations enter sensing or action selection. The existing standard tracker remains an explicit fallback.

## Six sealed held action replays

`protocol.json` freezes the three shortest-prefix old missed offers and three shortest-prefix separate routine controls, all release-disjoint from the earlier spin-practice data. `held.json.gz` contains all six completed physical replays and each causal estimate before separately recording evaluator truth. All six issued command/event prefixes exactly match the archived physics. This is a state/recognition experiment; no new motor policy acts in these replays.

Equal-release averages of each release's median absolute error on matched inbound frames:

| Quantity | Existing optics | Dense joint fit |
|---|---:|---:|
| x / y / z position, mm | .442 / 1.428 / .726 | .212 / .311 / .310 |
| vx / vy / vz, m/s | .00656 / .02485 / .01379 | .00179 / .00286 / .01037 |
| x / y / z spin, rad/s | 22.39 / 5.40 / 10.27 | 5.36 / 4.99 / 3.05 |

The numerical improvement is substantial for depth, lateral/longitudinal velocity, and transverse spin. It does **not** resolve the contact bottleneck: of three physically missed old action tapes, standard optics accepts all three, whereas the new fit accepts two. Of three physically legal tapes, standard optics accepts two; the new fit also accepts those two and is unavailable on the third at the decision, so exact standard fallback rejects it. Recognition is 3/6 versus 2/6. The exact-state evaluator classifies all six correctly. One near-bounce fitted vertical velocity is visibly wrong, so raw center coverage must not be presented as usable flight coverage.

There were 1,314 valid stereo observations from 1,409 exposures (93.3%). The six replays cost 965 actual control ticks, 197.87 s wall and 60.25 s CPU; stereo acquisition alone used 41.27 s CPU, 29.3 ms per exposure. Development additionally cost 20 projection-probe ticks and an interface-abort ceiling of 106 ticks (exact aborted count unavailable, disclosed in `development-cost.json`). No completed case was discarded. Detailed per-release metrics are in `held-summary.json`.

TT83 independently paired the same exact observer bytes on four paid-motion forks: fit available at 1/4 decisions despite 675/699 valid stereo exposures. On that one fork the new estimate suppressed an issued motor option which actually missed. This supports one correct rejection; it supplies no new successful return. TT83 owns its complete physical receipts.

## Ordinary actor integration

`tt75_observer_adapter.py` is opt-in. An exposure host supplies actual RGB and public timestamps. TT83's `PublicTracker` bridge supplies the selected joint p/v/spin fit to the unchanged nominal body estimator. The direct field, temporal actor and motor search all use the same selected camera state. If no usable fresh dense fit exists, they retain the existing standard tracking input and direct-field base. Body state is still reconstructed from actual own encoders and issued references.

`ordinary.protocol.json` fixes the first 32 complete TT80 games, the original frozen CARRY checkpoint, budget 28, cap 240 and the complete archived frozen parent. No game or checkpoint is selected by outcome. A disabled first-game check reproduced all 109 commands bitwise and the entire physical event record canonically (`disabled-parity.json.gz`). Initial Python direct equality incorrectly compared live tuple fields with JSON lists; canonical serialization proves equality. Three 109-tick parity/debug replays were paid, total 327 ticks, including the first unrecorded repetition; one intermediate source-hash rejection used zero native control. The final parity uses the same EGL and packed-camera configuration as TT80.

The first dense game completed 0 returns versus the parent's 0, with 29 dense and 80 fallback decisions. Its audit write then rejected a NumPy boolean; only the recorded flag was converted to a Python boolean, with no inference or action change. This additional 109-tick completed game is charged and its failure log is retained. The identical fixed first32 course was restarted after that serialization correction.

An initial32-game run is invalid instrumentation (`ordinary-context-failure.json.gz`): after closing the first crop renderer with the wide GL context current, the stock camera kept a frozen prior frame. It cost3,243ticks/161.57s wall/41.39s CPU and is fully retained. A340-tick issued-prefix causal comparison proved the failure and fix: unsafe secondgame0/171valid stereo, own-context close secondgame170/171. No sensing/fit/motor rule changed. The initial static-scene close check was nondiagnostic because a frozen identical frame passes it.

The corrected runner also uses a fresh actual arena and camera per game, maintaining the normally initialized persistent actor, bank and serial. Two complete disabled games reproduced the archived parent exactly: 0 and13 returns, all1,449 actual commands bitwise and all physical events canonically identical (`fresh-disabled-parity.json.gz`). The same768-row earned bank digest remains unchanged. The valid first32 course began only after that check passed. The valid actual dense first32 course uses `ordinary-dense-fixed.json.gz`; all completed assigned games, fallback counts, selected observations, physical actions, true uniform requests and returns are retained. This file is not a result claim before the complete comparison is reported.

Reproduction (from repository root):

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
/workspace/Fly-Connectome/.venv/bin/python -m scripts.experimental.tt75_incoming_state run --out /tmp/tt75-held.json.gz
/workspace/Fly-Connectome/.venv/bin/python -m scripts.experimental.tt75_ordinary_play dense --n 32 --out /tmp/tt75-ordinary.json.gz
```

The stored source-hash protocol must remain unchanged. Original held-source and carried-checkpoint files are required.
