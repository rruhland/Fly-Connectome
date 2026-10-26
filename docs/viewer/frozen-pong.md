# Frozen M1B Pong viewer

Watch a frozen learned checkpoint play the native deterministic test opponent, or control the right paddle yourself. Five real checkpoints can be selected in the same viewer. The left controller always receives the existing unmarked 64×64 camera through its original frozen vision, encoder and scaler. The viewer has no training, saving, worker-launch or policy-edit endpoints.

## Launch

From the repository root, using the existing experiment artifacts:

```sh
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  /workspace/Fly-Connectome/.venv/bin/python -m fly_connectome.viewer \
  --experiments /workspace/m1b-experiments \
  --export /workspace/m1b-experiments/pong-interception-checkpoint-20261025 \
  --host 127.0.0.1 --port 8767
```

Open `http://127.0.0.1:8767`. Use `--host 0.0.0.0` only when exposing the viewer through the environment's browser/forwarding facility. The app uses only Python's HTTP server plus the project's existing NumPy/Torch runtime. Its HTML, CSS and JavaScript are package data; there is no frontend build or external CDN.

The final export is required. Missing earlier checkpoints are omitted and reported by `/api/catalog`. The installed core source signature must match the export. The exported scaling hash and installed/exported visual prior must match their benchmark pins. The three historical curricula also declare the same visual model hash and configuration; the checkpoint arrays must match the current 4389-dimensional encoder. There is no fallback to a scripted or randomly initialized controller.

## Use

- **Watch a match:** the exact native deterministic opponent at speed .65. Ordinary rallies run until a score or the native 20-second limit, not a first-hit cutoff. A short presentation hold makes each result visible, then the next serve starts. Points, misses, timeouts, rally number and learned returns remain distinct.
- **Play the right paddle:** hold Up/Down or W/S; the touch buttons work with pointer capture. Right-paddle speed .65 and acceleration 2 are unchanged. The learned left policy remains frozen. Human games are outside the benchmark.
- **Pause, frame-step, reset:** a frame step is one native camera decision, with the existing alternating 2/3 physics-tick cadence. Reset repeats the seed from the first serve. Switching checkpoints resets the same seed and pauses so comparisons are deliberate.
- **Playback speed:** requested speed can exceed the machine's throughput; the UI displays achieved simulated-time/wall-time speed. Watch mode batches at most four camera decisions per HTTP request. Human mode uses one per request so input does not remain stale for a batch.
- **Serve seed:** the default is the neutral seed `123456789`. The optional **Selected gain example** loads `2047868701`, the explicitly selected illustrative r4-to-final improvement from the report. It is not aggregate performance evidence.
- **Vision:** the main canvas displays the actual latest camera. The small camera preview is always unmarked. Entity positions, observed velocities and forecasts come from `latest_encoded` after that frame was processed; none are computed from privileged game coordinates. Forecast horizons are the installed prior's actual 1/4/8 camera samples (20/80/160ms). A few likely mixture components are shown for legibility. The model retains the complete mixture. IDs are tracking beliefs, not labels such as “ball” or “paddle.” The last action/readout scores belong to the preceding camera, while the overlay shows the current observation.

## Isolation and clock semantics

Viewer sessions live only in memory. There is no autonomous simulation loop: the browser asks for bounded camera steps. Leaving the page, hiding the tab or losing its focus sends a pause and clears human controls; returning to the tab does not automatically resume. No requests means no advancing game clock. A request already executing may finish its bounded batch (at most one camera in human mode). After a heartbeat gap of 1.5 seconds, the server also requires an explicit play command. Network failures stop the browser pump.

`FreshPongEvaluator` is used directly in watch mode. Human mode replaces only the environment's right-paddle controller with the viewer-only `HumanPong` subclass: desired velocity is computed from user input before native acceleration, movement and ball collisions. The copied physics step is tested against native `Pong.step` when supplied the same desired velocities. No core physics, vision, encoder, policy or training code is changed.

Raster/overlay serialization is a pure read, after capture. Rendering toggles are browser-only and do not send model commands. All checkpoint arrays are loaded from a fixed read-only catalog; API callers cannot supply filesystem paths or model arrays. The evaluator verifies frozen learned contents at creation, reset and terminal boundaries. The server never writes checkpoints, journals, training state or benchmark reports.

The browser-created games are demonstrations, not extra held-out trials or additions to historical results. Each tab has an isolated session; the service limits concurrent in-memory sessions. This is a local demonstration server, not a multiuser production service.

## Evidence shown

The first learned return, DIRECT specialist and opponent-return checkpoint have historical results on different cohorts. Their numbers are labeled separately and are not plotted as a continuous curve. r4 and the final checkpoint share the later paired 128-start cohorts: r4 118/128 direct and 113/128 opponent-return starts; final 125/128 and 117/128. The final checkpoint's material limitation remains visible: across all 160 held-out starts per phase, opponent-return performance is 142/160 (88.75%), not a distribution-wide 90% guarantee. Rally improvement, winning, reliable training from scratch and continuous multi-game learning are unproven. Human play does not inherit the deterministic-opponent benchmark claims.

## Validation

```sh
PYTHONPATH=src /workspace/Fly-Connectome/.venv/bin/python -m pytest tests/test_frozen_viewer.py -q
```

The focused tests compare every native record and terminal session graph, exercise the float-valued forecast sigma, check pure overlay reads, verify human/native physics equivalence, pause/disconnect/reset behavior, reject training/save endpoints and validate the installed exact final artifact. Parent browser acceptance additionally exercises watch/pause/step, every checkpoint, overlays, human keyboard control, focus loss, seeds, speed and the mobile layout. Artifact file hashes are compared before/after interaction. No learning is launched by these tests or the viewer.
