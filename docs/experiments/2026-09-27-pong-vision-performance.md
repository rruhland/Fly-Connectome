# Promoted vision on rendered Pong

## Loop and scope

`scripts/benchmark_pong_vision.py` runs one seeded Pong game with zero motor
drive. Pong physics advances in its existing 120 Hz timestep; visual samples use
the exact 2/2/3/2/3 physics-tick cadence for 50 Hz. Each sample renders the
actual 64x64 Pong tensor image, derives dense OFF/ON events from consecutive
renders, runs the promoted M1A.5 visual service, and encodes its full result.
The game resets on scores while camera and vision histories continue across
serves. Learning receives rendered pixels and events only. Render here means
Pong's sensor rasterization; no display window or scanout is timed.

## Reproduction and measurement

The tracked benchmark and parity scripts accept `--source-root` and fail if
Python resolves `fly_connectome` from a different checkout. The reference is
`57600a9` at `C:/Users/rruhl/OneDrive/Documents/Fly-Connectome`; the optimized
runtime code is commit `4f01505` on branch `codex/m1a-performance` at
`C:/Users/rruhl/.codex/worktrees/m1a-performance/Fly-Connectome`. For example:

```powershell
New-Item -ItemType Directory -Force runs | Out-Null
$python = 'C:/Users/rruhl/OneDrive/Documents/Fly-Connectome/.venv/Scripts/python.exe'
& $python scripts/check_pong_vision_parity.py --source-root 'C:/Users/rruhl/OneDrive/Documents/Fly-Connectome' --samples 240 runs/original.json
& $python scripts/check_pong_vision_parity.py --source-root 'C:/Users/rruhl/.codex/worktrees/m1a-performance/Fly-Connectome' --samples 240 runs/optimized.json
& $python scripts/benchmark_pong_vision.py --source-root 'C:/Users/rruhl/.codex/worktrees/m1a-performance/Fly-Connectome' --checkpoint runs/original.online.pt --evaluation --output runs/evaluation.json
```

Each timing process used seed 1101, one PyTorch CPU thread, 40 warmup and 200
timed visual samples. Paired process order was original, optimized, optimized,
original in each mode. Online starts from the promoted default; frozen
evaluation loads the checkpoint after 240 online Pong samples, with 361–402
long-bank examples per horizon. Both modes issued **1,005 long-context
forecasts** in each timed run. The phase `events` includes previous-frame copy
and float-frame conversion. Initialization, first-use `vmap` setup, checkpoint
I/O, display, and real camera acquisition are excluded. Host load and CPU
frequency were not controlled; absolute latencies differed between sessions.

`compute samples/s` is 200 divided by the sum of the 200 measured loop
durations. It is compute capacity for this single process, not paced display
FPS. A deadline miss is an individual visual sample over 20 ms; the fixed
120:50 schedule is simulated, so deadline counts matter more than reciprocal
median latency for a real-time claim.
The ten source-verified run outputs and parity hashes are recorded in
`docs/experiments/2026-09-27-pong-vision-performance-results.json`.

| Mode and run | Original p50 / p95 ms | Optimized p50 / p95 ms | Original / optimized compute samples/s | Original / optimized >20 ms samples |
|---|---:|---:|---:|---:|
| Online 1 | 27.58 / 37.80 | 13.40 / 14.95 | 36.2 / 75.3 | 158 / 0 |
| Online 2 | 27.58 / 41.32 | 13.07 / 15.26 | 35.6 / 76.8 | 161 / 0 |
| Learned-checkpoint evaluation 1 | 36.65 / 57.63 | 19.20 / 27.26 | 26.4 / 50.7 | 184 / 71 |
| Learned-checkpoint evaluation 2 | 41.31 / 57.65 | 18.92 / 23.16 | 24.2 / 52.8 | 193 / 64 |

The paired p50 means improve **2.08x** online and **2.04x** in learned
evaluation. Early online compute capacity exceeds the 50 Hz camera target with
no timed deadline misses in either optimized run. Learned evaluation averages
above 50 compute samples/s but misses 64–71 of 200 deadlines; its p95 exceeds
20 ms. The 120 Hz physics simulation completed 576 ticks in every 240-sample
run. Each path recorded zero points, two missed Pong scores, and zero hits.

The optimized early online phase medians averaged 0.17 ms render, 0.04 ms
event/frame conversion, 9.35 ms vision, 2.50 ms encoder, and 1.12 ms physics
and scoring. Component medians need not sum to total median. Rendering and
event conversion are under 2% of total time. A prototype renderer grid cache
reproduced exact 64x64 and 32x64 pixels but saved only about 0.028 ms per
render in 2,000 calls, under 0.2% of the earlier full-loop timing, so it was
rejected.

## Late online window and exactness

With 1,000 online samples and only samples 800–999 timed, the original loop
measured **55.97 / 70.01 ms p50 / p95** and **18.3 compute samples/s**;
optimized measured **24.16 / 31.58 ms** and **40.5 compute samples/s**. This
is a **2.32x** p50 speedup, but the optimized loop missed **191 of 200** 20 ms
deadlines. Both paths recorded seven misses, two paddle hits, and identical
final banks: 3,113–4,192 short and 1,747–1,893 long examples per horizon.
The late optimized phase p50 was 18.18 ms vision, 3.76 ms encoder, 1.68 ms
physics/scoring, 0.28 ms render, and 0.06 ms event/frame conversion. Bank
growth, game state, and host load all differ from the early window; this
measurement does not isolate a single cause.

The original and optimized source paths produced byte-identical trajectory
JSON for **every** rendered frame, event tensor, complete visual state, encoder
result, Pong state, and physics outcome across 240 online and 240 evaluation
samples. Online learning began with default promoted weights. Evaluation
loaded the learned checkpoint, exercised long context, and left that
checkpoint unchanged. The matched trajectory JSON SHA-256 is
`bc2bc53f10e0e9e4c25a52ef074c22cb4b0b0f6d2199b75b34d53250c4eb01e6`;
the parsed checkpoint-content digest is
`96461b68280c4cdd2e61c56dcf91570e2fb52c0ddba0a3b68f7ead2dbbc4051c`.
The parity runner is tracked as `scripts/check_pong_vision_parity.py`.
Focused tests check dense events against `EventCamera`, cadence, source
selection outside the checkout, score accounting, and parity-runner output.
The full suite passed: **736 passed, 4 CUDA skips**.

Exact append-all bank retrieval, identity tracking, and mixture intervals
remain the main costs to investigate for sustained 50 Hz operation. Lowering
resolution, pruning examples, changing top-k, or shortening interval search
would change the approved visual service and was not used.
