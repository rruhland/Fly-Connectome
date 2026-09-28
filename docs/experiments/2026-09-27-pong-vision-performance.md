# Promoted vision on rendered Pong

## Loop and scope

The new `scripts/benchmark_pong_vision.py` runs one seeded Pong game with zero
motor drive. Pong physics advances at its existing 120 Hz; visual samples occur
at 50 Hz by alternating two and three physics ticks. Each sample renders the
actual 64x64 Pong tensor image, derives dense OFF/ON events from consecutive
renders, runs the promoted M1A.5 visual service, and encodes its full result.
The game auto-resets on scores; the camera reference and vision scene histories
continue across serves. `learn=True` updates only the approved visual learner;
there is no reward controller. Render here means Pong's sensor rasterization,
not browser presentation or display scanout.

## Paired timing

Both source paths ran the same script, seed 1101, CPU-only PyTorch with one
thread, 40 warmup and 200 timed visual samples. Times include rendering,
event conversion, complete vision and transport, score diagnostics, and all
physics ticks. Initialization, first-use `vmap` setup, checkpoint I/O, display,
and real camera acquisition are excluded. Host load was not controlled.

| Mode and run | Original p50 / p95 ms | Optimized p50 / p95 ms |
|---|---:|---:|
| Online learning 1 | 43.03 / 66.66 | 20.39 / 29.93 |
| Online learning 2 | 43.47 / 60.69 | 19.79 / 23.47 |
| Frozen evaluation 1 | 38.29 / 56.11 | 18.27 / 22.71 |
| Frozen evaluation 2 | 39.04 / 57.11 | 19.63 / 31.67 |

The means of paired p50 values give **2.15x** online speedup and **2.04x**
evaluation speedup. The optimized online p50 mean is **20.09 ms**, or **49.8
visual samples/s**; evaluation is **18.95 ms**, or **52.8 visual samples/s**.
The optimized 120 Hz physics schedule completed 576 ticks in every 240-sample
run, and both versions had the same two missed scores and no paddle hits.
The online median is at the 50 Hz boundary, while p95 exceeds the 20 ms
target. Neither mode reaches 120 visual samples/s.

The optimized online phase medians across the two runs were approximately
0.28 ms render, 0.06 ms dense events, 14.4 ms vision, 3.7 ms encoder, and
1.6 ms physics/score processing. These component medians need not sum to the
median total. Pong rendering and event conversion together are under 2% of
total time. A prototype that cached the renderer's pixel grid reproduced
exact 64x64 and 32x64 pixels but saved only about 0.028 ms per render in a
2,000-call microbenchmark, less than 0.2% of this full loop; it was not kept.

## Exactness and longer run

Across 240 online and 240 frozen Pong visual samples, the original and
optimized source paths produced identical SHA-256 digests for **every**
rendered frame, event tensor, full visual state, encoder result, game state,
and physics outcome. The learned and frozen version-3 checkpoints matched as
well. Both trajectory JSON files hashed to
`2567230bb8bb14771fe1419a7637a2ab636f2e48ae191dab039881d58c3aa383`.
Focused tests compare dense events with `EventCamera` and prove 50 samples
schedule exactly 120 physics ticks. The full suite passed: **733 passed,
4 CUDA skips**.

In a separate optimized 1,000-sample online run, timing only samples 800-999,
the p50 rose to **27.57 ms** and p95 to **58.24 ms**. The game recorded seven
missed scores and two hits. By then the short banks held 3,113-4,192 examples
per horizon and the long banks 1,747-1,893. This later window shows that the
early 50 Hz median does not persist. It combines larger banks with different
game states and host load, so it does not isolate one cause. Exact append-all
retrieval and Python identity tracking remain the main candidates for a
subsequent, parity-gated optimization.
