# Faster native stroke rollouts without changing stroke decisions

The optimization makes the clearance check cheaper. It does not change any MuJoCo integration step, actuator force envelope, contact calculation, or clearance sampling time. At the same sampled geometry state, it tests conservative world-axis bounds before calling `mj_geomDistance`. A pair is skipped only when an axis separation exceeds the existing narrow-test horizon. Potentially close pairs still use the original sphere test and native distance with the same thresholds and ordering. Invalid, unbounded and unsupported shapes bypass the new filter. Primitive bounds come from current dimensions, and the compiled mesh bounds and world transform are expanded outward to cover rounding. [Native code](../../scripts/experimental/tt29_rally_rollout.cpp)

## What the profile showed

The benchmark reconstructs camera-derived ball flight and real encoder/reference history at the first and last decisions in a recorded eight-return point. Each exact optical context evaluates the original physical actuator proposals and an independently generated candidate deck. Guarded 25-candidate batches made 52,402 or 53,825 native clearance distance calls. Warm model allocation took around 1 ms, and resetting candidate data around 0.5 ms, in calls whose uninstrumented CPU time was roughly 145–151 ms. Arm and ball integration after impact remained a substantial general cost. Skipping it would also skip live actuator limits, possible late faults, and output trajectory fields, so this patch does not do that.

The same two 25-candidate profile batches now call the native clearance distance 3,584 and 5,330 times, reductions of 93% and 90%. The 6,317/5,912 actual solver steps and 1,590/1,487 scheduled clearance checks did not change. [Instrumented counts and timings](rollout-bounds-profile.json)

## CPU and parity

Each row below is the range of two observed release contexts. CPU is the median of ten alternating reference/optimized calls on one native worker. A positive margin samples the guard every microstep.

| Native call | Reference CPU | Optimized CPU | CPU saved |
| --- | ---: | ---: | ---: |
| 9 candidates, no output trace | 43–68 ms | 37–56 ms | 15–17% |
| 25 candidates, all diagnostics | 134–150 ms | 110–116 ms | 18–22% |
| 72 candidates, all diagnostics | 411–449 ms | 336–358 ms | 18–20% |
| 25 candidates, +0.5 mm clearance margin | 255–273 ms | 163–181 ms | 34–36% |

All output scalars were bitwise identical for 262 candidate slots, including nine legal outcomes and 253 failed ones, and for each repeated run. The comparison includes unused/fault NaNs, issued rates, scores, weak-contact feedback, frame traces and approach state, not just the selected stroke. The same two entire physical points were then played independently through both libraries. All search records, commands, events, placements and scores were identical, with eight and fourteen legal arm returns. End-to-end CPU decreased from 46.57 to 43.16 s (7.3%). The remainder after separately timed camera and real arena physics fell from 12.59 to 10.61 s (15.8%). These timing samples are from the shared research environment and are not a guarantee for other workloads. [Frozen-state parity](rollout-bounds-parity.json) · [Physical A/B](rollout-bounds-physical.json)

The two-worker run also produced identical output bits for every case. Seven of eight subgroups used less CPU, while the first nine-candidate case was about 10% slower in this shorter, shared-host measurement. The 25- and 72-candidate ordinary groups saved roughly 16–23% CPU. All 88 targeted tests passed, including a direct comparison of the new bounds against 13,500 native distances for rotated box, sphere, capsule, cylinder, ellipsoid and mesh pairs with multiple horizons and signed penetrations. [Two-worker parity](rollout-bounds-two-workers.json) · [Conservative bounds regression](../../tests/test_tt40_rollout_bounds.py)

## Reproduction

Run from the repository root with a MuJoCo-enabled Python environment. The reference hash is pinned to the exact native file before the optimization. The benchmark builds separate library instances and does not change the working tree during testing.

```sh
mkdir -p /tmp/rollout-test
git show 6ad3234:scripts/experimental/tt29_rally_rollout.cpp > /tmp/rollout-test/stock.cpp
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt40_rollout_bench /tmp/rollout-test/stock.cpp /tmp/rollout-test/parity.json --repeats 10
python -m scripts.experimental.tt40_rollout_bench /tmp/rollout-test/stock.cpp /tmp/rollout-test/physical.json --physical 2
python -m scripts.experimental.tt40_rollout_profile /tmp/rollout-test/stock.cpp /tmp/rollout-test/profile-stock.cpp
python -m scripts.experimental.tt40_rollout_profile scripts/experimental/tt29_rally_rollout.cpp /tmp/rollout-test/profile-fast.cpp
python -m scripts.experimental.tt40_rollout_bench /tmp/rollout-test/profile-stock.cpp /tmp/rollout-test/profile.json --candidate /tmp/rollout-test/profile-fast.cpp --profile --repeats 1
```
