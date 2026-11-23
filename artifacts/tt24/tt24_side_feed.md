# Side-fed bounce on the stock wrist

The ordinary stock bounce course starts close to the blade, with at most 39 mm of initial offset on either horizontal axis and 0.075 m/s of speed on either axis. This course releases the ball from the side toward a location around the 76 mm blade edge. It tests the opening interception separately from subsequent bouncing. [Original release and judge](../../scripts/experimental/tt24_bounce.py)

Across the six frozen policy banks, all four lateral additions and both donor references caught the same two releases missed by stationary. Native Pong held onto one more first catch than each other bank across the full roster. Each difference came from a release that stationary caught. [Complete held trajectories](tt24_side_feed.json.gz)

## Release and feasibility

Tosses are balanced across the four cardinal directions, with small angular and sideways perturbations. Release heights are 0.22–0.30 m over the initial blade, inward speeds are 0.16–0.25 m/s, and initial vertical speeds are −0.12 to −0.015 m/s. The release generator uses a free-flight approximation to set horizontal distances so that the projected arrival is within one of three specified bands near the blade edge. The stock MuJoCo model determines the actual ball trajectory and contacts. Arm pose, ball spin, drag, damping, stock geometry and the qualifying rebound rules follow the original held fixture. [Locked releases and reach bands](tt24_side_feed-plan.json)

The reach bands were assessed using stationary control and a previously qualified reward-only direct joint controller. The donor was tested with both metric simulator observations and the frozen public stock wrist and encoder observer. The preregistered selection used only these outcomes. A band needed at least four stationary misses, at least three qualified metric-donor openings with an advantage over stationary, at least two public-donor catches of stationary misses and at least two public-donor opening streaks of two or more rebounds. The first band, at 82–99 mm projected arrival radius, produced no passive catches and two per donor. The second band, at 68–83 mm, passed. Stationary, metric donor and public donor caught 3/8, 4/8 and 4/8 respectively. The public donor caught three stationary misses and produced four opening streaks of two or more rebounds. In the held roster the selected band produced initial lateral release radii from 108 to 132 mm. The third band was left unused under the fixed first-passing rule. [Feasibility action traces and judge events](tt24_side_feed-feasible.json.gz) · [Donor qualification and prior physical practice](tt24_donor.md)

## Frozen comparison

The six historical source-learning streams each supply four independently balanced physical releases. Each release is played by all four source-bank policies, the unchanged wrist donor, the metric-observation donor diagnostic and a stationary control. The source-bank checkpoints followed eight equal reward-only updates in cheap 3D worlds, starting from coherent native Pong, shuffled native Pong, all-zero or equally sized random initial weights. [Source learning and carry geometry](pong-world-strong.md) · [Pinned checkpoint bank](pong-world-strong-bank.json)

All four source-bank modes share the same frozen donor joint controller and stock wrist observer. Their only change is an additive lateral correction to the direct joint target, derived from the first two task-space position outputs of the respective bank checkpoint. The correction is mapped into joints with the same damped local body relation estimated at release from the existing public calibration. Each component is bounded at 0.18 radians and relaxed with a 0.40 update fraction. It starts fading when camera motion rises faster than 0.24 m/s after observed descent beyond 0.26 m/s, or after 20 control frames. Those supplied limits are common to all bank conditions. The donor continues to receive its full observation and command the arm on each frame. The public observer was previously fitted from eight stock calibration drops. This evaluation adds zero stock motor practice or parameter updates. [Observer calibration](public-body.md)

The primary endpoint is a physically qualified rebound from the very first blade impact. The follow-through endpoint requires that the second consecutive impact also produce a qualified rebound. The original longest consecutive streak and total are recorded separately, including any later recovery. Every complete command tape is replayed without rendering against the same release and must reproduce the full original judge record. [Experiment implementation](../../scripts/experimental/tt24_side_feed.py)

| Frozen treatment | First qualified catch / 24 | Two consecutive from first / 24 | First catch on eight stationary misses | Sum of longest runs |
|---|---:|---:|---:|---:|
| Wrist donor + coherent Pong lateral | 18 | 10 | 2 | 46 |
| Wrist donor + shuffled lateral | 17 | 9 | 2 | 49 |
| Wrist donor + zero-initialized lateral | 17 | 9 | 2 | 40 |
| Wrist donor + norm-matched random lateral | 17 | 8 | 2 | 60 |
| Wrist donor alone | 12 | 7 | 2 | 29 |
| Metric-observation donor | 12 | 8 | 2 | 55 |
| Stationary | 16 | 11 | 0 | 41 |

Every donor and lateral condition followed both newly caught stationary misses with a second qualified rebound. Summed opening runs on those two releases were 5 for Pong, 14 for shuffle, 13 for cold and 7 for norm-matched random. Each of the three lateral controls tied coherent Pong on five of six matched source streams for opening catches. The lateral corrections helped the wrist donor retain passive-catchable releases, and the shared improvement across controls leaves no observed advantage specific to native Pong on releases requiring an active interception. Across the full roster, norm-matched random had the highest total while stationary reached a second consecutive rebound on the most starts.

One direction presented a physical obstruction. All six negative-y tosses in each condition terminated on the stock handle or gripper jaw. Those launches remain in the reported denominator. The other two stationary misses were both caught by every active condition. As a result, this held roster probes a small number of reachable catches that actually required movement. All 168 held command sequences and 48 feasibility command sequences reproduced the full physical judge record. [Paired outcomes and conditional counts](tt24_side_feed.json.gz)

## Reproduction

Use the repository's table-tennis Python environment from the repository root.

```sh
export PYTHONPATH=src MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt24_side_feed --feasibility-only
python -m scripts.experimental.tt24_side_feed --workers 2
```
