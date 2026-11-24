# Carrying practice across a change of sides

The same calibrated left-ready posture was used for every point. The starting checkpoint had 238 rewarded motor examples and 38 camera-derived shot errors after 16 individual drills. Four new measured releases started four live practice rallies. A neutral receiver verified four legal first deliveries, with two starting on each side. The releases and seeds were distinct from both held schedules. On each subsequent shot the far paddle attempted the opposite side. The SO-101 kept its physical pose and motion within a point. It read stereo RGB and its encoders, and updates used legal landings and outgoing RGB. [Calibrated posture](left-ready.json) · [Starting checkpoint](ready-staged-checkpoint.json.gz) · [Frozen schedule](transition-plan.json.gz)

| Training release | Actual incoming sides | SO-101 legal returns | Alternating legal returns |
| --- | --- | ---: | ---: |
| 3238 | Right | 0 | 1 |
| 6810 | Left, right, left | 2 | 5 |
| 12835 | Right | 0 | 1 |
| 11185 | Left, right | 1 | 3 |

On 6810, the arm landed the left and right replies consecutively before the third, left-side shot passed the paddle. Each point had a single launch and ran until a fault or a 12-return cap. Action-only replay restored one starting state per point, reproduced every referee event and placement, and confirmed an increasing simulation clock through all issued commands. The final checkpoint had 241 motor examples and 41 optical errors. [Training commands and events](transition-practice.json.gz) · [Replay evidence](transition-practice-replay.json) · [Final checkpoint](transition-checkpoint.json.gz)

The checkpoint was byte-identical before and after 3238. The 6810 switch used the 16-drill starting checkpoint, with online updates during that point. A paired run from the earlier 223 movements and 24 camera errors used the same pose, release, seed, cap and online update rules. It landed left, right and left in succession, scoring three near returns and seven alternating returns. The calibrated pose and earlier controller already supported this switch on 6810. [Original-checkpoint control](transition-6810-reference.json.gz)

## Paired tests

The final checkpoint was frozen for the four periodic held games and eight full-interval games. The prior 16-drill checkpoint was the comparison. Each arm saw the same launches, same seeded opponent, same starting posture and caps of 24 and 16 alternating returns, respectively. Later trajectories could differ in response to the arm's own shot. A switch counted only after a legal near return was followed by an actual delivery across the shoulder line, at least 7.5 cm into the other side. We counted both blade contact and a subsequent legal bounce. [Experiment driver](../../scripts/experimental/tt31_transition.py)

| Held course | Checkpoint | SO-101 legal returns | Alternating legal returns | Legal replies after a side change |
| --- | --- | ---: | ---: | ---: |
| Periodic | 16 drills | 3 | 10 | 0/3 |
| Periodic | 16 drills and four rallies | 4 | 12 | 1/4 |
| Full interval | 16 drills | 4 | 16 | 0/1 |
| Full interval | 16 drills and four rallies | 4 | 16 | 0/1 |

All twelve first held deliveries matched down to the far paddle's contact, near bounce and receiving-line sample. The periodic gain came from release 12451. The retained checkpoint landed the initial left delivery and the next right delivery. The other three periodic scores and all eight full-interval scores tied their matches. Live practice used 51.4 CPU seconds. The continued and 16-drill checkpoints used 72.1 and 63.2 CPU seconds for the periodic games and 134.5 and 136.4 for the full-interval games. [Held periodic commands](transition-periodic.json.gz) · [Held full-interval commands](transition-full.json.gz) · [Paired scores and actual incoming sides](transition-summary.json)

On 12451, both checkpoints saw the same first left delivery. The first different command was at zero-based tick 59. The 16-drill checkpoint chose a newly searched stroke and the continued checkpoint chose a verified recalled stroke. Their first physical paddle contacts occurred at the same time with a 2.1 mm difference in depth. The resulting far replies differed. At 1.10 m from the net, the second incoming ball measured 22.7 cm to the right in the 16-drill point and 21.5 cm to the right in the continued point. The 16-drill planner tried seven times without producing an executable second stroke. The continued planner searched at tick 149 and the resulting stroke landed on the far half. [Decision replay and physical comparison](transition-12451-trace.json)

The three second-side misses with the 16-drill checkpoint received no joint-rate command after the near bounce. At their closest, the gaps from the ball surface to the blade were 30.9 to 35.9 cm in the command-only replay. The continued checkpoint repeated two of those misses. [Replay measurements](transition-summary.json)

## Reproduction

Use the pinned measured feed and dependencies from the [regulation endurance instructions](../tt29/endurance.md), plus the existing 16-drill and held-game archives.

```sh
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt31_transition /tmp/tt25-rallies.json practice
python -m scripts.experimental.tt31_transition /tmp/tt25-rallies.json periodic
python -m scripts.experimental.tt31_transition /tmp/tt25-rallies.json full
python -m scripts.experimental.tt31_transition /tmp/tt25-rallies.json starting-reference
python -m scripts.experimental.tt31_transition /tmp/tt25-rallies.json summary
python -m scripts.experimental.tt31_transition_trace /tmp/tt25-rallies.json
```
