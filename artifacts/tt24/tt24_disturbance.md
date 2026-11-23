# Does the live rally pupil correct a disturbed flight?

The frozen 48-release RGB-to-joint fit plays from the first frame on 24 new receiving starts, using the original 240 × 180 stereo cameras and live far-arm opponent. There are six starts per cohort and lane-width stratum. Within each block, the experiment assigns one start each to lateral ball velocity changes of −0.30, −0.22, −0.14, +0.14, +0.22 and +0.30 m/s. The policy weights stay fixed, and only pixels, joint readings and its own issued commands reach the pupil. [Frozen fit and evaluation](rally-bank.md)

Each nominal live point establishes that start's near-arm command tape. A repeat applies that tape with the live far arm and reproduces the nominal result. In the two disturbed conditions, one branch continues computing near rates from live observations and the other reuses the nominal near-arm rates. The far arm responds live in both branches. If a disturbed point lasts longer than the nominal tape, the frozen near arm holds its final position target.

The opening test changes the actual ball's lateral velocity at 0.20 simulated seconds, after the initial far-side bounce. The continuation test uses starts with a legal opening and a subsequent far paddle strike. It changes lateral velocity two control ticks after that strike in separated inbound flight. Both branches start from release, share the same physics and commands through the kick frame, and receive no disturbance flag. The perturbation changes only the lateral velocity of the physical ball. Legal near returns are counted after the kick. The early motor comparison ends at the earlier near contact, or after 21 control intervals if there is no earlier contact.

Each recorded two-arm tape is then replayed with the identical initial state and impulse. Replay checks the sampled physics hash, full referee event trace, terminal frame and result. The zero-kick near-arm control uses the live far arm and checks both command streams as well as the same physical evidence. [Experiment and replay code](../../scripts/experimental/tt24_disturbance.py)

## Held results

| Inbound flight | Eligible starts | Nominal with a legal return | Disturbed live with a legal return | Disturbed tape with a legal return | Live-only / tape-only |
| --- | ---: | ---: | ---: | ---: | ---: |
| Early incoming serve | 24 | 12 | 9 | 2 | 7 / 0 |
| First opponent reply | 12 | 4 | 5 | 1 | 4 / 0 |

After the early kick the live pupil made 23 legal near returns across the 24 releases, and the disturbed tape made three. After the kick on the first reply, the live pupil made 13 and the disturbed tape made one. The corresponding nominal counts were 18 and six. The replay comparison reproduced all 96 two-arm tapes. All 24 nominal near-command tapes reproduced their live-opponent controls exactly.

The ball had moved laterally by a median 5.5 cm relative to nominal when sampled up to eight control ticks after the kick. A near command diverged before contact or the early motor cutoff in every pair. The median first divergence was one control tick for the serve and 7.5 for the reply, or about 33 ms and 250 ms. The median maximum absolute difference in normalized joint rate within that window was 0.84 and 0.81, respectively. All 11 live-only recoveries made their first contact while the original nominal tape was still available. Fifteen early disturbed serves and seven disturbed replies still produced no legal live near return. [Paired steps, events and motor data](tt24_disturbance.json.gz)

In the side-by-side replay, seed `1690569978` shares a legal opening and takes a −0.22 m/s lateral kick at 1.40 simulated seconds. The live pupil then makes four legal near returns, and the nominal command tape makes zero. Both panels show the simulator's native broadcast RGB with the score drawn outside the images. The rendered live replay also reproduces the pupil's original stereo RGB digest. [Side-by-side clip](tt24_disturbance.mp4) · [Video replay receipt](tt24_disturbance-video.json)

## Reproduce

```sh
MUJOCO_GL=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt24_disturbance --output /tmp/tt24_disturbance_reproduction.json.gz
MUJOCO_GL=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt24_disturbance_movie
```
