# Native Pong to SO-101 optical motion transfer

The shared learner predicts the next two-coordinate displacement of an image region from its three preceding displacements. Native Pong supplies ordinary rendered frames and issued motor actions. A neutral extractor tracks compact foreground changes across adjacent frames. It uses no role annotations, object correspondence between games, simulation position or reward as a training target. The same three numeric weights operate on either image axis and stay independent of object appearance or screen location. The Pong policy uses random drive bouts during data collection. [Model](../../scripts/experimental/tt22_transfer.py) · [Native training and scored trial](../../scripts/experimental/tt22_transfer_trial.py)

The three conditions start from chronological Pong motion, constant-velocity scratch, or the same Pong data with future displacements randomly reassigned to preceding tracks. All see the same chronological, unlabelled target practice openings with the SO-101 arm at rest. They update after each opening. The controls share the prior strength, target training samples and test schedule. Forecast error is measured on a separate bank of unlabelled motions from new openings, in units of each track's prior displacement scale. The practice receipt includes every numerical state and the native Pong holdout errors alongside the target curve. [Training curves and checkpoints](transfer-practice.json.gz)

In full scored starts the frozen destination receiver uses the existing SO-101 image proposals, table registration, geometry and factory stroke planner. The prior can smooth a supported image proposal or fill one missing view if the other camera still observes the candidate. Both resulting view rays must agree before the factory receiver sees the modified table-relative observation. This is a test of implicit motion transfer into an existing receiver. Choosing a legal stroke and identifying the ball in the TT perception system were supplied parts of the receiver. [Scored action and referee tapes](transfer-held.json.gz)

An initial extraction pilot at 80 × 60 produced four target training tracklets and one validation tracklet. The scoring comparison uses the revised 160 × 120 extractor and an independent held receiving seed. [Discarded extraction pilot](transfer-pilot-practice.json.gz)

## Learning curves

The native source consisted of 24 instances, 10,368 simulated controls, 3,480 observed images and 7,162 motion windows. Its holdout contained 1,878 windows from eight disjoint native instances. The target adaptation used 473 windows over ten openings and 317 target control frames. The six target validation openings supplied 292 unseen motion windows and never updated the weights. [Training and validation receipt](transfer-practice.json.gz)

| Target controls | Validation game | Chronological Pong | Scratch | Time-shuffled Pong |
|---:|---|---:|---:|---:|
| 0 | Native Pong | 0.450 | 0.563 | 1.020 |
| 128 | Native Pong | 0.488 | 0.540 | 0.565 |
| 317 | Native Pong | 0.495 | 0.530 | 0.525 |
| 0 | SO-101 | 1.069 | 1.010 | 1.213 |
| 128 | SO-101 | 0.992 | 0.973 | 0.987 |
| 317 | SO-101 | 0.978 | 0.972 | 0.972 |

Chronological Pong retained the lowest native forecast error after the target updates. Its transfer to the different target image dynamics left target forecast error slightly above scratch throughout the sampled curve. [Checkpoint trajectories](transfer-practice.json.gz)

## Scored SO-101 receiving

| Initialization | First legal return | All legal returns | Points won |
|---|---:|---:|---:|
| Chronological Pong | 11/16 | 11 | 7/16 |
| Scratch | 12/16 | 13 | 8/16 |
| Time-shuffled Pong | 11/16 | 12 | 8/16 |

Pong initialization changed 10 action tapes relative to scratch. It gained two points and lost three on paired starts. The only first-return disagreement favored scratch. The conditions changed hundreds of table-relative readings by at least a millimeter, and none filled a missing view. The trial therefore tested differences in supported image smoothing. All 48 physical tapes replayed with matching events and scores. [Matched receiving trials](transfer-held.json.gz)

## Reproduce

```sh
export PYTHONPATH=.:src:scripts/experimental MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 LP_NUM_THREADS=1 LIBGL_ALWAYS_SOFTWARE=1
python -m scripts.experimental.tt22_transfer_trial all
python -m scripts.experimental.tt22_transfer_trial verify
```
