# Pong spatial preference as a motor command

The SO-101 stroke was inherited from the open-loop search that selected a clock-driven joint sequence from native referee scores. Its original centered course had legal learner returns on 12 of 16 starts. The present trials keep that sequence, its native .45 serve spread and nominal speed, and the original base rival. The evaluator varies the launch's lateral family before each point. The player gets the two native RGB streams through the existing ball and blade segmentation and table calibration. [Reward-trained stroke](openloop-practice.json) · [Controller](../../scripts/experimental/tt22_motor_transfer.py)

The controller locks a lateral goal from the early RGB track. It applies the two Pong gains to relative lateral displacement and motion extrapolated to the ready blade's depth. The same transfer convention omits the source intercept for both Pong and cold coefficients. The requested displacement passes directly to the fixed ready and peak poses. Action selection uses only vendor forward and inverse kinematics for those positions and face directions, subject to physical actuator limits. Camera capture ends as soon as the goal locks. The inherited clock drives joint requests for the rest of the point. [Pong qualification](../tt3/discovered-source-refinement.json.gz) · [Native trial](../../scripts/experimental/tt22_motor_transfer_trial.py)

## Six practice points and six held starts

The scratch and Pong-initialized banks played the same six practice starts. They updated only the chosen option from its actual learner legal-return count. Scratch started with six broad signed options. Pong started with the same broad options and four Pong finalists, carrying their scored preferences. Both recorded their first legal return on their third practice point. On those six practice points the Pong-initialized player made six first contacts and scratch made three. Each made one legal return.

| Frozen policy for the six harder held starts | Points with robot contact | Points with a legal learner return | Points won |
|---|---:|---:|---:|
| Original centered macro | 1/6 | 0/6 | 0/6 |
| Initial broad cold selection | 6/6 | 1/6 | 0/6 |
| Native Pong selection | 6/6 | 1/6 | 0/6 |
| Scratch after the six practice points | 0/6 | 0/6 | 0/6 |
| Pong-initialized after the six practice points | 6/6 | 1/6 | 0/6 |

The Pong-initialized endpoint chose the same broad signed option as the initial cold control. The initial Pong selection requested up to 172 mm of lateral displacement. Both initial spatial policies reached all six balls, and each made its only legal return on held point three. Most remaining contacts landed on the learner's side. None of the policies reached a second legal learner return. The 42 scored tapes replayed exactly. [Pilot receipt](motor-transfer.json.gz)

## Direct timing screen

The next experiment tried five fixed clock offsets on four fresh matched starts per gain. Each branch selected the timing by real learner returns and used point reward only to break ties. Identical pre-cue controls let the candidates reuse their matched native RGB prefix. The audit checked that the first differing command followed the common optical choice.

| Initialization | −6 frames | −3 frames | Original | +3 frames | +6 frames |
|---|---:|---:|---:|---:|---:|
| Cold | 0/4 | 0/4 | 1/4 | 1/4 | 0/4 |
| Pong | 0/4 | 0/4 | 1/4 | 0/4 | 0/4 |

Both branches kept the original timing. Scratch tied at +3 frames and the smaller timing change broke the tie. The two-of-four legal-return gate for the 24-start holdout remained out of reach. The clock screen produced zero second learner returns or wins, and all 40 screened tapes replayed exactly. Moving the macro raised contact rates with both initial preferences. Both frozen gains produced the same number of legal returns. [Timing receipt](motor-transfer-phase.json.gz)

## Reproduction

Run from the repository root with the MuJoCo environment active.

```bash
export MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt22_motor_transfer_trial
python -m scripts.experimental.tt22_motor_transfer_phase --held 24
```
