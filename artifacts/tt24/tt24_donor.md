# Reward-only transfer from physical hinged donors

The control head begins at zero on three simulated physical SO-101-like donors. Each has an upper link 15% longer than stock and a tilted elbow axis. The designs differ in elbow-axis tilt, upper-link mass and inertia, physical wrist payload, servo gain and force. All use the changed ball and drive regime, with randomized drop offsets, velocity, spin, drag and contact damping. Finite-force joints, the welded paddle, table, arm and camera collisions remain in MuJoCo. This is bouncing experience on different physical bodies. The policy receives no Pong experience or teacher actions.

All five joint outputs use one learned 5 × 20 head. Its continuously sampled features include relative metric ball and paddle positions, ball velocity, paddle normal, and smooth position and motion products. The readout maps them to bounded joint target offsets, converted to normalized rates using its own issued commands. The direct controller has no task-space inverse, phase script, stored action path, or explicit impact scheduler. Donor observations contain privileged physical measurements. The native rollouts use the same installed MuJoCo fixture. Twelve paired native and Python rollouts matched rebound counts, consecutive runs, elapsed control steps and contacts before training. [Native rollout](../../scripts/experimental/tt24_donor_native.cpp)

The first search used 1,248 scored donor practice drops. On 24 independent donor drops its frozen readout averaged 2.125 consecutive rebounds versus 2.333 stationary, with a maximum of four. The gate prevented running that readout on stock. [First search](tt24_donor.json.gz)

The new refinement starts at zero again and trains on wider donor releases. Before reward search it issues 30 five-tick motor probes independently of ball motion. The change in observed paddle position and normal provides a basis for coupled search perturbations around a stable posture. The basis proposes weight changes and is absent from the runtime controller. Every motor coefficient is selected by actual task reward. The selection objective weights qualified rebounds and longest consecutive runs, with a small survival term. Separate donor development drops choose the best head. A new set of 24 donor releases independently compares that head with stationary. [Training and assessment runner](../../scripts/experimental/tt24_donor_trial.py)

The three donors use payloads of 30, 42 and 52 g and upper-link mass factors of 0.91, 1.04 and 1.15. Their changed-regime drive multipliers are 1.08, 0.99 and 0.93. The moving plants and elbow axes differ, while their five normalized joint channels retain the same order and rate units. The campaign uses 120 donor-only rounds, each scoring two newly drawn drops per body for 25 candidates, then ranking three candidates on the fixed eight-release development roster per body.

The target head receives zero stock motor-practice drops. The stock wrist observer uses a prior public optical and encoder calibration from eight bounded physical probes, without privileged stock pose labels. No stock outcomes choose the head. The stationary comparison and clean-observation diagnostic use the same stock releases and referee. Clean observations are a diagnostic comparison for sensor quality. [Stock public observer](public-body.md)

## Locked donor assessment

| Donor | Fresh drops | Learned mean consecutive | Stationary mean | Learned at least 10 |
| --- | ---: | ---: | ---: | ---: |
| 30 g wrist payload | 8 | 15.625 | 2.50 | 8 |
| 42 g wrist payload | 8 | 11.25 | 2.00 | 6 |
| 52 g wrist payload | 8 | 10.125 | 2.25 | 3 |
| Combined | 24 | 12.333 | 2.25 | 17 |

The refined search used 18,000 scored donor drops and 2,651,727 control frames. Donor validation added 8,664 runs and 2,143,144 frames, using 24 repeated validation release contexts. The shared body interaction and policy selection used about 22.2 hours of simulated control, in 559 wall seconds for the refinement on one CPU process. The separate failed pilot took 1,248 training drops and 975 validation runs. Performance qualifies as transfer with substantial donor experience. [Donor training and held replays](tt24_donor-refined.json.gz)

The selected head first appeared at round 36, after 5,400 practice drops and 630,763 control frames, plus 2,616 validation runs. The archive preserves the zero head and every donor-development improvement up to round 36 with its cumulative exposure. At independently scored donor impacts, the paddle contact point was rising faster than 0.055 m/s on 321 of 335 learned rebounds and zero of 54 stationary rebounds. The median was 0.121–0.126 m/s across the three learned donors. [Development checkpoints](tt24_donor-development-weights.npz) · [Impact audit](tt24_donor-extended.json.gz)

## Five-second stock deployment

| Stock body, 16 fresh matched drops | Mean consecutive | Maximum | At least 10 |
| --- | ---: | ---: | ---: |
| Donor head with public wrist observer | 11.938 | 17 | 12 |
| Same donor head with clean simulated observations | 12.188 | 19 | 12 |
| Stationary | 2.563 | 5 | 0 |

With the public wrist observer the frozen head beat stationary on 13 drops, tied two and lost one. All 96 donor and stock action traces reproduced the actual terminal results and complete rebound events. The head used all five normalized actuator outputs in held play. [Frozen head](tt24_donor-refined.npz) · [Held trajectories](tt24_donor-refined.json.gz)

On stock the public-wrist controller struck 175 of its 191 scored rebounds with a contact point rising faster than 0.055 m/s, versus zero of 41 stationary rebounds. The median upward impact speed was 0.129 m/s, and 12 drops had at least ten of these rising impacts. [Recorded impact speeds](tt24_donor-extended.json.gz)

## Changed body and physics

A separate 12-drop course uses an unseen elbow-axis direction, new upper-link mass and wrist payload, altered actuator drive and force, 0.84 times Earth gravity and 1.35 times the existing changed-regime drag. It retains the fixed donor readout and the original stock wrist observer. The public wrist branch averages 5.333 consecutive rebounds with a maximum of 14 and three drops reaching ten. The clean diagnostic averages 3.25 with a maximum of 12, and stationary averages 1.50 with a maximum of two. This is one held stress comparison. The frozen stock observer may have systematic errors on the changed body, and the higher public score here gives no general ordering between clean and estimated inputs. [Changed-body tapes](tt24_donor-extended.json.gz)

## Reproduce

```sh
MUJOCO_GL=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt24_donor_trial --refine --rounds 120 --directions 12 \
  --examples 2 --noise .18 --train-cap 260 --output /tmp/tt24_donor-refined.json.gz \
  --checkpoint /tmp/tt24_donor-refined.npz
MUJOCO_GL=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt24_donor_assess --training /tmp/tt24_donor-refined.json.gz \
  --checkpoint /tmp/tt24_donor-refined.npz --output /tmp/tt24_donor-extended.json.gz \
  --archive /tmp/tt24_donor-development-weights.npz
```
