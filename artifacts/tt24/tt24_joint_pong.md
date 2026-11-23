# Jointly updating one readout in native Pong and donor-arm bounce

The shared readout retained native Pong returns after real joint training. Coherent Pong credit slowed early donor acquisition relative to shuffled credit on all three paired learning streams. [Full paired record](tt24_joint_pong.json.gz)

## One policy variable in both games

The experiment exposes the same four lateral weights to reward from live native Pong and real MuJoCo donor-arm bounce. The weights act on tangential ball–blade separation, tangential ball velocity, blade displacement from its starting pose and separation multiplied by velocity normal to the blade. In Pong there is one tangent direction. On the donor arm both horizontal directions use that same four-weight formula. Direction and unit conventions are supplied. Positions use the ratio of the reference arm and Pong paddle radii, and one unit of normal speed in Pong is mapped to 3 m/s on the arm.

Pong uses the installed native 120 Hz physics and public raster at 30 Hz. The observer supplies the left image crop as the player's paddle and estimates the ball from pixels between the two paddle crops. Its fitted readout supplies a desired paddle position relative to the start. A fixed actuator servo converts positional error and estimated paddle motion to the signed native continuous drive. Positive drive moves up. This run learns movement from score and duration. It uses no intercept prediction or wall-reflection rule. [Native physics](../../src/fly_connectome/pong.py) · [Shared update implementation](../../scripts/experimental/tt24_joint_pong.py)

All arm runs share a separately reward-trained donor joint controller. The jointly learned lateral readout contributes a bounded-scale residual upstream of the donor joint output nonlinearity. A common, unscored set of 30 five-frame stock-independent probes on the three physical donors estimates the available joint combinations for two lateral paddle translations. The arm controller sees simulator metric ball, blade and motion variables. The comparison evaluates learning on additional releases on those donor bodies and uses zero stock-arm motor practice. [Physical donor head and actuator probe method](tt24_donor.md)

## Locked learning and retention budget

Both conditions start from the same four learned numbers. Nine warm updates use 990 real native Pong points. Each update uses terminal point score plus 0.43 times elapsed-play fraction. The initialized controller played 24 withheld games and returned at least once on 23, compared with 7 for the zero head. This passed the locked competence rule and allowed the joint phase to start. [Warm trajectory and native qualification](tt24_joint_pong-warm.json.gz) · [Registered schedules and budgets](tt24_joint_pong-plan.json)

The two conditions receive identical counts and ordered budgets. In each of three independent streams, eight updates use five signed perturbation pairs and the incumbent. Every candidate acts on six new actual Pong points and six scored physical bounces, two on each donor. Both tasks produce feedback on the same shared policy parameters. The joint arm uses the real score difference between the two sides of each Pong perturbation. The shuffle arm independently flips which signed direction receives that difference. Both arms use the real physical score difference. This preserves exposure and score magnitudes, changes whether native outcomes reinforce the action that earned them, and supplies no new target-selected weight or threshold.

The prespecified acquisition endpoint averages held consecutive physical rebounds after updates two, four and eight, with six held releases per donor and stream. The retention check runs the exact shared controller on the same 24 held native starts before and after both update conditions. Each retention point uses fresh native physics and camera observations under the evaluating parameter vector. [Recorded donor and native outcomes](tt24_joint_pong.json.gz)

## Learning and retention results

| Donor acquisition | Before joint updates | Update 2 | Update 4 | Update 8 | Primary early mean |
|---|---:|---:|---:|---:|---:|
| Real native credit | 6.61 | 7.26 | 7.43 | 9.35 | 8.01 |
| Shuffled native credit | 6.61 | 8.07 | 9.28 | 10.78 | 9.38 |

The early differences were −1.57, −1.57 and −0.94 consecutive rebounds across the three paired streams. In an exploratory check, the native and arm update directions pointed against each other on 13 of 23 updates with a nonzero recorded direction in both tasks. Their mean directional cosine was −0.15. These measurements are consistent with some interference between the two objectives in this four-weight coupling. Each treatment used 1,584 real Pong points and 1,584 physically scored donor drops after the common warm phase. Across the two treatments the active rollouts accounted for 2,102,902 native physics ticks and 588,860 donor action frames.

| Native tests on the same 72 held starts | Games with an own return | Later own returns | Losses | Earned points after own return |
|---|---:|---:|---:|---:|
| Shared state before joint updates | 68 | 41 | 3 | 0 |
| State after real native credit | 67 | 39 | 3 | 0 |
| State after shuffled native credit | 62 | 31 | 15 | 0 |

Pong retention here measures interception through actual play with a 0.45-speed opponent and a six-second cap. The shared state trained with true native outcomes retained its return behavior closely while improving on both tasks' common starting state on the donors. The shuffled condition learned donor bounce faster and lost more Pong returns. The tested coupling provided little support for the hypothesis that native practice would accelerate donor acquisition. [Frozen shared vectors, per-seed evaluations and paired update signals](tt24_joint_pong.json.gz)

## Reproduction

Use the repository's table-tennis Python environment and run from the repository root.

```sh
export PYTHONPATH=src MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt24_joint_pong
```
