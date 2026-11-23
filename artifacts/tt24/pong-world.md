# Seeding 3D reward learning from a played Pong policy

Two initializers learned from the actual motion of the qualified native RGB Pong player. Neither accelerated the receiving world policy over both the cold and shuffled starts. The absolute-pose initializer matched the shuffled result. The initializer restricted to active optical displacement slowed acquisition on the held 3D courses. [Absolute-pose learning curves](pong-world-learning.json.gz) · [Active-displacement learning curves](pong-world-displacement-learning.json.gz)

## Native motion and cross-game registration

The native actor was the same RGB, anonymous-action duration player as the earlier qualification. Its value readout originally used 576 random-action native points, accounting for 1,648 simulated seconds, and its native motor fit read the observed effect of issued anonymous commands. The present collection froze that policy and rendered 112 new source-fit and 52 new source-held points. The actor generated every command. These new tapes covered 432.5 and 239.5 simulated seconds and yielded 1,872 and 1,089 paired optical inputs and future-pose observations. There were 44 first and 28 later returns in the collection used for fitting, and 28 first and 21 later returns in the native holdout. [Native score and optical-action receipts](pong-world-native.json.gz) · [Earlier policy qualification](source.md)

The native observer inferred the controllable paddle and its signed movement axis from anonymous command effects. It measured the signed tangential gap, tangential speed, and current paddle offset in that axis. The offset is relative to the observed paddle center on the first frame. Motion normal to that axis supplied distance and signed approach or departure speed. It measured a future paddle pose from native camera frames four through seven after each input. The fit targets were those actual positions, using no intercepted-ball coordinate as a target. The active-displacement variant used the observed change from the current pose and only cues where the frozen actor issued a visibly bound nonneutral command. It used 516 fitting and 276 held cues. About 83% of the held nonneutral commands agreed in sign with the subsequent observed paddle displacement.

The registration assumes that one learned scalar response along the native tangent direction can act identically on both tangent directions of the receiving world. It scales positions by the ratio between a 0.076 m receiving blade radius and the measured native silhouette half-span. It scales both velocity components so that the median native normal speed corresponds to 2.1 m/s. It bounds the observed native target within 72% of the receiving positional readout limit using a smooth size-dependent transform. These are design choices for comparing the two geometries. Twelve of the 42 coefficients can receive Pong evidence. They govern tangent position dependence on relative offset, ball velocity and blade offset. Vertical motion, tilt and the position response to tilt initialize at zero. [Absolute initialization and native-held audit](pong-world-initial.json) · [Displacement initialization and audit](pong-world-displacement-initial.json)

The shuffled control keeps the observed current actuator position and attainable displacement distribution, but exchanges the subsequent optical changes within coarse position, separation and approach strata. It matches the coherent block's total coefficient norm. A second cold control uses independent random coefficients with that same norm and sparsity. Zero cold supplies the same initial zero coefficients as the original world learner.

## Paired reward-only acquisition in cheap 3D worlds

Each experiment registered 24 independent matched learning streams, with all four initializations receiving the same world seeds and Gaussian perturbations. Updates use the original source-world reward and top signed directions. All 42 coefficients may change after training begins. The fixed Gaussian noise scale is identical across arms and updates. Each update uses 16 signed pairs over twelve world instances, for 384 scored rollouts, including three wider-profile instances for each perturbation. The world model varies gravity, drag, damping, blade size, actuator delay, speed and observation noise. There is zero new physical-arm practice. Total real simulated frames differ because the same rollout caps can end at different times. For the absolute-pose test, the Pong, shuffled and zero-cold streams averaged approximately 554,000, 555,000 and 560,000 training frames at update eight. [Absolute protocol](pong-world-plan.json) · [Independent displacement protocol](pong-world-displacement-plan.json)

The primary endpoint was mean consecutive rebounds on equally weighted nominal and stress holdouts, averaged over updates one, two, four and eight. Each displayed checkpoint used 128 fresh launches per profile and stream, each matched across arms. Later updates never receive those held rewards. Intervals resample whole paired learning streams.

| Initializer | Update 0 | Update 1 | Update 2 | Update 4 | Update 8 | Update 16 |
|---|---:|---:|---:|---:|---:|---:|
| Absolute Pong pose | 1.43 | 2.42 | 4.34 | 8.40 | 11.77 | 13.72 |
| Absolute pose shuffle | 1.40 | 2.41 | 4.46 | 8.40 | 11.69 | 13.81 |
| Absolute zero cold | 1.55 | 2.46 | 4.63 | 8.16 | 11.57 | 13.65 |
| Active Pong displacement | 1.48 | 2.20 | 3.79 | 7.50 | 11.34 | 13.64 |
| Active displacement shuffle | 1.59 | 2.45 | 4.45 | 8.29 | 11.87 | 13.80 |
| Active displacement zero cold | 1.53 | 2.28 | 4.06 | 7.81 | 11.56 | 13.70 |

Absolute Pong differed by +0.02 rebounds on the primary endpoint versus zero cold, with a paired 95% interval of -0.10 to 0.13, and by -0.01 versus shuffle, with an interval of -0.10 to 0.08. Its advantage over magnitude-matched random initialization was +0.75, with an interval of 0.59 to 0.90. The absolute fit puts most initial weight on the blade's current position. Native pose persistence also predicted future pose more accurately than the absolute fit overall.

The active optic displacement readout improved on predicting zero movement on held native cues where the actuator was commanded. Its normalized receiving-coordinate error was 10.62 mm versus 11.21 mm for persistence and 12.44 mm for shuffle. This improvement did not carry to reward acquisition in the 3D worlds. The paired early difference was -0.22 versus zero cold, with a 95% interval from -0.31 to -0.14, and -0.56 versus shuffle, with an interval from -0.68 to -0.45. Its difference from norm-matched random was -0.23, with an interval of -0.37 to -0.09. Eight-update checkpoints were saved from each protocol's predesignated first stream for reproducibility. This study leaves the existing independently learned 3D source prior as the candidate for physical-arm assessment.

## Reproduction

Use the table-tennis environment from the repository root. Set `PYTHONPATH=src`, `OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1`.

```sh
python -m scripts.experimental.tt24_pong_world_train --plan-only
python -m scripts.experimental.tt24_pong_world
python -m scripts.experimental.tt24_pong_world_train --workers 2
python -m scripts.experimental.tt24_pong_world_displacement --workers 2
```
