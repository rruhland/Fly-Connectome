# Protecting visual Pong interception during score placement

The score learner retains its acquired value table and its original shot choices. The opt-in controller checks each proposed noncenter aim against public evidence. It estimates a lower bound on opposing paddle speed from a continuous visual track and compares it with the observed ball pace. When pursuit exceeds the shot margin, the controller returns to center for that approach. With a learned request delay, a short time budget combined with unsettled motor evidence or an unstable predicted crossing can also cancel the aim. A canceled aim receives no terminal score credit.

The controller uses the existing v3 timing core and the same visual goal. The environment controls opponent speed, remap, and delay only at execution. The goal wrapper uses the timing estimate inferred by the same controller. The center arm retains its usual behavior, including the optional forecast-center path.

The frozen entry point is `pong_risk_aware_placement.build_risk_aware_controller`. It accepts the same arguments as the original builder. Use it for all compared arms, passing the corresponding score table and exploration setting. The original builder remains available for the legacy behavior. The source is frozen at commit `bbd7f2fe9db6f3ef631dd1347e8a1d8e4fa508df`, SHA-256 `52819308f61bafada00c97b39d0afff84a165c9102dbd1823eb9f9b6244c7312`. [Controller](../../scripts/experimental/pong_risk_aware_placement.py)

## Paired development

Each pair began with the same native serve and the same timing checkpoint. The learned and guarded arms used the frozen score table. The difficult cases reused all 12 faster-opponent starts from the mixed course with the corresponding carried timing state and a fixed 16-second limit. The ordinary and faster zero-delay groups used eight fresh starts each and a 12-second limit.

| Condition | Aim | First returns | Later returns | Wins after own return |
| --- | --- | ---: | ---: | ---: |
| Faster opponent, carried remap and 12-camera delay | Guarded | 7/12 | 24/25 | 0 |
| Faster opponent, carried remap and 12-camera delay | Unguarded | 7/12 | 8/14 | 0 |
| Faster opponent, carried remap and 12-camera delay | Center | 7/12 | 26/26 | 0 |
| Faster opponent, zero delay | Guarded | 8/8 | 17/17 | 0 |
| Faster opponent, zero delay | Unguarded | 8/8 | 16/16 | 0 |
| Faster opponent, zero delay | Center | 8/8 | 20/20 | 0 |
| Ordinary opponent, zero delay | Guarded | 7/7 | 5/6 | 5/8 |
| Ordinary opponent, zero delay | Unguarded | 7/7 | 5/6 | 5/8 |
| Ordinary opponent, zero delay | Center | 7/7 | 18/18 | 0 |

In the difficult condition, the guard avoided five game losses seen with unguarded scoring and incurred one additional game loss relative to center. All three arms lost the same five openings. At zero delay against the faster opponent, all arms returned every observed opportunity. Their shot trajectories produced different numbers of later opportunities before the time limit. Against the ordinary opponent, the guarded and unguarded request hashes matched on all eight starts. The eighth start ended before an incoming interception opportunity. The guard is frozen for an independent comparison with the same trained, center, and exploration arms. [Development receipts](/workspace/m1b-experiments/pong-risk-aware-development-20261119-r1)
