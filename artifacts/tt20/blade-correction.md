# Optical command correction on the loaded SO-101

A single 72-frame movement supplied command rates and native stereo measurements of the blade rim and gripper tip. The fit used the first 48 frames and reserved the last 24 to check the optical prediction. A separate nominal CAD simulation supplied the reference motion. The fit never used actual joints, payload parameters, ball measurements or scoring outcomes. It learned five small rate gains and five timing offsets. [Calibration and point tapes](blade-correction.json.gz) · [Correction code](../../scripts/experimental/tt20_blade.py)

| Optical error | Nominal model | Fitted model |
| --- | ---: | ---: |
| Fitting frames | 0.106 px RMS | 0.100 px RMS |
| Reserved frames | 0.744 px RMS | 0.976 px RMS |

The fitted offsets ranged from −0.044 to +0.025 control frames. The frozen inverse filter adjusted the motor commands sent by the existing receiver. Its command accumulation absorbed clipping and returned to the intended neutral target. Both conditions used the same gantry priors, nominal motion catalogue, 80 g payload and native wrist-plus-fixed RGB during play. The trial charged all 72 target calibration frames, including the reserved frames. [Trial and replay](../../scripts/experimental/tt20_blade_trial.py)

| Receiving starts | Frozen legal returns | Corrected legal returns | Frozen points won | Corrected points won |
| --- | ---: | ---: | ---: | ---: |
| 12 previously published payload starts | 7/12 | 7/12 | 3/12 | 1/12 |
| Six fresh paired starts | 4/6 | 4/6 | 3/6 | 3/6 |

Both conditions returned legally on the same individual starts in each block. The correction increased optical prediction error on the reserved movement and yielded no additional legal returns. All 36 physical point tapes reproduced their original scores and events, including the frozen tapes reused from the prior 12-start assay. Three focused tests passed. [Earlier payload comparison](../tt19/direct-receive-payload.md) · [Paired receipt](blade-correction.json.gz)

```bash
MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt20_blade_trial artifacts/tt19 artifacts/tt20
```
