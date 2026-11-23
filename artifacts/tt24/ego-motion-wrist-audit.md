# Isolating camera motion in the carried wrist velocity estimate

The wrist observer can separate apparent motion in the image from displacement caused by the calibrated camera moving with the arm. On six archived physical plays, removing the camera term raised visible-flight velocity error from 0.177 to 0.234 m/s and changed the frozen joint commands. The complete correction exactly matched the original bundled commands. The native moving-camera Pong test also favored the complete correction supplied by camera geometry. It yielded no new setting to carry into a held wrist game. [Wrist audit data](ego-motion-wrist-audit.json.gz) · [Native source pilot](pong-egomotion-source-pilot.json.gz)

## Common gain and public wrist measurements

Let the camera carry a point `s` in local coordinates to `c + R s` in the world, with camera center `c` and rotation `R` computed from own encoders and the prior public wrist calibration. At a current observation `t` and previous one `p`, the observed displacement is the sum of `R_t (s_t − s_p)` and `(c_t − c_p) + (R_t − R_p) s_p`. The former is apparent scene motion expressed in the current world axes. The latter restores the camera's own displacement. The signed gain multiplies the latter term alone. Gain one uses the original world displacement. Gain zero leaves camera-relative displacement in current world axes. This is the translation-only relation used by the native source, extended to the wrist's three-dimensional calibrated rotation and translation.

The wrist audit applies the decomposition to the existing lag-one, lag-two and lag-three differences before the frozen motion regressor. Current ball position, face pose, reference coordinates, image acceptance, dead reckoning and internal tracker histories remain those of the bundled observer. Only the exposed three velocity values can change. The five-joint motor readout, limits and prior stock and altered calibrations remain fixed. [Original carried policy](carried-wrist.npz) · [Archived public calibrations](public-body.md) · [Altered optics](public-morph.md) · [Audit code](../../scripts/experimental/tt24_ego_motion.py)

## Archived physical controls

The audit replays the first three stock and first three altered cases from the prior compound course. It checks the first 180 frames or the earlier end of each drop, including preparation. The absolute simulator ball velocity is used only after an estimate is issued, to score error on visible airborne frames. All six archived command trajectories reproduce their complete physical referee outcomes, ending ticks and recorded ball states at reached gravity switches. Alternative gains change the counterfactual motor output on the same observation and original issued servo state. They receive no new closed-loop physical play. [Original physical course](integrated-stress.md)

Feeding the same images with a fixed public camera transform produced bit-identical motion output across all gains on 851 frames. This isolates the zero-camera-motion case while the image can still change. On the actual moving-camera sequence, gain one also preserved every recorded five-joint command on all 851 audited frames. The median camera contribution to the first finite difference on those frames was 0.191 m/s.

| Gain | Pooled velocity component RMSE, m/s | RMS change of five-joint command vector |
| --- | ---: | ---: |
| −0.5 | 0.290 | 0.246 |
| 0 | 0.234 | 0.176 |
| 0.5 | 0.192 | 0.093 |
| 1 | 0.177 | 0.000 |
| 1.5 | 0.193 | 0.092 |

The velocity score covers 676 visibly airborne frames. Gain one improved on zero separately on stock (0.166 vs 0.215 m/s) and altered (0.195 vs 0.266 m/s). Native Pong terminal point totals at the same five gains were −8, −4, −5, +2 and −7 over 64 moving-camera starts. Native full correction matched the originally registered views, and the source pilot did not advance to a learned-gain lock. The wrist work therefore ended at the archived observer and command diagnostics, with zero receiving practice or tuning. [Frozen native source results](pong-egomotion-source-pilot.json.gz)

## Reproduce

```sh
MUJOCO_GL=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt24_ego_motion --output /tmp/ego-motion-wrist-audit.json.gz
```
