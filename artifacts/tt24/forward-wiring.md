# Causal forward identification of remapped motor commands

The nominal model predicts the next change in joint velocity from current encoder angles and rates, four previous velocity differences and four issued-command taps. In a remapped interface, the same hidden five-by-five matrix multiplies every issued command tap. The motor-effect portion becomes a 25-parameter linear fit to causal encoder innovations. The stock and alternate forward models are separate and the compound controller chooses between them using the existing wrist image identity. The motor map can be recalled across the two bodies. [Identification code](../../scripts/experimental/tt24_forward_wiring.py) · [Compound controller](../../scripts/experimental/tt24_forward_wiring_game.py)

The model fits only the previously issued commands and measured angles and velocities. Its shared stock archive has 1,920 ball-free encoder frames. The stock and altered models additionally reuse 337 and 154 generic probe frames and 1,533 and 1,234 frames of their own previously issued play commands. The first alternate forward model used 480 new ball-free frames from four existing self-issued probe traces. The contrast iteration added 48 repeated equal-and-opposite command pairs per body at identical pre-pulse encoder histories. It used 1,312 stock and 1,384 alternate frames, including both branches and their matching prefixes. A separate small diagnostic used 60 one-step measurements, evenly split between the two bodies. No reward, hidden moving pose, or reference wiring was used in any fit. The compound task keeps the previously demonstrated stock motor readout frozen.

## Non-task identification

The contrast iteration evaluated 12 fresh dense remaps under each of the two bodies. Each trial measured up to 64 ordinary motor frames and predicted encoder innovations on 40 independent test commands. Repeated paired directions were equal effort and common to both estimators. Relative wiring error is the Frobenius distance from the simulator's known matrix divided by its norm. The matrix was read only for this audit.

| Measured frames | Stock forward | Stock inverse observer | Alternate forward | Alternate inverse observer |
|---:|---:|---:|---:|---:|
| 8 | 1.100 | 1.427 | 1.158 | 1.427 |
| 16 | .819 | .914 | .948 | 1.052 |
| 32 | .471 | .480 | .710 | .690 |
| 64 | .361 | .368 | .607 | .601 |

At 16 frames the paired mean error gains over the existing inverse observer were 0.095 on stock with a 95% bootstrap interval of 0.055 to 0.135 and 0.104 on alternate with an interval of 0.036 to 0.170. At eight frames the inverse observer had yet to receive its first usable causal observation because its longest input tap is nine frames. By 32 frames the relative wiring errors were close. Forward held innovation prediction was also lower, although that metric uses the same nominal forward law for both estimators. [Initial dry receipt](forward-wiring-identification.json.gz) · [Contrast dry receipt](forward-wiring-contrast-identification.json.gz)

The small diagnostic found that without contrasts the alternate model's instantaneous-action kernel differed by 52% from directly measured one-frame command contrasts. Its lowest singular value was 0.27 versus 1.29 under those direct measurements. The extra contrast calibration raised the fitted minimum to 0.49. Weak response directions still limit inversion.

## Paired wrist gameplay

The first controller estimated a new matrix after encoder surprises and injected balanced directional probes. It used 238 probe frames and scored 44 summed longest streaks on 15 archived compound releases, versus 88 for the body-specific inverse and 55 for the vision-only control. A conservative iteration used the contrast-calibrated model, zero inserted probes and several strictly later observations to accept a map. It ran on two other archived compound streams without reward updates or adaptation from the earlier stream.

| Controller | Stream 1 streak sum | Stream 2 streak sum | Changed middle six, both streams |
|---|---:|---:|---:|
| Conservative forward | 47 | 51 | 7 |
| Body-specific inverse observer | 76 | 86 | 45 |
| Shared inverse observer | 89 | 67 | 47 |
| Vision only | 73 | 66 | 17 |
| Familiar wiring | 121 | 99 | 94 |

The compound experiment alternates camera geometry and drive regimes independently of the hidden command map. The conservative controller selected a recalled or newly estimated nonidentity map on 97% and 100% of changed-wiring frames in the two held streams. Its mean relative matrix error on those frames was 0.82 and 1.04. Neither stream reached a relative matrix error of 0.25 on any changed frame. The controller often detected change without identifying a usable inverse. The modest dry advantage therefore did not yield faster physical rebound recovery. Each new motor tape reproduced its full physical judging metrics. [Probe pilot](forward-wiring-compound-0.json.gz) · [Conservative stream 1](forward-wiring-compound-1-conservative.json.gz) · [Conservative stream 2](forward-wiring-compound-2-conservative.json.gz)

## Reproduce

Use the repository source on the Python path and its existing MuJoCo environment.

```sh
export PYTHONPATH=src OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
python -m scripts.experimental.tt24_forward_wiring --alternate-probe-replays 4
python -m scripts.experimental.tt24_forward_wiring --alternate-probe-replays 4 \
  --contrast-pairs 48 --seed 7238625 \
  --model artifacts/tt24/forward-wiring-contrast-model.npz \
  --out artifacts/tt24/forward-wiring-contrast-identification.json.gz
python -m scripts.experimental.tt24_forward_wiring_game --stream 0
python -m scripts.experimental.tt24_forward_wiring_game --stream 1 --conservative \
  --model artifacts/tt24/forward-wiring-contrast-model.npz
python -m scripts.experimental.tt24_forward_wiring_game --stream 2 --conservative \
  --model artifacts/tt24/forward-wiring-contrast-model.npz
```
