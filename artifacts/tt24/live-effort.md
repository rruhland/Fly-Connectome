# Global effort headroom on the long altered-wrist course

The precheck found no common gain improvement. Increasing effort added five consecutive rebounds on one feasible altered launch and lost sixteen on the other. The test stopped at the registered headroom gate. [Matched physical data](live-effort-headroom.json.gz)

The carried continuous five-axis policy from the 120-second course was tested at three constant effort settings. A single scalar multiplies the ordinary normalized motor rates. Its internal issued state advances by the actual requested integral, so the intervention changes how quickly the policy moves toward its live targets. The familiar sensory and motor origins, tool-local workspace estimate, body optics and feedback coefficients are the same in every condition. Gain 1.00 directly executes the original policy. [Released carried policy](carried-wrist.md) · [Effort wrapper](../../scripts/experimental/tt24_live_effort.py)

The matched precheck uses the two already demonstrated long alternate launches, the ones that qualified for 154 and 240 rebounds in the larger course. This feasibility selection limits the result to a headroom check. Each candidate plays the first 12 seconds from the full original reset and launcher. Original variations in body, ball mass, drag and gravity apply in all conditions. Gravity changes remain outside the observation interface. Every candidate computes fresh actions from live wrist RGB and encoders. The earlier tapes validate nominal parity and replay rather than supplying actions to a live controller. [Original altered anchor 0](long-wrist-0.json.gz) · [Original altered anchor 7](long-wrist-7.json.gz)

The preregistered gains are 0.82, 1.00 and 1.18. Adaptation can proceed if one of the two nonunit gains increases consecutive rebounds by at least two in both matched anchors and by at least 20% in aggregate. This keeps any later online learning test conditional on a useful, replicated effect at the planned scale. [Frozen headroom protocol](live-effort-plan.json) · [Matched physical result](live-effort-headroom.json.gz)

| Rate gain | Anchor 0 longest | Anchor 7 longest | Sum of longest runs | Total qualified rebounds |
|---|---:|---:|---:|---:|
| 0.82 | 36 | 14 | 50 | 71 |
| 1.00 carried | 37 | 28 | 65 | 77 |
| 1.18 | 42 | 12 | 54 | 74 |

All six trajectories reached the 12-second cap. The two nominal action sequences exactly matched their corresponding archived longer trajectories through that point. All six complete physical judge records reproduced on replay. This small feasibility course tests a global rate multiplier while the actor keeps its normal closed-loop control. It did not support starting the planned model-free effort updates or their subsequent arm and Pong retention tests.

## Reproduction

Use the repository's table-tennis Python environment from its root.

```sh
export PYTHONPATH=src MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python -m scripts.experimental.tt24_live_effort
```
