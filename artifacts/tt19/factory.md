# Factory-calibrated SO-101 play

Both controllers received the published SO-101 kinematic chain and joint limits, the fixed robot-to-table transform, the tool mounting dimensions, and calibrated fixed cameras. The player read 30 Hz joint encoders, RGB pixels, service role and point reward. It triangulated the orange ball and predicted interception from its observed motion. A constrained inverse kinematic controller aimed the blade and converted planned stroke speed into joint commands. The wrist camera was available for inspection. The controller used the two fixed RGB cameras for ball tracking. [Controller](../../scripts/experimental/tt19_factory.py)

The gantry condition carried the fitted flight law and 32 recorded racket impulse examples. The nominal condition used the same factory geometry, vision, actuator caps, shot planner and exploration with the analytic flight and racket priors. Both conditions used the same seeds within each comparison. The tuning policy ranked attempts using point reward and RGB distance of the outgoing ball from a landing target. It selected an intercept offset, swing-rate scale, pitch adjustment and timing offset for each service role. Each paid practice frame involved a physical command or a zero command while waiting for the point to end. Controllers froze their selected parameters before the complete games. [Flight source](flight-source.json.gz) · [Trial and replay](../../scripts/experimental/tt19_factory_trial.py)

| Controller | Charged practice frames | Held score | Legal learner serves | Legal learner returns | Receiving points with contact |
| --- | ---: | ---: | ---: | ---: | ---: |
| 15 Hz RGB, gantry | 958 | 1–11 | 4 | 0 | 1/6 |
| 15 Hz RGB, nominal | 927 | 1–11 | 6 | 1 | 2/6 |
| 30 Hz RGB and encoder roll compensation, gantry | 646 | 0–11 | 3 | 0 | 6/6 |
| 30 Hz RGB and encoder roll compensation, nominal | 760 | 0–11 | 2 | 1 | 5/6 |
| Feasible interception search, gantry | 641 | 2–11 | 1 | 1 | 6/7 |
| Feasible interception search, nominal | 627 | 1–11 | 1 | 1 | 5/6 |

Each pair had its own matched training and held seeds. Within each pair, the conditions differ in the initial flight and racket priors. The separate full-rate trial changed observation cadence, stroke update timing, roll compensation and game seeds. The final pair also searched nearby interception times and blade angles subject to the supplied joint constraints. Each condition landed a legal return in its held game. The gantry condition converted its return into a point. The sample supports physical feasibility and leaves a transfer advantage and a full-game win unestablished. [15 Hz gantry game](factory-gantry-held.json.gz) · [15 Hz nominal game](factory-nominal-held.json.gz) · [30 Hz gantry game](factory-gantry-stabilized-held.json.gz) · [30 Hz nominal game](factory-nominal-stabilized-held.json.gz) · [Feasible gantry game](factory-gantry-reachable-held.json.gz) · [Feasible nominal game](factory-nominal-reachable-held.json.gz)

The shared engineering pilot used eight physical points and 319 control frames under the earlier end-line scoring. The matched experiments used strict end-line calls, distinct practice and held starts, no target-side dry calibration, and raw observation-to-command replay on selected points where marked. Every reported game stores the commands, public encoder readings, RGB sighting traces, score and full evaluator events. The score verifier reruns each physical tape. [Development receipt](factory-development.json.gz)

```bash
MUJOCO_GL=egl LP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt19_factory_trial artifacts/tt19 gantry --reachable --train 7 --held 1 --verify-rgb
```
