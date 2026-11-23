# Native Pong credit in planar gravity

The same native Pong visual readout accelerated a fresh learner when the new body had a fixed camera and simple direct motor rates. With eight practice drops, the lateral learner reached **1.38 verified rebounds per held drop**. The cold and source-label-shuffled learners reached 0.26 and 0.64, and a stationary paddle reached 0.31. A stricter stage required upward paddle motion for any verified rebound. After sixteen practice drops in that stage, the native-source learner produced **140 powered rebounds in 160 held drops**, including 38 drops with multiple rebounds and a maximum of four. [Lateral and two-axis trials](planar-planner.json.gz) · [Powered trials](planar-powered.json.gz)

| Stage and practice budget | Native Pong | Cold | Shuffled source | Stationary | Paired held drops per treatment |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lateral, 8 drops | 1.383 | 0.258 | 0.638 | 0.312 | 240 |
| Lateral, 16 drops | 1.796 | 0.308 | 0.925 | 0.312 | 240 |
| Lateral, 24 drops | 1.729 | 0.396 | 0.604 | 0.312 | 240 |
| Two-axis, 8 drops | 0.708 | 0.233 | 0.592 | 0.292 | 240 |
| Two-axis, 16 drops | 0.929 | 0.275 | 0.562 | 0.292 | 240 |
| Two-axis, 24 drops | 0.888 | 0.329 | 0.342 | 0.292 | 240 |
| Powered only, 8 drops | 0.725 | 0.044 | 0.112 | 0 | 160 |
| Powered only, 16 drops | 0.875 | 0.156 | 0.144 | 0 | 160 |

Rebound means are per drop. The lateral and two-axis experiments used twelve independent learner cohorts. The powered-only follow-up used eight fresh cohorts. Each cohort held out twenty launches and used identical practice schedules and exploration RNG initialization across the source treatments. At the final budget, the paired mean Pong advantage over cold was 1.33 rebounds in the lateral stage, 0.56 with two axes, and 0.72 in the powered-only stage. The corresponding 95% cohort-bootstrap intervals were [0.88, 1.74], [0.26, 0.85], and [0.18, 1.29]. [Learning curves](planar-learning.png) · [Paired statistics](planar-summary.json)

## Physics and causal learning

The simulator integrates gravity and solves continuous-time crossing of the ball and the moving paddle plane within each physics tick. A verified rebound requires relative approach, separation, and at least 5.5 cm of new rise at the subsequent apex. Positions, velocities, contact information, and judge events are kept separate from agent observations. The learner receives fixed-view RGB and the scalar on a verified rebound. The rendered paddle is elongated and the ball is compact. Motor outputs are normalized, independent rate channels. [Simulator](../../scripts/experimental/tt24_planar.py) · [Visual learner](../../scripts/experimental/tt24_planar_learning.py)

The lateral stage has one motor and restitution 0.78–0.87. The two-axis stage adds vertical motion and restitution 0.52–0.65. Its 24-drop native-source policy averaged 1.17 rebounds when the vertical channel was pinned, so lateral interception accounted for much of that stage's gain. The powered-only stage lowers restitution to 0.12–0.20 while keeping the same 5.5 cm score threshold. From the highest allowed launch, the maximum possible rise off a fixed paddle is under 4.4 cm. Pinning the vertical channel in the trained policy yielded zero in all 160 paired drops. All 140 verified native-source rebounds had positive paddle speed above 0.05 at contact. On held launch `2121443467`, four rises reproduced in a command replay, and the vertical-pinned control reproduced zero. [Paired powered video](planar-powered-witness.mp4) · [Recorded action replay](planar-replay.json)

The model learns two online regressions from complete practice episodes. One maps issued rates and recent actor image motion to observed actor displacement. The other maps recent observed ball flows to subsequent ball flow. The frozen Pong readout ranks short visual hypotheses made with these learned models. It starts offering guidance only after observing target camera sequences and issued action effects. The critic and direct motor readout update from actual scalar task reward through local eligibility traces. The cold and shuffled conditions use the same apparatus and budgets. Cold uses a constant source prediction, and the comparison source has its native duration labels shuffled within the original strata. [Optical effects](../../scripts/experimental/tt24_planar_effects.py) · [Experiment runner](../../scripts/experimental/tt24_planar_trial.py)

Reloading the 640 final native-source held cases reproduced their per-drop rebound counts. They scored 0.34, 0.39 and 0.19 after replacing the native readout with the constant cold value at evaluation, with the learned target policy and optical effects fixed. The short-budget skill still relies on the source readout during control. [Source-removal control](planar-replay.json)

A separate ten-cohort test used the same readout as a temporal potential applied to the executed optical sequence. Its 16-drop native-source mean was 0.106, with 0.206 cold, 0.262 shuffled, and 0.262 stationary. Its native-source mean recovered to 0.438 at 24 drops. The action-conditioned qualification above used a fresh schedule. [Temporal-potential trial](planar-lateral.json.gz)

## Return to the SO-101 wrist

The bridge trained another visual consequence readout on 192 completed planar drops with translated, mirrored, and contrast-varied RGB. Its label was a verified rebound within 0.65 s of the optical window. On 72 independent visually varied drops, its AUC was 0.876 across 734 valid windows, compared with 0.839 for shuffled labels and 0.500 for a constant cold score. The shuffled labels preserve the coarse stage, frame time and optical-range marginals. [Planar-earned readout](planar-value.json) · [Optical source corpus](planar-value-data.json.gz)

The new readout was then tested on the full five-joint simulated SO-101 with twelve wrist-only practice drops per treatment and twelve paired held launches. The wrist learner fitted action-conditioned apparent motion for both the paddle and ball. The planar-credit controller averaged **2.08 rebounds**, against 1.17 with cold credit, 1.00 with shuffled credit, and 2.17 stationary. Its vision-ablated variant averaged 2.42. The transfer improved the learned comparison arms while the stationary and vision-ablated controls set higher scores. [Native wrist trial](planar-wrist.json.gz)
