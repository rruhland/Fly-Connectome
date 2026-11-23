# Recovering a useful playing posture from independently rotated joints

The usual stock releases vary the initial joint positions, but wrist pitch is chosen to compensate for shoulder and elbow angle. A new start distribution independently added up to 13°, 9°, 12°, 11° and 13° on the five motor joints. With immediate release above the current paddle the frozen reward-learned wrist controller reached at most one rebound on 24 starts. It had reached ten on 11/12 ordinary starts in the paired course. Returning its motor equilibrium toward the mean of successful donor starting poses raised the wide-pose maximum to 17, with three of 24 starts reaching ten. The same correction reduced ordinary ten-rebound starts to 6/12. [Fitting the source posture](../../scripts/experimental/tt24_posture.py) · [Immediate wide and ordinary starts](posture-frozen.json.gz) · [All-axis correction](posture-all_axes.json.gz) · [Learned manifold](posture-manifold.json.gz)

For a separate benchmark, an external launcher holds the visible ball over the usual delivery area for a randomly drawn 0.23–0.63 s and releases it regardless of the arm's position. All controllers see the same scene, have equal time and issue all five joint commands from their first frame. A momentary release signal never enters the policy. The prior is the mean starting pose across 24 archived plays on three other donor arms. Weighting by source reward gives successful starts modestly greater influence. The motor policy is the same frozen reward-trained 5 × 20 matrix in every condition.

| Reference for sensory and motor coordinates | Stock mean longest (16 starts) | Stock ≥10 | Changed body mean longest (12 starts) | Changed ≥10 |
| --- | ---: | ---: | ---: | ---: |
| Reset both to current starting configuration | 0.81 | 0 | 0.67 | 0 |
| Move the motor equilibrium only | 4.50 | 3 | 1.25 | 0 |
| Use the reward-weighted donor pose for both | 10.25 | 9 | 5.67 | 3 |
| Use the unweighted donor pose for both | 11.38 | 10 | 4.42 | 2 |

The same reference lets the controller express ball position and its own joint commands consistently as the arm recovers from an unfamiliar pose. The precise reward weighting contributed little on these cohorts. The preparation time, setup of the fixed delivery area, prelearned wrist geometry and prior reward-trained donor motor are all part of this result. All 112 trajectories, including their preparation, replayed exactly. [Continuous policy and fixed launcher](../../scripts/experimental/tt24_ready.py) · [Source pose data](posture-source.npz) · [Initial-pose reference](ready-as_started.json.gz) · [Motor only](ready-motor_only.json.gz) · [Consistent reference](ready-coherent.json.gz) · [Unweighted control](ready-unweighted.json.gz)
