# From one aerial delivery to repeated wrist bouncing

We trained new zero-initialized 5 × 20 feedback heads on the three physical hinged donors used in the earlier body experiment. The same five joint channels, MuJoCo contact model, actuator limits and paired motor-probe search basis apply. The probes issue short independent motor rates before ball contact. The source learner receives clean simulated ball position and velocity, paddle position and normal, and its own issued commands. Its policy receives no demonstrations.

The source task feeds one flying ball laterally toward the paddle from the left or right. Initial lateral offsets are 65–125 mm, heights are 230–390 mm, and vertical launch speeds range from −0.58 to +0.24 m/s. The simulator chooses incoming horizontal velocities from a near-center ballistic interception with randomized aim and speed error. The calculation sets the release only. The goal is to reverse that lateral direction and send the first legal rising departure through an elevated receiving region on the arrival side, centered 80 mm off the starting paddle center and 165 mm above it. The region has a 55 mm lateral half-width and a 70 mm transverse half-width. The crossing must travel back toward the entry side at least 0.045 m/s. A powered delivery has contact-point upward paddle speed over 0.055 m/s. Rewards use gate acceptance and powered delivery plus bounded shaping for flight height, direction, aim and lift.

The learner issues no new action after the first paddle encounter. The referee holds the current servo targets to judge the first upward flight, stopping at the gate plane, its apex, or a fault. A recontact during this flight immediately fails the delivery before another decision or judged rebound. Thus the learner experiences no recurrent bouncing trajectory. Twenty-five mirrored perturbation candidates each score two source releases per body per update. Thirty separate source-development feeds select the head, and 72 independent source feeds compare it with stationary. [First-flight simulator](../../scripts/experimental/tt24_volley_source_native.cpp) · [Runner](../../scripts/experimental/tt24_volley_source.py)

A second source profile was specified before any stock outcomes were read. It adds incoming feeds that begin ascending, and targets a lower receiving region at 118 mm above the starting blade, centered 57 mm to the arrival side. This narrower gate accepts upward crossing speeds from 0.16 to 1.18 m/s. It distinguishes a controlled delivery from an unrestricted hard launch. It preserves the first-encounter decision cutoff and uses independent donor releases. This variant achieved only six powered routes in 72 donor held tests and did not proceed to stock. [Bounded-speed source audit](tt24_volley_source-catchable.json.gz)

The final source profile is an exploratory change after the first transfer result. Donor-only diagnostics found 33 lower-gate crossings with acceptable route geometry if either receiver was allowed, versus 20 when the gate had to be on the entry side. This calculation precedes the crossing-speed check. The new profile therefore accepts either left or right receiving gate and raises the maximum upward crossing speed to 1.45 m/s. A new head starts at zero, chooses weights using only its own donor validation and goes to a fresh stock roster. [Two-receiver training and audit](tt24_volley_source-two-receivers.json.gz)

The stock assessment freezes all motor and observation parameters. On matched stock releases it compares the single-delivery head, the separate repeated-bounce donor head, and stationary. The live public conditions use an existing stock sensor calibration from eight public stereo and encoder probes during ordinary bouncing. This calibration is separate from the source motor learner, which never trains on recurrent bouncing. The clean diagnostic conditions use the same corresponding motor heads with privileged physical state. The target is the ordinary repeated-bounce referee with its five-second limit, and no stock motor-practice drops are used. The comparison tests transfer between two reward tasks and two body distributions. Neither source is Pong. [Repeated-bounce head and calibration](tt24_donor.md)

## Locked first-flight source and transfer

The elevated delivery head made 64 powered deliveries in 72 independent source tests, with 23, 20 and 21 on the three bodies. Stationary earned zero deliveries despite 71 legal first departures. The source used 15,000 reward trials, 212,832 observed control decisions and another 53,126 referee-only physical frames. Donor validation added 9,030 trials and 122,385 observed decisions. The one-process training ran for 53 seconds. [Exposure and first-flight evidence](tt24_volley_source.json.gz)

| Frozen head and stock observation | Mean consecutive rebounds, 16 drops | Maximum | At least 10 |
| --- | ---: | ---: | ---: |
| Single delivery, public wrist | 0.938 | 1 | 0/16 |
| Single delivery, clean diagnostic | 0.875 | 1 | 0/16 |
| Repeated bounce, public wrist | 10.875 | 18 | 11/16 |
| Repeated bounce, clean diagnostic | 12.688 | 18 | 13/16 |
| Stationary | 2.375 | 4 | 0/16 |

The delivery head produced higher first launches on stock. With public wrist its median first rise was 334 mm versus 200 mm for the repeated-bounce head. Eleven of sixteen public delivery trials ended outside the permitted bounce region, and the remainder hit the table. Its clean-state branch also never achieved a second consecutive rebound. The reward-task failure persisted with clean sensing. All 80 stock rate tapes reproduced the physical referee outcomes. [Matched stock action and rebound evidence](tt24_volley_source.json.gz)

## Two receiving regions

The modified game permits either lateral gate while keeping the direction and speed-band requirements. On fresh donor feeds the learned head made 45 of 72 geometric deliveries, including 40 powered deliveries. Stationary made 26 geometric deliveries and none with the required rising contact point. The head used 12,000 source trials with 200,064 learner decisions, and 7,230 validation trials with 126,807 learner decisions. The separate mirrored bounded-speed profile used 15,000 training trials and 251,436 decisions, with 9,030 validation trials and 155,616 decisions.

| Frozen head and stock observation | Mean consecutive rebounds, 16 new drops | Maximum | At least 10 |
| --- | ---: | ---: | ---: |
| Two receivers, public wrist | 1.438 | 3 | 0/16 |
| Two receivers, clean diagnostic | 1.125 | 2 | 0/16 |
| Repeated bounce, public wrist | 12.250 | 18 | 12/16 |
| Repeated bounce, clean diagnostic | 14.813 | 19 | 15/16 |
| Stationary | 2.500 | 3 | 0/16 |

The two-receiver head reached a second rebound on six public-wrist drops and the stationary arm did so on all 16. Across the three source experiments, the learner received no feedback decisions after its first paddle encounter. Three invalid recontacts in bounded mirrored training ended in referee-only evaluation. Both stock assessments reproduced all 160 full command tapes. The final simulator also reproduced the 432 independent donor assessment score records exactly. Source training exposures differ in duration across the delivery and repeated-bounce tasks. [Two-receiver tapes and exposures](tt24_volley_source-two-receivers.json.gz)

## Reproduce

```sh
MUJOCO_GL=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt24_volley_source \
  --output /tmp/tt24_volley_source.json.gz --checkpoint /tmp/tt24_volley_source.npz
MUJOCO_GL=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt24_volley_source --catchable \
  --output /tmp/tt24_volley_source-catchable.json.gz --checkpoint /tmp/tt24_volley_source-catchable.npz
MUJOCO_GL=egl LP_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  python -m scripts.experimental.tt24_volley_source --catchable --two-receivers --rounds 80 \
  --output /tmp/tt24_volley_source-two-receivers.json.gz --checkpoint /tmp/tt24_volley_source-two-receivers.npz
```
