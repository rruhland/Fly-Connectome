# What the arm learns after an unfamiliar incoming shot

[Learning across successive changes](learning-across-changes.png) · [Watch a recovered 12-return rally](compact_motor-live.mp4)

The primary measure is a legal landing on the opponent’s half. A second return requires the opponent to have legally sent the first one back. The regulation table measures 2.74 × 1.525 m. Both ends use physical arms. The near SO-101 receives two fixed RGB camera feeds, its encoders and its own commands. The far arm uses simulator state for its decisions. This pass did not use backpropagation.

## From the opening image to the end of the point

A single controller now selects its opening joint posture from the first four camera images, detects the opponent’s reply from visible ball motion, and updates its joint target as incoming images arrive. After the ball visibly travels away it can receive another reply. The initial six joint postures and the continuation’s seven strokes came from earlier simulator-assisted demonstrations. The continuation uses a fitted physical flight model, calibrated arm geometry, and a learned contact example. It uses task-specific interception structure. The compact direct joint controller is a separate policy. [Full-point runner](../../scripts/experimental/tt27_full_points.py) · [Demonstration and controller details](reply.md)

| Regulation-table launch cohort | Starts | First legal returns | Opponent returned the opening | Second legal returns | Longest point in alternating legal returns |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previously returned original recordings | 7 | 7 | 4 | 2 | 4 |
| Fresh perturbations of practiced recordings | 12 | 8 | 8 | 1 | 4 |
| Original recordings sampled without a success requirement | 32 | 1 | 1 | 0 | 2 |

All near commands and opponent decisions were generated live. Replaying both arms’ motor tapes reproduced the referee events to point termination. The new controller also uses the next optical reversal after a successful second return. Bringing the arm toward a ready posture learned from demonstrations between exchanges gave the same legal counts as holding the current target. The video compares the same measured launch and same live opponent. [Watch the regulation-table point](regulation-point.mp4) · [Original recordings](full-points.json.gz) · [Fresh variations](full-points-nearby.json.gz)

A test opponent shifted its target and the region used to rank its returns 0.57 m farther from the net, matching the difference in court half-length. Both opponents legally returned the same number of opening shots. The shifted opponent produced zero second returns by the learner across all three cohorts. The existing opponent remains available for comparison. The end-aligned opponent provides a physically different distribution of replies for future practice. [Recorded cohort with shifted target](full-points-end.json.gz) · [Fresh variations with shifted target](full-points-end-nearby.json.gz)

## Learning from a few misses

On four previously unpracticed surfaces the learner observed one to two balls before scoring. It estimated vertical rebound and down-court speed retention from stereo trajectories. Four earlier surfaces supplied 128 practice launches and 896 physical attempts covering seven small vertical adjustments to the chosen joint goal. For each scored feed the learner received only its chosen action’s actual landing reward. Each condition lasted for 32 different feeds. The same laws and launches were used for all comparisons. [Practice and paired rewards](surface-transfer.json.gz) · [Sequential policies](context-memory.json.gz)

| Motor strategy | First four new surfaces, 128 feeds | Independent four new surfaces, 192 feeds |
| --- | ---: | ---: |
| Frozen opening posture | 47 | 76 |
| Learn from target rewards alone | 44 | 72 |
| Practice plus local reward updates carried between surfaces | 54 | 77 |
| Same source practice, reset local updates at each change | 56 | 77 |
| Keep the source-trained policy frozen | 56 | 76 |
| Best of seven actions chosen after seeing all outcomes, diagnostic | 73 | 96 |

On the independent course the carried policy landed six of the first 16 scored balls pooled over the four changes. Reset also landed six. Source practice helped relative to learning entirely on the new tables. Retaining recent rewards did not make the next surface consistently easier. Across 128 presentations of unmodified recordings at the first four settings, only six had a legal outcome anywhere among the seven actions. Access to all seven outcomes is used only to measure coverage. [Independent launches and measured bounces](surface-confirmation.json.gz) · [Independent learning curves](context-confirmation.json.gz)

In a further chronological test, a simulated crosswind exerted a lateral force on the incoming ball in a fixed zone around the net. The learner estimated its deflection from stereo before and after that zone and used the preceding three observations to adjust the next motor target. Across six unannounced settings it made 19/90 first returns on new variations of practiced feeds, versus 30/90 frozen. On original-recording presentations it made 3/90 versus 4/90. Effective motor registration does not yet give the receiver an equally effective response to a novel ball path. [Crosswind receipt](crosswind.json.gz)

A separate learner estimated the lateral and vertical direction of its own miss from ball images, a paddle silhouette and the arm’s encoders, then adjusted its next command for visually similar balls. Its eight seven-shot streams around unfamiliar original recordings produced zero legal returns in 56 attempts versus two for the frozen policy. On another eight streams with a lateral velocity change applied to familiar feeds, it produced 11/56 versus 12 frozen. Optical miss direction alone did not resolve the timing and face angle needed for a legal shot. [All chronological commands and optical errors](miss-learning.json.gz)

Continuously steering the wrist toward the latest observed flight produced 22/120 opening returns, versus 39 frozen, over nominal, soft and higher-gravity conditions. A ballistic predictor using a rolling estimate of the table’s bounce produced 37/144 versus 35 with the original material estimate and 49 with the original receiving posture. The geometric re-selection diagnostic used the adaptive player’s preceding material estimate. [Continuous tracking](opening-beginners.json.gz) · [Physical prediction and gravity changes](physics-course.json.gz)

## New actions and new trajectories

The arm practiced 342 distinct perturbed feeds around 19 successful source recordings. On each feed it tried a neighboring successful action, then perturbed joint targets, stopping at the first legal return. The 2,795 attempts gave 92 new successful image–action examples. Thirty-four successes came after the initial attempt on that feed. The policy can interpolate motor goals among visually nearby examples whose joint configurations are close. [Practice outcomes](growing-practice.json.gz) · [Reusable camera-to-joint checkpoint](growing-receiver.npz)

On the existing 136 original-recording roster, the expanded, interpolated policy made eight returns versus four with the original 19 examples. On another 76 wide variations of the source shots it tied at eight. On an independent roster of 114 nearby and 96 unrestricted original recordings it tied again at seven legal returns. A model that also used the 2,703 unsuccessful attempts to estimate which goals were likely to work made six legal returns on that roster. Increasing the number of remembered examples did not establish better generalization. [First cohort](growing-held.json.gz) · [Independent cohort](growing-confirmation.json.gz)

A second practice method measured each original motor action in dry swings, perturbed the joint goal to move the paddle laterally at the demonstrated contact time, then kept only goals that actually returned a ball on a regulation table. It produced 37 successful examples from 445 shot attempts across 114 displaced feeds. On the first 210 independent recordings, interpolating this smaller library landed 14 returns versus seven for the original. On a new disjoint set of 370 original recordings that became 14 versus 12. On the same 370 with changed table damping and 4% greater gravity it became 10 versus nine. The improvement shrank on replication. The curriculum receives the original contact time, commands the feed and exercises the arm without a ball during practice. Deployment receives the first four RGB frames and sends joint targets. [Guided practice](lateral-practice.json.gz) · [Learned motor checkpoint](lateral-memory.npz) · [First test](lateral-held.json.gz) · [New recordings and changed physics](lateral-confirmation.json.gz)

A geometry audit of the second, 370-launch roster found 230 inbound trajectories passing within a ball-plus-paddle radius of a sampled collision-clear paddle center. All 14 successful openings fell in that group. The sample contained 8,192 arm configurations. This screen ignores arrival time and paddle attitude, so it identifies opportunities for investigation without proving all 230 were returnable. It also cannot prove that every one of the other 140 was physically unreachable. [Paired workspace and return audit](workspace-audit.json.gz)

In 70 end-to-end live points on the earlier independent roster (38 nearby, 32 unrestricted), growing the action library by ordinary perturbations and using the original 19 goals each yielded two legal opening returns and zero seconds. Both policies used the visual reply controller whenever a ball came back. [All 140 full games](growing-games.json.gz)

The cameras also measured four lateral clamp movements (4.5 to 13 cm in magnitude) to within 0.7 mm using the ordinary red paddle face at rest. Changing the target joint posture to preserve the predicted final paddle position and normal reduced opening returns to 15/112 nearby feeds versus 26/112 with the uncorrected policy. Correctly locating the base was insufficient to control the moving paddle at contact. Returning the clamp to its original position restored identical outcomes for all gains. [Physical mounting trials](mount-position.json.gz)

## Can a better choice among the learned motions solve a new opening?

On 38 original recordings excluded from the preceding 210- and 370-shot evaluations, we physically executed every one of the 56 learned joint goals on every opening. Of 19 recordings from the neighborhoods of practiced shots, 14 elicited contact with at least one goal and four had a legal option. Of 19 unrestricted recordings, nine elicited contact and three had a legal option. An evaluator with hindsight could return seven of 38 from the fixed library. This ceiling applies to these 56 goals with their fixed start time and motor rate. It places no limit on a controller that changes its motion. [All 2,128 executions](action-coverage.json.gz)

The online comparison started with the goal nearest to the first four camera observations. After a miss it tried a different goal, favoring similar poses when the referee reported contact. It got three of 38 on the first try. Repeating each identical opening for up to five attempts added two successes, on attempts four and five. Each selected action reproduced its events during a separate live replay. The other 55 outcomes on a given shot were available only to the evaluator. [Implementation](../../scripts/experimental/tt27_action_coverage.py)

We then allowed eight attempts per original shot and tested the accepted setting on four previously unseen variations. All branches started with the same camera-selected movement and received the legal-return reward. Three also received the referee’s contact bit. The branch called reward only used the number of previous misses to decide when to vary its movement. The shift memory held 128 small joint differences from past failures followed by successful training attempts. [Each branch’s own actions and independent variations](contact-refine.json.gz) · [Implementation](../../scripts/experimental/tt27_contact_refine.py)

| Exploration strategy | Recordings returned within three tries, of 38 | Within five | Within eight | Legal returns on 152 new variations |
| --- | ---: | ---: | ---: | ---: |
| Search the fixed library after a miss | 3 | 5 | 5 | 11 |
| After contact, try a continuous shift learned from prior corrections | 3 | 3 | 4 | 10 |
| After contact, perturb either of the last two active joints | 3 | 4 | 5 | 8 |
| Try previously learned shifts with legal reward alone | 3 | 3 | 3 | 8 |

The two-joint perturbation returned two original recordings for which every one of the 56 stored goals failed. It needed five and eight attempts respectively. The acquired goals returned only one of the eight subsequent variations of those two shots. Changing the movement can overcome the measured library limit, but carrying these small corrections did not yield dependable ball control. The two exploration axes and their amplitudes were chosen by the experimenter.

## Preparing for a changed trajectory

A camera-selected receiving policy had 14 control ticks before release to move toward or away from the posture it used for the preceding approach. The practice comparison exercised all six preparations. During evaluation the carried learner saw only the reward from its chosen preparation. We checked that the first four images yielded the same choice of receiving action across all six preparations on every feed. With varied table damping and fresh launches near practiced shots, carrying the reward history scored 36/90, level with the neutral ready position. Always moving 20% toward the previous receiving posture scored 14/90. On fresh variations of ten unfamiliar recordings, carrying scored 2/90 versus 1/90 neutral. Changing where the arm waited did little to recover these trajectories. [Paired physical trials and chronological rewards](ready-learning.json.gz)

A timing replay reproduced all motor commands and referee events of the four-return regulation point. With other experiments running on this host, the active reply controller took 5.8 ms per frame at the median and 17.4 ms at the 95th percentile. Generating the camera observations took 35.5 ms per frame at the median. The full 4.77 simulated seconds took 16.79 wall seconds including simulation and opposing-arm planning. This measures simulator throughput on one point. [Per-component timings](control-clock.json)

## Reproduction

The input file is the same hash-checked DeepMind measured-launch dataset used in the regulation benchmark. [Dataset and reference measurements](../tt25/recorded-benchmarks.md) · [Google DeepMind table tennis data](https://github.com/google-deepmind/competitive_robot_table_tennis)

Run from the repository root with the project’s MuJoCo dependencies and set `MUJOCO_GL=egl`, `PYTHONPATH=.:src`, `LP_NUM_THREADS=1` and `OPENBLAS_NUM_THREADS=1`. Commands with a dataset argument use the downloaded measured-launch JSON. The context-memory evaluation reads the preceding surface trial.

```sh
python -m scripts.experimental.tt27_flight_feedback /path/to/rallies.json
python -m scripts.experimental.tt27_opening_course /path/to/rallies.json
python -m scripts.experimental.tt27_physics_player /path/to/rallies.json
python -m scripts.experimental.tt27_surface_transfer --raw /path/to/rallies.json
python -m scripts.experimental.tt27_context_memory
python -m scripts.experimental.tt27_surface_confirm /path/to/rallies.json
python -m scripts.experimental.tt27_mount_position /path/to/rallies.json
python -m scripts.experimental.tt27_miss_learning /path/to/rallies.json
python -m scripts.experimental.tt27_growing_receiver /path/to/rallies.json
python -m scripts.experimental.tt27_outcome_receiver /path/to/rallies.json
python -m scripts.experimental.tt27_growing_games /path/to/rallies.json
python -m scripts.experimental.tt27_lateral_practice /path/to/rallies.json
python -m scripts.experimental.tt27_lateral_confirm /path/to/rallies.json
python -m scripts.experimental.tt27_workspace_audit /path/to/rallies.json
python -m scripts.experimental.tt27_action_coverage /path/to/rallies.json
python -m scripts.experimental.tt27_contact_refine /path/to/rallies.json
python -m scripts.experimental.tt27_crosswind /path/to/rallies.json
python -m scripts.experimental.tt27_ready_learning /path/to/rallies.json
python -m scripts.experimental.tt27_control_clock /path/to/rallies.json
python -m scripts.experimental.tt27_full_points /path/to/rallies.json
python -m scripts.experimental.tt27_full_points --nearby /path/to/rallies.json
python -m scripts.experimental.tt27_full_points --align-to-end /path/to/rallies.json
python -m scripts.experimental.tt27_full_points --align-to-end --nearby /path/to/rallies.json
python -m scripts.experimental.tt27_full_movie /path/to/rallies.json
```
