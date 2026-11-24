# M1B research handoff

## Continuously random physical opponent and joint learning

On 24 previously unseen random openings, expanding the joint pupil's practice from 86 to 289 rewarded arm returns raised legal openings from eight to ten and legal later replies from two to six. The best point on this comparison rose from two to four arm returns. A separate visual readout gave a five-arm-return point, but its total across the course was level with the original. A privileged practice controller reached nineteen legal arm returns, and the independent opponent demonstrated a hundred per side against a second privileged carriage. A sustained hundred on the autonomous SO-101 against broad random placement remains open. [Results and physical replays](../../artifacts/tt37/learning.md) · [Actual online learning from misses](../../artifacts/tt37/online-drive.md)

## What the direct joint learner can correct after missing or hitting

On six previously unsuccessful table-tennis openings, practicing how to reach and then how to strike produced legal returns on three. On two of those, removing either learned movement lost the return. The movie shows the same incoming ball missed, touched, then returned after 19 attempts. With 20 varied first deliveries around the six, legal arm returns rose from two to seven. Practicing reach alone tied that seven. A separate controller using the arm's body map learned from 20 scoreless games and examples of rewarded approaches increased first contacts from 35 to 41 and legal returns from 20 to 26 across 64 unfamiliar openings. Later returns rose from four to seven. The newly reached balls still failed to land. One controller using the current camera and joint observations retained the gains on the practiced and 64 unfamiliar openings. It landed four on the 20 displaced deliveries, versus seven for the practiced corrections alone, and 14 on the eight-continuation retention test versus 16 originally. Next prioritize converting those weak or off-center touches into playable returns while preserving successful earlier movement, and score the second and third returns under changed opponent trajectories. [Watch the identical shot before and after practice](../../artifacts/tt36/learning-12373.mp4) · [Evidence, learning curves and controls](../../artifacts/tt36/learning.md)

## Direct joint control and learning from missed shots

The raw joint pupil now adapts to camera and actuator changes from observations collected without a legal return. After identifying a 100 ms image delay during one missed point, it landed five opening returns across 16 held releases versus one with stale timing. On eight openings repeated through four unannounced delay blocks, online timing estimation increased legal arm returns from 14 to 19, including two additional later returns. At the final zero-delay block all eight scores matched the original block. This used the observed ball position, velocity and joint angles, with the existing direct motor policy frozen. The nominal motor experiments still struggle to improve beyond the 30-demonstration pupil on unfamiliar trajectories. [Research and controls](../../artifacts/tt34/zero-reward.md) · [Watch the matched real-speed point](../../artifacts/tt34/timing-6323.mp4) · [Simultaneous camera and actuator changes](../../artifacts/tt34/mixed.md)

On another 24 release IDs the same camera calibration raised two legal arm returns to eight, including three later replies. With timely cameras, refitting the existing joint readout on 114 observed ticks from five of its own scoreless games raised seven returns to eleven. A control refit on the same number of familiar states scored seven. Most of the difference came from one five-reply point. On eight new opponent continuations from that opening the refit scored 13 versus 16 for the original. A separate reward-trained five-joint adjustment scored nine versus seven on the 24-release course. Retain the sensor adaptation. Next prioritize learning the outgoing effect of contact and verifying later-reply gains across fresh continuations. [Evidence and controls](../../artifacts/tt35/confirmation.md) · [Watch 15 planner-free legal returns](../../artifacts/tt35/varied-5530.mp4)

## Direct joint feedback on unpredictable shots

The SO-101 can play with the stroke planner removed. A 60 Hz joint policy fitted without backpropagation to 30 rewarded teacher strikes landed six own returns across eight fresh points, including two consecutive returns on two openings. A student using current optical ball position and velocity and the arm's joint angles landed three. On the same eight, the stationary arm scored zero and the widened stroke planner scored six. The main teacher-free policy used 25 CPU seconds for these points versus 161 for the widened planner. On an earlier eight the same direct policy landed two versus the teacher’s eight. With a second set of opponent seeds it landed three alternating-side arm returns on one opening and missed the other seven. Replaying the original commands blind landed zero. The generalized local feedback policy without an explicit incoming gate or return-to-home instruction also returned twice in one point. Its extra reward-only practice did not raise held performance. Next, learn joint-action consequences from successful and failed transitions, and use the planner sparingly during training for corrective examples on the learner’s own difficult states. Keep planner-free matched and changed-opponent tests throughout. [Watch the seven-return direct policy](../../artifacts/tt33/direct-seven.mp4) · [Feedback controls and limitations](../../artifacts/tt33/feedback.md) · [Why the prior arm stayed still](../../artifacts/tt32/preparation.md)

## Continuous placement and learning to switch sides

The table-clamped SO-101 returned seven successive balls, switching sides six times, in a 15-return simulated rally. The arm used the same paddle face on all seven. On four fresh alternating points its own legal returns totaled 2 with the earlier memories, 2 after 16 lessons, 4 with the earlier memories and a new post-contact recovery rule, and 11 when practice and recovery were combined. Two of the four openings still yielded zero. On eight points with independent continuous placements, recovery raised four legal arm returns to six with either memory. Regular alternation benefited from practice. The unpredictable game still showed no added score from the lessons. The earlier fixed posture with the new memories retained the familiar 24-return capped point. Next, train recovery from the outcome of the following shot on repeated-side, crossover and switching trajectories. Preserve the fixed-recovery and original cooperative controls. [Watch the 15-return point](../../artifacts/tt31/alternating-15.mp4) · [Watch an unpredictable point](../../artifacts/tt31/varied-game.mp4) · [Shots, learning controls and paddle-flip tests](../../artifacts/tt31/shot-variety.md)

## Learning to correct wider table-tennis returns

On eight unseen measured openings, reusing 24 camera-measured shot errors and making a late camera-guided motor adjustment raised the SO-101’s own legal returns from 49 to 58. On a different opponent schedule using those same openings it went from 48 to 51. Across the 16 paired points the combined controller lengthened eight and shortened five. On the first schedule, reversing the learned error signs reduced its total from 116 to 84 alternating returns. Lifting the 24-return stop on a point that originally lasted six produced a verified 30-return rally. The combined policy still uses stereo RGB, encoders, a nominal dynamics model, and an explicit stroke generator. The original controller remains the cooperative default. Adding the wide error memory to the 200-return cooperative point reduced it to 114. [Watch the wider rally](../../artifacts/tt30/extended-5736.mp4) · [Paired results and exact-state checks](../../artifacts/tt30/shot-correction.md) · [Signed outcome learning](../../artifacts/tt29/optical-outcomes.md)

On another schedule the original outgoing-flight measurement captured 21 of 23 legal arm returns and zero of three contacted misses. Using a short sideways or downward deflection as evidence of insufficient forward travel raised the development course from 21 to 35 arm returns. On a previously unused opponent schedule the same rule learned from one failed stroke during play and raised legal arm returns from 29 to 31. The benefit varied by point. A separate exact-state diagnostic showed a physically attainable legal return for five of six terminal misses, including two balls that passed the learner’s paddle. The next useful target is selecting a reachable correction when there is little or no visible outgoing flight and applying it without degrading already successful returns. [Sensor coverage](../../artifacts/tt30/label-coverage.json) · [Physical feasibility](../../artifacts/tt30/reach-check.json.gz)

## Table-edge SO-101 and continuous regulation rallies

The table-clamped SO-101 and a physical practice carriage exchanged **200 consecutive legal returns** in 138.02 simulated seconds. Each landed 100. With the same opening, opponent seed and fresh optical tracker, an empty movement library scored 55. The warm arm started with 173 rewarded movements and learned another 100 during play. The full replay reproduced 1,006 events, sampled 66,249 CAD states without penetrations, and stayed inside the signed motor limits over 331,240 joint samples. Three measured openings separately reached 100 with stereo RGB and encoders. The random forehand feed suited the fixed arm’s reach. Wider placement, wrist-only perception and simultaneous physics and latency changes remain open. Solo bouncing transferred an observed command delay to table play, and the retained solo skill reproduced its held performance after the table route. [Watch the complete 200-return point](../../artifacts/tt29/coupled-camera200.mp4) · [Evidence, controls and reproduction](../../artifacts/tt29/coupled-endurance.md) · [Camera and clamp views](../../artifacts/tt29/fixture.md) · [Warm-up and adaptation](../../artifacts/tt29/adaptation.md)

## How long to pretrain on solo bouncing

On two hidden paddle mounts the same wrist controller accrued 2,071 physical rebounds over two streams of 192 drops, with visual tool-angle and reward updates after every release. On 24 paired held drops the total rebound counts at 0, 4, 16, 64, and 192 practice drops were 21, 216, 136, 126, and 175. After zero, four, and 192 drops it made 29, 30, and 30 legal opening returns on 96 fresh variations of demonstrated feeds. On 192 unrestricted measured launches across both mounts it made 1, 1, and 1. Across 54 full points it returned 15, 20, and 20 openings at 0, 16, and 192 drops respectively. The 16- and 192-drop players returned two and one of their 16 opposing replies. A source-only rule judged the visible grip estimate stable after seven drops on each tool. Use these checkpoints for comparisons. This experiment transferred grip calibration between existing task policies. [Full result and fixed-budget comparison](../../artifacts/tt28/transfer.md) · [Dose curves](../../artifacts/tt28/bounce-practice.png) · [Matched wrist replay](../../artifacts/tt28/wrist-bouncing.mp4)

On two disjoint courses the motor directions learned from increasingly fast solo tosses achieved ten and 14 held opening returns per 152 feeds after eight regulation practice shots per donor, versus four and 14 frozen. Early camera measurements of outgoing flight gave a small gain after three probes on a contact-supported course, reaching 47/368 versus 42/368 using only reward. Both branches reached 50 after five probes. Combining 16 solo drops and the outgoing method on a fresh, contact-supported comparison scored 17/144, level with the outgoing method alone. [Outgoing feedback](../../artifacts/tt28/outgoing.md) · [Combined trial](../../artifacts/tt28/combined.md)

The next target is online learning on laterally arriving balls and opposing replies, including converting paddle contact into a playable return and recovering in time for the next ball. Keep a brief solo calibration, early versus late checkpoint controls, and unrestricted regulation starts. More repeated vertical drops have not reliably improved that transfer.

## Regulation receiving after changes to the ball, table and motor interface

In a matched active-probe test, the arm inferred three earlier relative changes in motor wiring from encoder responses. It identified ten new combinations of those changes in 1.27 simulated seconds each versus 1.93 seconds with its change memory reset. Two surprise combinations took 1.93 seconds. Both learners identified all 12 configurations and made 12/36 legal opening returns. A deliberately altered untested channel fooled the short inspection on a separate test. Encoder feedback flagged the mismatch six ticks into play and the longer inspection corrected the map before the next feed. [Successive changes](../../artifacts/tt27/motor-recurrence.md) · [Learning-speed figure](../../artifacts/tt27/learning-across-changes.png)

The best demonstrated rapid adaptation is in the control interface. With the ball held, the SO-101 compared five small pairs of motor commands to encoder responses learned on the original arm. With zero intermediate resets it identified all eight unfamiliar signed permutations and zero- or one-tick actuator delays in 1.93 simulated seconds per setting. It landed 10/40 first returns on laterally shifted and speed-varied regulation feeds, matching the known-map arm. Stale wiring landed zero. An equally active reset learner did equally well. Transferring an independently estimated zero-delay mapping from regulation to compact play restored a 12-return rally on a fresh release, six per arm, versus zero with stale wiring. An identified actuator delay still hurt play. [Motor experiments](../../artifacts/tt27/motor-active.md) · [Compact transfer](../../artifacts/tt27/compact_motor.md) · [Watch the matched 12-return point](../../artifacts/tt27/compact_motor-live.mp4)

The direct compact controller completed two 14-return points out of 20 starts as placement, contact damping, and camera delivery changed. A fixed motor policy also achieved both. A one-frame camera update restored sustained play after camera rotations. Successful registration transfers more consistently than rally strategy. [Watch a 14-return point](../../artifacts/tt27/compact-live.mp4) · [Rotated views](../../artifacts/tt27/compact_camera.md) · [Reset, carried and frozen controls](../../artifacts/tt27/compact.md)

On regulation, the camera policy controlled the arm from the opening image to point termination. It landed two second returns on four original recordings that elicited live opposing replies. On 32 sampled original recordings without a first-return filter it landed one opening return and zero seconds. An evaluator comparing simulated candidate strokes landed three second returns on the four reply opportunities. It is a planning comparison. [Watch one matched regulation point](../../artifacts/tt27/regulation-point.mp4) · [Full regulation points and trajectory studies](../../artifacts/tt27/trajectory-learning.md) · [Second-return controls](../../artifacts/tt27/reply.md)

On eight fresh compact releases with one tick of actuator delay, advancing the learned visual prediction made one first return and zero seconds. Linear pixel extrapolation made two first returns and the intact hardware timing made eleven total near returns, including a seven-per-arm rally. A diagnostic ten-return recovery did not transfer. [Timing readout, all replays and motion measurements](../../artifacts/tt27/delay_readout.md)

On another 38 original measured openings, 23 admitted physical contact and only seven admitted a legal return after executing every one of the 56 stored goals. A camera-initialized learner found five of those seven within five attempts. Perturbing two joints after a contact turned two openings that defeated all 56 goals into legal returns on attempts five and eight. Only one of eight slight variations of those two openings was then returned. Prior learned motor shifts scored below discrete exploration overall. [Execution receipts and few-mistake comparison](../../artifacts/tt27/trajectory-learning.md#can-a-better-choice-among-the-learned-motions-solve-a-new-opening)

The next research target is a more versatile continuously adjustable camera-to-joint controller that learns from contact timing and outgoing ball motion over ordinary play. Score first, second and third returns on disjoint original recordings, each learner’s first few mistakes at a new change, and transfer to the following change against equally active reset. Keep the proven compact and body-identification controls for retention.

## Solo bouncing as preparation for full-size receiving

Start with the [bounce-to-table report](../../artifacts/tt26/bounce-to-table.md), [paired video](../../artifacts/tt26/bounce-to-table.mp4), and [learning curves](../../artifacts/tt26/learning-speed.png). On two concealed grip changes the simulated SO-101 selected wrist roll from all five joints by rebound reward. Updating its estimate of paddle attitude raised the median fresh wrist-camera streak to 24 versus one cold. In eight matched online receiving streams it reached five legal returns in a median eight shots versus 18 cold. On four new grip angles, carrying just the learned order for exploring joints reached the final 50% beginner-feed return rate after 24 probes versus 88 for uniform exploration.

Both tasks retain their existing visual heads and the receiving arm has six previously demonstrated motor options. Against a live opponent it legally returned 28/48 fresh beginner feeds versus 12/48 cold. The opponent returned all 28 and the learner missed the next ball on every point. On 144 independent nearby recorded feeds under both grips it improved to 23/288 from 7/288, with zero on the unrestricted 32 under either grip. Keep second returns, broad recorded launches and source retention in the next assessment. Target-calibration gains and full-match competence need separate scores.

The [visual impact experiment](../../artifacts/tt26/visible-impacts.md) gives an independent transfer mechanism. On three changed tools, 16 solo drops supplied an optical face estimate within 0.8° of the physical setting. Updating both motor targets and perception reached at least ten rebounds on 20/24 new wrist-only drops on the two handle twists. An independent unpaired source test using one new drop yielded 97/192 beginner receiving returns versus 62/192 with zero new drops. For the third, canted mounting, solo rewards selected a more flexible blade-position target and produced a median of 18 consecutive rebounds on 12 fresh starts. Imposing the same tolerance on table tennis reduced receiving performance. Separate reply practice produced one held second legal return in 14 attempts, on a new variant of a practiced donor. [Reply-level results](../../artifacts/tt26/match-continuation.md)

Broadening the camera receiver from six to 19 postures with 13 additional verified demonstrations raised legal returns from 1/136 to 4/136 on nominal and from 1/136 to 5/136 with a positive handle twist and optical compensation. Both expanded policies returned two of 64 unrestricted physical recordings. With the expanded library fixed on the positive twist, optical bounce calibration scored 5/136 versus 1/136 without it. Future practice should target recovery and broaden the action repertoire, while varying the received trajectories and retaining per-grip cold controls. [Training cost, unchanged-motion control and expanded checkpoint](../../artifacts/tt26/match-broader.md)

## Realistic receiving and the next ball

Use the collision-enabled, regulation-size SO-101 receiving benchmark for natural-play claims. A new flight profile fitted on five opponents' measured matches cut median bounce-location error from 38.5 to 16.2 cm on 417 held recorded shots. Restoring moving-link/table contact removed gripper penetrations beyond 5 mm on the 16 audited compact points. The stock direct pupil still returned 25 shots on 24 compact starts and produced one 12-return rally there. [Flight physics, arm clearance and scores](../../artifacts/tt25/recorded-benchmarks.md) · [Watch the compact rally](../../artifacts/tt25/recorded-arm-stock.mp4)

On the full-size table, six verified simulator-assisted demonstrations taught a small camera-selected receiving repertoire. The pupil made seven legal blocks on 96 distinct recorded launches in the demonstrations' neighborhoods and zero on a separate unrestricted 32. A reward-only stream of 192 progressively varied practice balls added one net return on a further 48 neighboring shots (6 versus 5). Both policies scored zero on another unrestricted 32. [Methods, controls and checkpoint](../../artifacts/tt25/recorded-receiving-lessons.md)

Letting the opponent play out all seven unseen blocks produced four legal replies. The SO-101 missed the second ball every time. Training a camera-to-joint-rate continuation on two successful moving-contact demonstrations produced two own returns and two far replies on each practiced feed. It preserved the first return on all seven unfamiliar feeds but did not reproduce a second. It also missed every second return on 16 perturbed versions of its two practiced feeds. A Pong-trained camera delay gain did not improve recorded-shot learning speed, and the time gain fitted in these table-tennis games reduced the aggregate wrist-bounce streak count versus leaving compensation off. [Live play and transfer controls](../../artifacts/tt25/full-size-rally.md) · [Practiced and unseen points on video](../../artifacts/tt25/recorded-two-arm-points.mp4)

The next target is to collect opposing replies from many different successful first returns, record synchronized images, actual joint motion and commands, and train a continuously updated response from approach through recovery. Measure second legal landings on new shots with a live opponent and keep the unrestricted cohort in every assessment. Proceed beyond receiving blocks before interpreting a 10–20 return run as natural full-size match play.

## Recommended online motor learner

A single live stream of Pong commands and encoders updates the drive estimate at each point break. A rolling 0.5 MB numeric history reproduced both independently qualified full-map fits after twelve points, then continued past the window limit. On 32 stock wrist drops across two fresh cohorts those maps reached ten consecutive rebounds on 14/32 versus 1/32 with the stale mapping. Moving the electronics to the altered body reduced ten-rebound starts to 2/12. [Watch Pong and the first wrist release](../../artifacts/tt24/pong-to-wrist.mp4) · [Load the continuing estimator and check timing](../../artifacts/tt24/rolling-motor.md)

On the changed body, refitting after six ordinary wrist games reduced held mean longest streak from 2.875 frozen to 1.625. A prediction check recovered 2.25. For competitive table tennis the original Pong estimates yielded no legal near returns on sixteen starts. Added 1.45-second inspections yielded eleven complete exchanges versus 21 with the true maps. [Continuing on the new body](../../artifacts/tt24/rolling-motor-wrist-report.md) · [Competitive transfer](../../artifacts/tt24/rolling-match.md)

## Body learning during native Pong transfers on the first bounce games

The first half-second of ordinary physical joystick motion by a reward-trained native Pong player identified one changed actuator direction. Without any target practice, the carried wrist policy reached ten consecutive rebounds on 9/12 fresh drops, versus 5/12 with the original map. Two seconds produced the same count, or 0/12 with shuffled command and encoder timing. An earlier paired experiment found that two interfaces indistinguishable in the source game behaved very differently in five-axis wrist play. The unobserved dense change gave zero ten-rebound runs across ten starts. [Source play, controls and transfer](../../artifacts/tt24/pong-arm-report.md)

A joystick operated by the physical arm from varied postures exposed all five directions. A reward-trained Pong player learned two new coupled interfaces from 59 and 68 seconds of ordinary source play. Its wrist controller reached ten bounces immediately on 8/16 stock drops versus 0/16 stale and 10/16 with the true maps. With the same electronics on the physically altered body, these counts were 2/12, 0/12 and 6/12. [Source histories, runnable learners, both physical bodies and retention](../../artifacts/tt24/learned-tip-transfer.md)

A 1.60 s targeted inspection on the altered arm improved measured map accuracy on 10/12 starts but reduced qualifying rebounds from 90 to 75. The body-model error explains only part of the remaining game loss. [Same-pose controls](../../artifacts/tt24/tip-two-probes.md)

## Two-minute wrist qualification

A frozen direct wrist controller sustained the whole 120 seconds on five of six stock releases under 18–19 hidden gravity changes apiece. The strongest made 428 consecutive rebounds. The altered arm had a best survival of 100.95 seconds and a best uninterrupted streak of 53 across six drops. Four altered starts ended within five seconds. This course used familiar motor wiring, prior optical calibration, and a visually adjusted workspace. [Watch all 428](../../artifacts/tt24/long-wrist-428.mp4) · [All twelve results and passive controls](../../artifacts/tt24/long-wrist.md)

A separate wrist-only tool-angle correction raised 15-second completions to 7/12 from 4/12 on the changed arm and produced its first full two-minute release, with 407 qualified rebounds. It reduced the aggregate across the six long releases and reduced returns on a mixed motor-interface course. The original observer is still the default. [Short and long cohorts](../../artifacts/tt24/wrist-rim.md) · [Watch the complete changed-arm release](../../artifacts/tt24/wrist-rim-407.mp4)

## Combined single-arm checkpoint

A persistent wrist learner inspected the actual receiving arm and played without resetting it. Across 36 sequential compound-physics releases it scored 456 qualified rebounds versus 278 frozen and 537 with true actuator mappings. Stock reached ten on 10/15, altered on 7/21 versus 0/21 frozen. All 11,381 commands reproduced after reloading the model at each game boundary. That version acted on encoder surprise at the next release. Accepting the same request while the current ball was still held raised the total to 529. On the three invisible changes it earned 35 versus four from the stale map after exactly the same held prefix. Its restored-stock performance dropped to 141 from 162 across nine releases. [Inspections during the current release](../../artifacts/tt24/current-pause.md). [Load the checkpoint and see every paired result](../../artifacts/tt24/continual-wrist.md) · [Sharing motor knowledge between bodies](../../artifacts/tt24/motor-recombine.md)

## Body knowledge transferred immediately into a real two-arm game

A motor interface learned in one 1.45-second movement test carried straight into table tennis on 16 fresh receiving points with four unknown mappings. The arm made 12 legal returns and 10 completed two-arm exchanges, versus one return and zero exchanges with its original commands. The true inverse gave 11 and 11. The longest point was a win with eight own returns and seven opponent returns. The existing match head had prior demonstrations and used two cameras. All 48 points replayed exactly. [Watch the same point with and without the learned mapping](../../artifacts/tt24/settled-match-witness.mp4) · [Transfer and retention](../../artifacts/tt24/settled-match.md)

With the original wrist hinge mechanically jammed on a six-hinge body, the carried reward-trained pupil reached 20 solo rebounds on five of eight starts, versus one when the extra motor was sent zero and none when both hinges were locked. The benchmark supplied the correct previously calibrated public map. On a separate course with no topology change, encoder-triggered settling learned eight new five-channel mappings to within 0.4% in 1.37–2.60 seconds. [Six-hinge topology](../../artifacts/tt24/carried-six.md) · [Efficient identification](../../artifacts/tt24/motor-sampling.md)

Across 64 releases with unannounced changes, the arm's own tracking-error monitor earned 594 rebounds versus 376 frozen and 938 when the current inverse was supplied. A native-Pong threshold for that monitor transferred poorly. On delayed camera packets, retaining future-motion observations from earlier games produced 94 summed consecutive rebounds on 18 fresh releases, versus 54 clearing memory each game, 49 with the frozen forecast and 229 with prompt input. [Unannounced motor changes](../../artifacts/tt24/autonomous-drive.md) · [Online visual adaptation](../../artifacts/tt24/online-temporal-memory.md)

On a fresh compound-physics course, shifting the familiar horizontal workspace from three wrist sightings, while preserving the familiar paddle height and tilt for the current arm, produced ten consecutive rebounds on eight of twelve altered-arm starts, versus three with the original reference and four with a constant shift. Full 15-second runs still occurred on only four of twelve. Extra training across held releases improved donors and failed to improve the changed receiving arm. [Visually adjusted workspace](../../artifacts/tt24/workspace-stress.md) · [Donor-training test](../../artifacts/tt24/held-donor-report.md)

## Rapid motor-interface learning

Eight new dense five-channel mappings were learned from one 1.45-second unscored encoder inspection each. On the first four drops per mapping, stock earned 258 rebounds versus 14 uncorrected and 251 with the true inverse. The physically altered arm earned 125 versus 17 and 165. Previous mappings were recognized in 0.70 seconds. This test provided a setup opportunity before play. [Results, controls, learned interfaces and inference code](../../artifacts/tt24/motor-settle.md)

## Native Pong experience under visual delay

Four new native-Pong fits were assessed on 24 previously unseen arm starts. Using their learned visual lookahead produced a summed longest streak of 62 against 55 without extrapolation and 56 for shuffled source reward. Fixed gains of one and two produced 70 and 68. The initial 12-start gain over cold was real, but the independent test does not support preferring this learned adjustment over the simpler fixed settings. On prompt input the complete native and wrist commands matched their original tapes. [Source and both arm studies](../../artifacts/tt24/pong-temporal.md)

## The first catch is the limiting step on the changed arm

In a new mixed-body course, one frozen wrist policy sustained 40–56 consecutive rebounds over the full 15 seconds on all eight stock starts. A common stock launcher, a new mass and drag draw per release, independent starting arm angles and two hidden gravity changes were combined. Only two of eight altered-arm starts lasted the full interval, with streaks of 12 and 28. All six other altered runs failed before the first gravity change. All 48 physical replays matched. [Stress test](../../artifacts/tt24/integrated-stress.md)

The packaged motor responds causally to unexpected ball movement. On paired interventions, it made 105 uninterrupted post-kick rebounds against 23 replaying the commands from its original undisturbed games. Two generic altered-arm movements totaling 3.2 seconds yielded a wrist-only camera and link calibration. It reached ten rebounds on 7/16 altered-arm starts versus 3/16 with stock optics, and recovered the complete original stock commands on all four revisits. [Flight intervention](../../artifacts/tt24/carried-perturb.md) · [Wrist-only recalibration](../../artifacts/tt24/wrist-source-only.md)

A source-learned causal correction reduced lateral speed on all 12 comparable wide first rebounds, but only two of 16 complete starts improved. Broad donor training regained some second rebounds at the expense of ordinary-start performance. Online scalar reference learning also improved mobile-station revisits but lost much of the advantage when the fixed station returned. [Event-based correction](../../artifacts/tt24/event-align.md) · [Donor posture generalization](../../artifacts/tt24/tt24_tilted_start_context_report.md) · [Online station learning](../../artifacts/tt24/station-switch.md) · [Motion-gain test](../../artifacts/tt24/tt24_motion_ready.md)

## Replicated native Pong transfer to five-joint wrist control

The [portable wrist policy](../../artifacts/tt24/carried-wrist.md) reproduced 8,011 live decisions across 32 drops running at a median 1.43 ms per frame.

Pong reward taught a single numerical preference for reusing a familiar reference position when beginning in an unusual pose. The same scalar controls the visual origin and motor equilibrium on the arm. Twelve fresh native fits selected full reuse; a within-serve shuffle did so on five. On 72 fresh physical stock starts, the native preference reached ten rebounds on 51, versus 22 for shuffled credit. On 48 changed-arm starts it reached ten on eleven, versus four. The native and fixed full-reuse controls were identical. The five-joint motor skill came from three other hinged arms, and the wrist sensor had prior optical calibration. [Paired video](../../artifacts/tt24/reference-transfer-paired.mp4) · [Source and both physical replications](../../artifacts/tt24/reference-transfer.md)

In another 32-start course, recalling the altered body’s calibrated wrist model from its first image while using a consistent motor and visual reference gave ten consecutive rebounds on 10/16 changed starts and 8/16 stock starts. [Mixed-arm recall](../../artifacts/tt24/ready.md#switching-physical-arms)

## Sustained wrist-only control and the initial catch

Seven of twelve new stock releases lasted the full 30 seconds through five unannounced gravity changes. All five early losses occurred in the first four seconds. The strongest actual wrist streak was 99. Using an online wrist estimate of gravity raised mean consecutive rebounds from 48.08 to 49.25. The motor weights stayed frozen and came from bouncing on three donor arms. [Watch the first release through every change](../../artifacts/tt24/continuous-gravity-first.mp4) · [All courses and controls](../../artifacts/tt24/continuous-gravity.md)

The usual donor release distribution favors a nearly level paddle. Independently rotating every joint exposed an important limitation: stock starts rarely got past the first contact. With an external ball launcher and 0.23–0.63 seconds of preparation, a common visual and motor reference learned from donor poses took the arm from zero of sixteen ten-bounce starts to nine. On a changed arm it went from zero of twelve to three. An unweighted average of earlier postures also worked. [Body coordinates and paired starts](../../artifacts/tt24/ready.md)

Authentic native arcade Pong has yet to teach the entire five-joint bouncing skill. Native outcome credit produced more total qualifying rebounds than target-only credit in one small comparison, but a lower sum of longest streaks. Separately, two physical partner-volley policies learned up to ten and fifteen consecutive two-racket exchanges and both fell below stationary on the solo-bounce target. [Native outcome credit](../../artifacts/tt24/tt24_outcome_pong.md) · [Independent partner game](../../artifacts/tt24/partner-transfer.md)

On two visibly different games played by one donor body, a public wrist-image and encoder router learned from 24 scalar-reward trials. Across 32 fresh plays including 14 game switches it made 85 solo rebounds and 16 completed overhead exchanges, matching a selector given the actual game. The individual motor policies had already learned each task. Five of sixteen overhead starts achieved an exchange, and that head missed its initial stricter feasibility screen. [Dual-game learning](../../artifacts/tt24/dual-game.md)

## Reward-only motor learning transfers between hinged arms

A direct five-joint readout learned on three physically varied donor arms, with no action teacher, reached ten consecutive rebounds on 12/16 fresh stock SO-101 starts using wrist RGB and encoders. It averaged 11.94 against 2.56 stationary; 175 of its 191 scored impacts came with the contact point moving upward faster than 0.055 m/s. There was no stock motor practice or stock outcome used in model selection. Training received metric ball, paddle and motion observations on the donors. Stock sensing used earlier unlabelled wrist/stereo calibration. The source task was bouncing on different arms. Native arcade Pong provided no input to this successful policy. The first held stock drop ran for seventeen rebounds. [Experiment and weights](../../artifacts/tt24/tt24_donor.md) · [Watch the first stock drop](../../artifacts/tt24/donor-first-drop.mp4)

On twelve additional stock drops capped at 15 seconds, the same motor weights averaged **31.08 consecutive rebounds**, reached 57, lasted the full interval eight times, and scored zero three times. The unfamiliar tilted-axis, lower-gravity arm averaged 5.33 with the same wrist policy and peaked at 14. A matched surprise midair kick gave 41 consecutive post-kick rebounds using live feedback versus ten replaying the arm’s own commands. [Long trials](../../artifacts/tt24/tt24_donor.md) · [Disturbance and controls](../../artifacts/tt24/donor-recovery.md) · [Source learning curve](../../artifacts/tt24/donor-curve.png)

Eight scored practice attempts on each of four target physics settings raised the held sum of longest streaks from 588 frozen to 639, versus 616 with equally scaled random adjustments. Reusing the resulting weights on the source donors exposed forgetting. Broader source training also reduced performance on two of three wrist conditions. Keep the earlier donor weights as the reference. [Online adaptation and source retention](../../artifacts/tt24/donor-adaptation.md)

The practiced four-demonstration control head separately averaged 23.3 consecutive rebounds over twenty fresh 15 s stock drops after public calibration, and reached 43. Four generic calibration drops on a changed stock-family arm with a moved camera took 2.57 s, improving the same head from 4.19 to 7.94 without new demonstrations. [Changed-arm calibration](../../artifacts/tt24/public-morph.md) · [Wrist-only calibration using scene boundaries](../../artifacts/tt24/wrist-egomotion.md)


## Source learning, motor recall and public wrist calibration

A source controller learned repeated rebounds using reward alone and reached ten in all 256 held MuJoCo gantry drops. Driving a five-hinge SO-101 through an inverse movement model exposed a transfer gap even with accurate state. On the stock arm, a wrist observer fitted to 337 frames of unlabelled optical/motor probes supported 204 summed consecutive rebounds in 16 drops when paired with the already taught motor readout, versus 142 using its old observer and 40 stationary. For a practiced wrist, a separate encoder-based memory recovered an earlier motor mapping during an ongoing drop in 8–14 ticks. [Body and world transfer](../../artifacts/tt24/source-world-transfer.md) · [Motor recall](../../artifacts/tt24/body.md#switching-motors-during-an-ongoing-bounce)

The stronger native Pong actor reproduced 2,753 commands from its archived qualification. Carrying its observed lateral placements into cheap 3D reward learning helped against zero and shuffled initializations; the magnitude-matched random comparison was inconclusive. Across the next five independently trained source streams and 60 new SO-101 drops per condition, the physical summed streaks were 84 Pong, 44 shuffled, 84 zero, 105 matched random and 158 stationary. The initial positive physical seed did not generalize. [Stronger Pong results](../../artifacts/tt24/pong-world-strong.md) · [Physical confirmation](../../artifacts/tt24/native-bank-arm-summary.json)

## Direct joint control and body adaptation on November 22

A direct stereo pupil won a 25-return physical point (13 near, 12 far). Across independent receiving cohorts, the best frequency reaching ten alternating returns was 3/24. Five probes learned during a different game restored twelve legal table-tennis returns from zero on 24 receiving starts after a wiring change. The arm drove itself from the first frame. [Current research index](../../artifacts/tt24/README.md) · [Native winning point](../../artifacts/tt24/match-bridge-native.mp4) · [Changed wiring, 10 vs 0](../../artifacts/tt24/match-bridge-changed.mp4)

The stock wrist-only bounce pupil learned its motor skill from four demonstrations. On 20 new drops with a 15 s cap, it averaged 14.30 consecutive rebounds, reached 42 and survived the full time on six. A public sensory recalibration using 248 ordinary altered-body frames (4.13 s) improved mean wrist-only streak length from 4.44 to 6.38 on 16 new drops. A physically added hinge took over when the original wrist pitch jammed, earning 89 rebounds on 12 fresh drops versus 23 when both hinges were jammed. Native Pong alone has yet to establish rapid acquisition of this motor skill. The next useful distinction is between general optical/body learning and new task-specific motor learning. [Camera adaptation](../../artifacts/tt24/metric-transfer.md) · [Long-horizon controls](../../artifacts/tt24/wrist-transfer.md) · [Jammed-arm experiment](../../artifacts/tt24/redundant-jam.md) · [Learning curves](../../artifacts/tt24/transfer-progress.png)

## SO-101 rally learning on November 22

The longest observed and command-replayed point reached **62 consecutive legal returns**, 31 by each arm, from a rightward opening jog selected on the fourth point of a source-trained online stream. On the same three prior observations and same random exploration draw, flattening the Pong prior selected center. Actual center and left counterfactuals each produced zero returns. The flat learner reached the threshold on the first point of this stream. Across sixteen independent mixed streams the source prior reached ten within eight points on 13/16 versus 10/16 for the identically updated flat learner. Mean capped launch counts were 4.31 versus 5.44 (paired p=.094). On twelve additional wider streams the same learner reached ten in 4.00 versus 5.17 potential launches, with 11/12 versus 10/12 solving (paired p=.148). Both prospective comparisons estimated approximately a one-point benefit. In the latest complete game the unpracticed, frozen-practice and online arms made 45, 32 and 40 legal returns respectively. Online acquired 33 additional RGB-derived corrections. All three games reached regulation and 34 two-arm tapes replayed. [62-return video and same-history intervention](../../artifacts/tt23/implicit-witness.md) · [Prospective online confirmation](../../artifacts/tt23/implicit-confirm.md) · [Wider receiving streams](../../artifacts/tt23/implicit-wide.md) · [Ordinary coached matches](../../artifacts/tt23/coached-match.md)

The cleanest source-transfer experiment gives the arm three fixed body-relative opening actions. A preference learned from visual context and elapsed Pong time reached ten in 4.80 capped launches versus 6.09 for flat initialization on four eight-point training streams (32 independent simulated receiving starts). All 96 physical alternatives replayed. Five separately trained native fits each preferred useful actions on this bank; only 3.15% of joint within-source reward permutations matched or beat their mean launch count. The frozen pooled control happened to reproduce the same actions. Only seven of the 32 starts allowed any of the three choices to sustain ten with this continuation. On independently varied crossed flights, holding center often remains better. [Online study](../../artifacts/tt23/implicit-online.md) · [Conditional source attribution](../../artifacts/tt23/implicit-permutation.md) · [Crossed flights](../../artifacts/tt23/crossflight.md)

On six shared physical serves against the learned coach, a single wrist RGB camera reached 19 alternating legal returns and a single external camera reached 25. Nominal camera pose from issued joint commands resolved a wrist registration failure while the ball remained visible. The near arm made 27 legal returns with wrist, 26 with one external eye and 39 with fixed stereo. All six were complete simulated points beginning with the same legal near-arm service across camera conditions. [Wrist view and rally](../../artifacts/tt23/single-eye-wrist-rally.mp4) · [All matched points](../../artifacts/tt23/single-eye.md)

A separate matched partner intervention produced 46 consecutive returns, 23 by each physical racket. On that receiving start, the identical stereo arm and neutral opening reached six with the original partner. Across the 24 starts, the trained partner plus a Pong-derived opening produced five points with at least ten returns versus four with a neutral opening; only the latter reached twenty (three points). [46-return replay](../../artifacts/tt23/tt23_sequence_coach_46.mp4) · [Matched services and controls](../../artifacts/tt23/tt23_sequence_coach.md)

A live duration-updated selector with a Pong prior first reached ten on points 2, 3, 1 and 1 in four independent twelve-point streams. Flat-prior learners reached it on points 5, 3, 3 and 6. With the same physical swing and coaching, the source learners made 139 versus 91 legal arm returns over all 48 points. On eight independent confirmation streams the source reached ten on eight and flat on six, with the same point of first success on all six jointly successful streams. The source ranked three supplied spatial motor rules. Fixed displacements and ordinal motor choices are being tested separately to find out how much spatial strategy can emerge without that rule. [Learning curves](../../artifacts/tt23/online-rally.md) · [Independent confirmation results](../../artifacts/tt23/online-rally-confirm.md)

The five-axis practice partner learned from 87 earlier physical exchanges which legal feeds this optical arm could return. On eight fresh starts, with the arm frozen and the same services, the arm made 44 returns versus 19 against the stock partner. Four points reached ten and two reached twenty, versus one and zero. On a matched serve, changing the partner alone raised the rally from one to 27. Use this partner when measuring learner progress and hold it fixed across learner treatments. [Side-by-side replay](../../artifacts/tt23/coach-comparison.mp4) · [Complete paired scores](../../artifacts/tt23/coach.md)

With the original physical partner, a Pong-trained opening and a neutral opening each reached ten on two of the same 24 scheduled starts, with their first successes on start 13. Pong improved the chance of touching the ball. The same target-trained physical swing and stereo continuation were available to both. [Uninterrupted points](../../artifacts/tt23/tt23_sequence.md) · [Online comparison and retention](../../artifacts/tt23/sequence-learning.md)

An independent eight-start near-serving course produced 41 and 37 legal alternating returns, including 20 and 18 from the SO-101 respectively. Seven corrections learned from four practice points increased 10-return points from two to four on the eight matched starts and total arm returns from 22 to 55. Both 20+ points used a practiced physical stroke controller with the upper-level lateral command set to hold its current region. Applying transferred Pong gains on every late stroke reduced the arm total to 15. [41-return replay](../../artifacts/tt22/after-service-neutral.mp4) · [Matched controls](../../artifacts/tt22/lateral-rallies.md) · [TT23 research index](../../artifacts/tt23/README.md)

The end-clamped arm and five-axis practice partner produced a 23-return point in an ordinary first-to-11 match. Seven optical stroke corrections learned from four practice points raised the arm's own legal returns from 19 to 28 against the same partner across 11 points. Two practiced points exceeded ten versus one before practice. Further updates during the game produced 24 arm returns, longest 15. All three finished 0–11. The strong arm uses fixed stereo, a body model and a contact-and-landing stroke planner. [Watch 23 returns](../../artifacts/tt22/ordinary-rally-frozen.mp4) · [Three complete games](../../artifacts/tt22/ordinary-game.md)

An earlier opening-only comparison covered 176 unique matched serves. Training on native Pong duration raised arm contacts from 65 to 168 and legal first returns from 36 to 56; the held benchmark supplies a previously practiced physical motor stroke. Training on later physical robot replies separately increased points with a second arm return from 1/10 to 6/10 with the same opening. The newer three-action experiments above measure time to ten directly. Rapid adaptation when initial position and flight direction give conflicting cues remains open. [Source learning curves](../../artifacts/tt22/source-transfer.md) · [Later returns](../../artifacts/tt22/branch-motor.md)

## SO-101 cooperative rallies on November 21

TT22 reached ten to fourteen uninterrupted legal returns with fixed stereo RGB on the near arm, and 24, 10, 32 and 22 on four ordinary starts with privileged control on both sides. A real five-axis cooperative opponent and an optically learned placement target produced ten on three of six wider starts, then two of twelve with the original near stroke ranker on an independent cohort. Four-point near-arm optical calibration improved individual points but did not increase the frequency reaching ten on the twelve-start four-axis comparison. The complete alternated-service game ended 2–11, longest eight. See the distinct controls, replays and videos in the [TT22 report](../../artifacts/tt22/README.md).

The uncapped Pong lateral preference reached one first return in four, cold reached zero and native-score permutation reached two. It is premature to attribute sustained arm rallies to Pong. A motion-conditioned shared store recovered native Pong retention after arm practice. The early no-contact-planner opening benchmark measures implicit lateral transfer separately from the supplied long-rally physical planner. [Pong and arm scores](../../artifacts/tt22/spatial.md) · [Opening benchmark design](../../artifacts/tt22/anonymous-practice-design.md)

## Online adjustment during SO-101 matches on November 21

The end-clamped arm can recognize an unannounced motor delay during ordinary play through either the wrist camera alone or one outside camera alone. When a four-frame delay appeared, it identified the change after one ordinary point in four of four games (34–44 control frames). On 33 post-change starts reached by both policies it won 7 points versus 0 frozen, made 6/18 legal first returns versus 0/18, and served legally 13/15 versus 0/15. In a separate match it detected both a two-frame delay and its removal; the adaptive player lost 0–11 versus the frozen player's 1–11. Carrying only the delay learned with the other eye increased legal first returns on new opponent serves from 0/12 to 3/12 on wrist and from 1/12 to 2/12 outside. The policy still relies on known ball, court and arm geometry and a model-based stroke planner. [TT21 research index](../../artifacts/tt21/README.md) · [Watch one wrist-only point change from a loss to a win](../../artifacts/tt21/game-delay-wrist.mp4) · [Cross-eye details](../../artifacts/tt21/game-eye-carry.md)

A cheaper detector measures visible movement onset after an ordinary motor command, using zero nominal renders. On a fresh game sequence with one-, five- and two-frame delays it identified all three values from the first affected point in each lens. The wrist game improved 0–11 to 3–11, and the outside game fell from 1–11 to 0–11. Estimation cost was under six milliseconds per game beyond the shared live optical and motor processing. [Estimator and live-game receipts](../../artifacts/tt21/motion-onset.md)

Preparing the shoulder from a spatial map fitted to 15 optical exchanges improved later play on 32 matched strong-rival starts. The prepared arm legally answered 5 rival replies versus 2 centered, won 12 points versus 9, and produced two legal learner returns in 3 points versus 2. Each opened with 14/16 legal returns. A zero-practice straight-line predictor using the same shoulder motion answered 4 rival replies and won 11 points. Two new complete games tied on returns and scores (2–11, 4–11) even as the online fit accumulated another 17 optical examples. Transferring the stereo rule to 12 starts per single eye also tied the centered score. [Lateral preparation and evidence](../../artifacts/tt21/ready-position.md)

Sustained rally learning remains the main open task. In 15 physical practice points the arm learned an earlier timing choice for fast replies. This gained a second legal return on two fresh starts; the independent 32-serve replication gained one and lost one, with one second return in 15 opportunities per policy. On a limited family of services, practicing 24 physical points raised first legal returns from 10 to 12 on 20 new serves, but the strong rival answered all 12 and point wins fell from three to zero. A cooperative physical rival provided verifiably playable replies on six of eight matched openings. An RGB-driven generic stroke retargeter gained a legal continuation and a point on a fresh competitive start. [Matched rally video](../../artifacts/tt21/recovery-paired-638131348.mp4) · [Receiving lesson](../../artifacts/tt21/receive-lesson.md) · [Coach](../../artifacts/tt21/coach.md) · [Stroke retargeting](../../artifacts/tt21/apprentice.md)

Resume from the [TT21 player bundle and commands](../../artifacts/tt21/carry-checkpoint.md). After loading the new SO-101 delay state, the stored gantry policy reproduced its 7–11 game with 18 legal returns and the Pong policy reproduced all 180 routed tapes. These are separately addressed policies.

Tilting the table and net four degrees sideways gave the spatial learner a distinct adaptation test. Four observed serves reduced median lateral crossing error on ten new serves from 251 mm with the carried fit to 67 mm after the update. On eight additional serves, adaptation raised actual contacts from 2/8 for the straight-line predictor to 6/8 but landed 0/8 legal returns in both branches. [Physical table and held trials](../../artifacts/tt21/changed-bounce.md)

**Next experiment.** Use the playable coached replies to train the choice between the stock and retargeted strokes from optical landing and subsequent rally outcomes. Transfer that choice unchanged to unassisted games against the physical rival. Score sustained returns, point wins and target frames spent practicing. On the tilted table, train outgoing face angle and speed from visible landings to convert the newly gained contacts. The exact receiving shortlist now runs 2.09× faster on 36 optical windows. [Speed receipt](../../artifacts/tt21/planning-speed.json)

## SO-101 end clamp and single-camera learning on November 21

The wrist lens is outside the gripper, the base clamps below the end of the compact table, and all key single-camera tests use ordinary painted lines, orange ball and red paddle. Each camera can control an entire game independently. On a new four-frame actuator delay, a single camera inferred the delay in 28 probe commands. Fresh opponent serves with zero destination-camera calibration produced 7/20 first legal returns with wrist → outside transfer and 11/20 with outside → wrist. Each unadapted branch returned 0/20. The directly calibrated destination controls made identical decisions. For a new five-axis port permutation and sign change, a 35-frame single-eye probe restored the normal-interface actions. On one held game, transferring the mapping to the outside camera changed 0–11 to 8–11, and transfer to the wrist camera changed 0–11 to 1–11. This establishes reusable body calibration on a known fixture. General visual strategy and consistent match wins remain open. [Research index](../../artifacts/tt20/README.md) · [Cross-camera delay replay](../../artifacts/tt20/cross-camera-transfer.mp4) · [Motor maps](../../artifacts/tt20/motor-carry.md)

Eight marked then eight unmarked practice serves selected the same program as sixteen unmarked serves from the start. Their entirely unmarked held games exactly matched the frozen controller at 8–11 with eleven legal returns. Survey marks improved ball ranging while their colors interfered with the blade detector. Paddle rim dots gave the wrist measurable blade angle but competed for the ball detector. The camera-only arc prior improved forecasts in one direction without improving legal returns there. Preserve the shared body response and pursue the next ball after a successful return. Test the scored recovery and second return under changed feeds, camera placement and motor dynamics, against the existing single-eye checkpoints. The first-return and whole-match baselines are different measurements. [Longest unmarked rally](../../artifacts/tt20/so101-longest-rally.mp4) · [Marker curriculum](../../artifacts/tt20/curriculum.md) · [Single-camera practice and carry](../../artifacts/tt20/camera-carry.md)

## SO-101 follower, scored games and transfer on November 21

A factory-model server and receiver won **2/4 full-spread games with zero SO-101 target practice**, with two initial 11–7 wins followed by 10–12 and 3–11 losses on a blind replication. It served legally on 31/33 attempts, with 36 legal returns and five points containing at least two learner returns. The server tried 432 synthetic toss/stroke combinations in an independent model. With the same CAD and nominal ball/impact priors the player also won 2/4. Actual gantry experience added legal returns, 36 versus 25 across the four schedules, and five multiple-return points versus two. Twelve distinct target serves changed the selected stroke and yielded one win in four games with 27/36 legal serves. A separate 80 g wrist load reduced zero-practice returns from 9/12 to 7/12 with gantry priors on the same starts. A 24-frame optical probe identified the correct accessory mass from four candidates. Updating the independent planning model for that mass still gave 7/12. Practical first-game competence is established on this compact station, with weak evidence for a learning-speed advantage. [Watch the first game](../../artifacts/tt19/so101-first-game.mp4) · [Portable checkpoint and replay instructions](../../artifacts/tt19/README.md) · [Factory method](../../artifacts/tt19/factory-serve.md) · [Payload robustness](../../artifacts/tt19/direct-receive-payload.md)

The benchmark body is now the stock-scale SO-101 follower with the upstream six axes and inertias, wrist camera and a 115 g gripped paddle. On the compact 1.60 × 0.848 m table, a practiced player won all four native RGB games with narrow launch variation. On the full spread, direct practice won three of the first four games and lost all six fresh confirmations. Practicing progressively wider launches won three of four same-seed comparisons and two of four additional fresh games. On one matched full-spread game, removing the learned opening movements turned an 11–9 win with six legal returns into 0–11 with none. A separate factory-model receiver won 11–5 and 13–11 and made two legal returns on multiple points. Across four matched full games, carrying gantry flight and impact priors earned 34 legal returns versus 29 with nominal priors, including a 13–11 win versus a 6–11 loss. It uses calibrated RGB and a supplied forward model. [TT19 index and playback](../../artifacts/tt19/README.md) · [Full game with wrist view](../../artifacts/tt19/so101-full-game.mp4) · [Stroke learning and game counts](../../artifacts/tt19/match-learning.md) · [Factory-model receiver](../../artifacts/tt19/model-rollout.md)

With no SO-101 stroke bank and zero target practice, the factory-model receiver made a first legal return on 17/24 new opponent serves with gantry priors versus 15/24 matched nominal. It won eight points versus five. Four paired starts favored gantry on first return and two favored nominal. Both reached a legal landing at a median of 36 frames on successes. [Zero-practice receiving](../../artifacts/tt19/direct-receive.md)

Gantry-trained ball flight helped the arm serve sooner after just 24 target setup frames. Native Pong's lateral gains made the same six successful first returns in 36 starts as neutral gains. Ranking early arm strokes using gantry reward also failed to beat the no-source player. Feeding SO-101 table estimates back into the gantry reduced legal returns from 18 to five on a matched game; restoring its stored table estimates recovered all 18 tapes. The winning arm endpoints retain the original gantry and native Pong policies. The simulator uses a held grasp, known court and marked optics. Wrist-only range on the ball is sparse. [Transfer controls](../../artifacts/tt19/online.md) · [Pong comparison](../../artifacts/tt19/pong-forward.md) · [Old-task checks](../../artifacts/tt19/retention.md)

**Next benchmark.** Target a useful change in stroke execution after the first movement probe. The 24-frame optical mass estimate alone left loaded-arm performance at 7/12. Use measured blade path error to change the reachable stroke candidates or their timing. Compare against the same frozen factory player and the mass-only correction on a new load or paddle mount. Charge both target points and internal model time. Measure legal own serves, points with a legal first and second return, full-game score, and time to each on the first and subsequent games. Replay the retained Pong and gantry policies afterward. [Match receipts](../../artifacts/tt19/match-audit.json.gz)

## Changed table physics, cameras and body within scored play on November 21

TT18 introduced physically heavier balls, body and link inertia, moving camera payloads, resized tables and joint travel, and visually identical table regions with different rebound. Camera registration from surveyed colored landmarks let a player complete a camera-remount game 9–11 with ten legal returns, versus 1–11 with four on frozen calibration. All four frozen returns came before the switch. A monocular fallback tracked some short motions but still missed both live tosses. A location-specific table model cut rebound prediction error from 0.145 to 0.088 on 59 matched impacts and won one of four held games. Two independent source streams selected the same local update settings as the manually specified control. [TT18 index](../../artifacts/tt18/README.md) · [Surface results](../../artifacts/tt18/surface.md)

Switching body, table and cameras together kept each player on one scored stream. The carried gantry won seven points versus zero for a cold motor fit on one schedule, then two versus six on the replication. On the new articulated arm the carried gantry motor fit made zero legal returns; restarting the fit produced one after the change. Choosing among old and new fits restored one legal own serve. The carried gantry endpoint retained all 180 reference Pong tapes. [Actual camera replay](../../artifacts/tt18/changing-table-and-views.mp4) · [Body comparisons and checkpoint](../../artifacts/tt18/combined.md) · [In-game motor and service learning](../../artifacts/tt18/online-body.md)

**Next pass.** Target the first useful stroke on a new articulation. Use optical blade response to detect when a practiced stroke is unreachable and choose a short, feasible exploratory movement during available play time. Compare that choice against passive play, a fresh motor fit and the old fit on two previously untested arm geometries. Match elapsed control frames as well as starts. Score time to legal return, legal own serve, repeated returns and the complete game. Keep game knowledge, camera landmarks and interaction costs explicit. Demonstrate a gain on both geometries before extending source training or adding more environment variations. [Plan](../plans/2026-11-21-continual-world-and-body-learning.md)

## Continuous motor calibration, changed-arm wins and a service bottleneck on November 21

A previously trained table player used 24 changed-control frames and a 68-frame original-control camera setup to fit continuous gain and cross-axis motion from stereo. On four fresh altered installations its held-out movement-change prediction error fell from 53.3 to 18.0 mm. With the existing stroke controller it landed 32 legal returns versus five for signed-only calibration, but lost all four games. Established TT15 in-game calibration landed 30 and won one. The new bounded fitter accepted no late updates in these games. On four additional installations that changed motor order, sign and coupling together, the new fit raised legal returns from six to 37 and multiple-return points from one to seven. All four were losses. [Motor evidence](../../artifacts/tt17/motor.md) · [Compound changes](../../artifacts/tt17/compound.md)

Applying the same travel constraint in the learned arm coordinates turned 3–11 into **11–9** on the fresh coupled gantry, with 22 legal returns on 20 points. Six points contained at least two legal returns. Across the four fresh games, points won rose from 11/55 to 30/72 and returns from 32 to 46. Forking the exact winning endpoint onto a fresh game without further setup produced a 10–12 loss versus 9–11 with the old travel limit. An alternative rate allocator won 12–10 on the weakened gantry but lost on the coupled gantry. The five-axis arm lost every altered game, and the changed controllers also lost on the repeated TT14 matrices. Resume the 11–9 checkpoint using the saved installation and controller via the linked continuation script. [Full game video](../../artifacts/tt17/gantry-coupled-full-game.mp4) · [Checkpoint and comparison](../../artifacts/tt17/allocation.md) · [Resume procedure](../../scripts/experimental/tt17_allocation_resume.py)

The bounded-calibration coupled gantry checkpoint preserved all 180 reference Pong action tapes and 41 earned points. The newly trained motor map also predicted movement better on an inertial puck task. Time to first dock averaged 124 frames versus 136 cold and 123 with the signed-only map. On six-joint rotary the camera often predicted the second ball accurately while the blade arrived too low or slow. Neither weighting the deadline nor matching the previous stroke speed improved distinct multiple-return points on the replication. On familiar gantries, a reward-trained choice of interception plane turned a loss into a win and then a win into a loss on the repeat. Keep the ordinary receiving stroke as a control. [Puck task](../../artifacts/tt17/payload.md) · [Rotary outcomes](../../artifacts/tt17/rotary.md) · [Learned stroke choices](../../artifacts/tt17/strikes.md)

**Next pass.** Target learning a legal own serve under altered arm dynamics. The weakened five-axis player delivered zero legal serves in six tries even with corrected travel. Freeze the motor fit, start from the established service skill, and let the arm test small toss-to-contact timing and stroke variations from optical blade/ball motion and its ordinary reward. Score legal serves, subsequent legal returns and full-game wins separately. Pair fresh preset streams across controls, include unchanged hardware, and transfer the learned setting to a second installation without selecting a controller by fixture name. On six-joint rotary, prioritize a genuinely different preparation or recovery movement that puts the blade in a reachable position before the second strike. [TT17 index and receipts](../../artifacts/tt17/README.md)

## Active motor identification, held-out transfer and puck control on November 21

A single 12-command optical movement on previously unseen signed wirings led to wins on all four fresh gantry/five-axis fixtures, 83 legal returns and 18 points with at least two returns. Direct remapping from scored points won one game with 54 returns. The search-to-table rule frozen after TT15 won both gantry games and lost both five-axis games with 57 total returns. The direct movement reached the first legal return within 125–129 control frames on every fixture, including the 68-frame camera setup performed on the original wiring. After a reload the five-axis player won its next game 15–13 with no further setup. Resume from the [second-game endpoint](../../artifacts/tt16/coded-five-resumed-player.json.gz) with its saved TT16 camera and wiring. [Full first-person game](../../artifacts/tt16/five-axis-full-game.mp4) · [Held-out comparison](../../artifacts/tt16/transfer.md) · [Resume procedure](../../scripts/experimental/tt16_coded_resume.py)

The search-conditioned alternative retained all plausible signed maps and queried its two most uncertain ports. On two more cameras and wirings it recovered both maps with 4.2 units of requested motion at handoff, versus 28 and 35 for the direct all-port query. It reproduced both complete game tapes of the direct calibration. Including the previous search still made it slower. When latency also changed during calibration, choosing among optical timing hypotheses recovered 9/12 whole maps versus 3/12 assuming immediate response. On 24 further wirings a continuous second movement raised exact fits from 11/24 to 20/24, using 23 frames on average. Its four fresh games produced 44 returns versus 20, including one 12–10 gantry win and one five-axis regression. [Sparse query evidence](../../artifacts/tt16/sparse.md) · [Calibration and timing](../../artifacts/tt16/calibration.md)

The table-trained motor geometry helped a new physical-arm task involving an indirectly accelerated puck reach its first goal in 100 frames on average versus 109.25 with a cold learner. Both earned 128/128 holds across 32 courses. This tested transfer of motor geometry into a supplied task controller. The search-assisted gantry and directly calibrated five-axis checkpoints retained all 180 reference Pong tapes. On six-joint rotary, the independent replication reversed the small first-cohort gain from a lateral sweep. Across 60 matched rival serves the inherited stroke produced five distinct multiple-return points and the sweep four. Neither full-game variant returned two balls on a rival-served point. [New puck task](../../artifacts/tt16/payload.md) · [Rotary outcome audit](../../artifacts/tt16/rotary.md)

**Next pass.** Test whether the active optical movement and an online correction can identify continuous cross-axis coupling and gain changes outside the signed permutation family. Start with the existing TT14 actuator fixtures plus disjoint motors and actual full games. Compare the TT16 signed fit, the established TT15 in-game correction, and a bounded local correction built from the same 12–24 camera frames. Include stable installations, score points with two legal returns explicitly, and charge active movement. For rotary, the first-stroke alternatives have not generalized and the optical second-landing cue recognized only one of three known examples in its historical audit. Measure where a reachable second reply is lost using matched camera tracks and the reconstructed arm response, then test a correction to intercept timing on reachable replies. Keep diagnostic physics in the evaluator. [TT16 index](../../artifacts/tt16/README.md)

## Cross-task motor transfer and selective relearning on November 21

After searching for an invisible scalar source on unfamiliar controls, the arms returned to first-person table tennis. Recent search motion recovered all 18 port-to-axis correspondences across four fixtures. Three wrist signs were wrong: the weakest motor translated the tool only 3–5 mm per unit and search did not see its angle. Search transfer alone won one of four complete games and landed 20 legal returns versus zero for the frozen original motor map. A 12-command, 0.4-second pitch probe at handoff resolved the signs, yielding three wins and 61 returns. Without the search correspondence the same probe yielded zero returns in four losses. Continuing optical remapping directly from the original table player also won three and landed 61. Resume from the [five-axis winner](../../artifacts/tt15/pitch-five-0-player.json.gz) with its saved installation. [Watch the gantry play](../../artifacts/tt15/search-to-table.mp4) · [All transfer controls](../../artifacts/tt15/back-transfer.md) · [Pitch evidence](../../artifacts/tt15/pitch.md)

In an uninterrupted hidden-source task, separate motion and reward prediction errors identified all 40 switches between previously learned motor maps and 37 of 40 source relocations. Twenty stable intervals gave zero alarms. Selective updates kept the arm on target for 1,338 frames during motor-only changes versus 886 when resetting both fits. Over all conditions the comparison was 4,566 versus 4,417 frames and 64 versus 66 acquisitions. On wirings entirely new to the arm it could create additional maps, improving acquisition over a cold start but also creating redundant models. [Trajectories and controls](../../artifacts/tt15/context.md)

In table tennis, requiring a separate point to corroborate a motor correction cut false switches but missed both restorations. Short dry confirmations recognized changes on four harnesses and reproduced the fixed player's 38 action tapes on unchanged controls. Across matched and untouched harnesses they produced 35 changed-control returns versus 27 with the old selector and four with fixed controls, with mixed game wins and an explicit probe cost. On six-joint rotary, none of the three new recovery variants increased distinct sustained points in three complete fixtures. All full-game second returns arose on the same incoming point. [Motor tests and probe cost](../../artifacts/tt15/motor.md) · [Rotary evidence](../../artifacts/tt15/rotary.md)

**Next pass.** Freeze the search-to-table transfer and single-probe rule, then test new wiring and camera seeds chosen before observing results. Include equal-probe/no-search and direct table-online controls. Measure time to first legal return and points per total control frame, charging search exposure and dry actions separately. For rotary, change the intervention: test whether the player can use the incoming trajectory and its own first-stroke outcomes to choose shots that induce a reachable second ball. Compare against the inherited first stroke on matched serves, then count distinct points with multiple legal returns and full-game scores. [TT15 research index](../../artifacts/tt15/README.md)

## Coupled actuators, rotary recovery and hidden reward changes on November 21

Online motor correction generalized beyond signed permutations. Learning from unannounced coupling during scored points cut fresh physical motion-change error from 54 to 28 mm on gantry and 78 to 33 mm on five-axis. The first changed gantry game was an 11–4 win versus a fixed map's 4–11. On a second wiring and independent fixture, the correction lost 6–11 versus 4–11 fixed, and tied local refitting on score. Five-axis lost 3–11 versus 1–11 fixed. Both arms won on return to the original controls, as did their respective fixed controls. The learned player recognized the originals after one point and later switched away despite stable hardware. On separate unchanged fixtures the selector switched four times on gantry and five on five-axis. Under reduced gain alone the adaptive players landed 11 and 10 returns on changed controls in their first games, versus zero per fixed control. Keep the continuous motor correction experimental until it can distinguish genuine changes from ordinary prediction error. [Watch the gantry game](../../artifacts/tt14/gain-gantry-coupling.mp4) · [Motor games and checkpoints](../../artifacts/tt14/gain.md) · [Pulse audit](../../artifacts/tt14/signal.md)

On six-joint rotary, more outgoing pace or loft improved points on two fixtures but produced zero second returns. Between-stroke recentering gave the arm more usable motor travel and reduced near-impact vertical error. In a fresh paired cohort the baseline returned the same serve once, and the recentering player returned it six times and won the point. All five later successes in that cohort came from this one rally. One second return on rival service in the initial complete games failed to repeat on rival service in the next games. The arm still lost all its complete games. The five-axis and rotary players preserved all 180 native Pong reference tapes. [Six-return rally versus matched control](../../artifacts/tt14/rotary-six-returns.mp4) · [Rotary tapes, continuation checkpoints and diagnosis](../../artifacts/tt14/rotary.md) · [Research index](../../artifacts/tt14/README.md)

A genuinely different task asked the same arms to find an invisible source from scalar strength and RGB tool position. On the coupled five-axis arm the original and corrected table maps each acquired 11/12 targets versus 8/12 with a fresh motor learner. With the earlier signed rewiring and equal probes, starting fresh spent more time on target on both bodies. When the source moved mid-trial, detecting unexpected signal readings and resetting the reward fit recovered 14/16 targets, versus zero for ongoing fitting of one reward surface. A fresh motor learner also recovered 14/16. [Transfer, stable controls and acquisition times](../../artifacts/tt14/signal.md)

**Next pass.** Test how to recover for a second shot on the rotary arm using the actual optically observed blade position and time available, rather than applying the same neutral return after every strike. Start on identical fresh rival replies, retain a blind fixed-center control, and require gains on several distinct points before promoting the change. Separately, test a controller that distinguishes a change in the arm from a change in the task objective: give it interleaved motor-only, objective-only, simultaneous and stable episodes, with no change labels, and compare to resetting both learned models. The motor-model and reward-prediction residuals used in TT14 supply independent observations for that decision. [Motor model](../../scripts/experimental/tt14_gain_model.py) · [Signal change detector](../../scripts/experimental/tt14_signal_change.py)

## In-game motor adaptation and its limits on November 21

A four-axis gantry and a five-axis arm recovered four unannounced signed motor permutations from ordinary visual play, each after one point with changed controls. They also recognized their original controls after one point on restoration. Across six complete games, adapting won three and landed 90 legal returns versus zero wins and 36 returns with fixed motor maps. During the intervals that actually used changed wiring the contrast was 54 returns versus zero. Three transient false selections corrected on the following point, and a stable-control fixture preserved all 18 point tapes. The five-axis player after its 14–12 win preserved all 180 native Pong action tapes. Its new table-tennis map and the gantry's transferred into a moving-target task on the same bodies, raising time within tolerance from 2,431 to 3,234 frames out of 6,144 relative to learning in pursuit alone. [First-person 14–12 game](../../artifacts/tt13/five-revisit.mp4) · [Gantry checkpoint](../../artifacts/tt13/live-gantry-remap-player.json.gz) · [Five-axis checkpoint](../../artifacts/tt13/live-five-remap-player.json.gz) · [Exact schedules and audits](../../artifacts/tt13/live-ports.md)

Return-depth learning on the gantry won 11–9 and 11–7, compared with 11–9 and 10–12 on the original stroke. On 13 matched playable balls the frozen learned choice produced 16 legal returns and six points, compared with 22 returns and seven points on the original stroke. The five-axis arm lost the transfer game 6–11 versus its original 11–4. Keep placement experimental. The six-joint rotary arm improved first returns on rival service from 1/14 to 5/14 across two games with three-probe geometry recall, but the scores stayed 2–11 and 4–11. Even after improving the optical blade estimate, the learner failed to make a second return on rival service. [Placement evidence](../../artifacts/tt13/place.md) · [Rotary evidence and checkpoint](../../artifacts/tt13/rotary.md)

**Next pass.** Try an unannounced motor-gain change or cross-axis coupling outside the learned permutation family. Compare an online correction around the recalled map with pure remapping and local fitting, using actual game points and a return to the original controls. In parallel, take the rotary arm's eight held contacts that landed on its own half and test whether the player can learn an outgoing-speed correction from observed flight. Verify on identical incoming serves and then on fresh full games. Hold the service controller fixed, and count second returns separately. [TT13 research index](../../artifacts/tt13/README.md)

## Recovering motor maps and revising service-role choices on November 21

Two 34-frame RGB probes recovered an earlier motor map on the four-axis gantry that had lost 0–11. On 16 fresh own tosses the standard player landed 1, the new player landed 14 and won 8 of those points. Transferring just spatial coefficients gave 12 legal serves, just timing gave 1, and a different tracker reference alone gave 1. Across two initial full games, the new player sustained six rallies and landed 21 legal returns, versus two and 11 for normal commissioning, but both scored five points and lost both games. From the checkpoint after its 2–11 game, ordinary carried updates won the next two independent fixtures 11–6 and 11–8. Starting each of those directly from the original two-probe player gave 11–5 and 7–11. A further correction learned from match RGB made 27 legal returns in a 12–10 repeat win but reduced returns in the first fixture. [Evidence and exact checkpoint histories](../../artifacts/tt12/motor-recall.md) · [First two held tosses on video](../../artifacts/tt12/gantry-motor-recall.mp4)

The reward-driven five-axis timing selector learned separate choices by optically recognized service role. On C it won 14–12 in adaptation, but its held choice lost 8–11 and reversing the assignment won 12–10. On D it lost 11–13 in adaptation and won the next game 16–14, where reversing the assignment lost 7–11 and either fixed model lost 8–11. Receiving wins on matched starts tied the prior dry-swing choice on both installations. The D checkpoint retained all 180 native Pong action tapes. On a physically different six-joint arm, cursor-fitted timing improved first returns from 2/12 to 6/13 on rival service and still lost both complete games. The gantry→rotary→gantry revisit lost 9–11 versus a matched restart's 11–7. [All comparisons](../../artifacts/tt12/README.md) · [Role learner](../../artifacts/tt12/online.md) · [Cross-body carry](../../artifacts/tt12/body.md)

**Next pass.** Resume the [two-probe gantry checkpoint after its first game](../../artifacts/tt12/motor-gantry-recall-player.json.gz) for sustained exchanges; use the [original two-probe start](../../artifacts/tt12/motor-gantry-start.json.gz) for reset controls. Test whether reinforcement from the visible landing location of the player's own returned ball can improve *where* it places a legal return across future exchanges. Compare identical incoming replies and qualify on fresh complete games with the now-reliable service map. Count wins on playable replies and opportunities for second and later returns. Carry the learner into a different installation and revisit the original with equal game budgets. For rotary, prioritize landing the first opponent-served ball: qualify port-aligned geometry recall from earlier six-joint experience against an equal-probe cold fit, then inspect optical contact versus failure on the same balls before spending more games on timing selection. Keep the new role learner as a measured five-axis comparator; the reversed policy winning on C rules out promotion across installations.

## Choosing motor timing from fast stereo on November 21

The arm compared its inherited and cursor-trained timing estimates using eight dry swings, 5.6 seconds of simulated movement. It picked the model with smaller two-frame stereo blade-motion error before the fixtures. Two previously unseen five-axis wirings and appearances played three full games each, including an independent repeat with the same frozen selection. The player won 4/6, scoring 63 points, making 81 legal returns and sustaining 23 rallies. Fixed table timing won 2/6 (48 points, 76 returns, 21 sustained), and fixed cursor timing won 3/6 (62, 70, 18). Across ten fixtures on four installations the advantage was one game over either fixed policy. Each primary fixture started fresh from the commissioned player and learned throughout the game. We separately carried fixture two's player into fixture three. D won 11–3 and C lost 9–11, versus 11–4 and 11–8 when restarted. C and D are different installations of one arm design. Physical-body generality of this choice is still open. [Reports, replay and checkpoints](../../artifacts/tt11/README.md) · [40-second first-person game](../../artifacts/tt11/timing-d-full-game.mp4)

A second route learned ten actual strokes from successful camera-observed far-half bounces. Re-use slightly increased first legal follow-ups on identical replies, but cost reply wins. Keep it experimental. In a new moving-target task, the table-trained motor map helped both five-axis installations and the six-joint arm under rotated camera coordinates. A generic settling curve worked well for this continuous movement and failed on the table-tennis serve. [Stroke recall](../../artifacts/tt11/stroke.md) · [Transfer beyond hitting](../../artifacts/tt11/pursuit.md)

**Next pass.** Use the [post-game D player](../../artifacts/tt11/carried-d-player.json.gz) for a continuous revisit, the [fresh D selection](../../artifacts/tt11/timing-d-selected-start.json.gz) for matched comparisons, and [the integrated player](../../artifacts/tt8/span-after-four.json.gz) for cross-body tests. Test the timing choice on actually different actuators (gantry and six-joint rotary) with opponent service and own service scored separately. For the gantry with failed serves, first compare the learned serve with its known competent gantry serve on the same tosses; a game with zero legal serves cannot qualify follow-up learning. In parallel, let the player revise a close timing choice during play using its own point rewards, conditioned on the visually recognized service role. Give the fixed choices and adaptive player equal practice and action budgets. Qualify on disjoint full games and after returning to an earlier body. Measure conditional return rates and point wins as well as sustained counts, since different-length games produce different opportunities.

## Table tennis and 3D cursor transfer on November 21

The RGB-calibrated table-tennis motor map reached 297/336 targets on a new 3D cursor task across two five-axis installations and a six-joint arm. Target-only methods reached at most 220 on familiar wiring. After rewiring before play the carried learner reached 209 against 222 for target-only probing. Detecting surprise and re-probing during play recovered 70/240 targets across the changed and restored phases versus 23/240 with ordinary carried updates. The benchmark supplies marker identities, metric axes and the reaching policy. The learned reusable component is the response of the arm to its motors. [Forward transfer, controls and replay](../../artifacts/tt10/target-transfer.md)

Fitting actuator timing on 16 cursor targets per installation turned 8–11 and 3–11 losses into 11–9 and 12–10 wins on five-axis A; on five-axis B it turned 7–11 and 6–11 into 6–11 and 3–11. The second fixture on each installation held commissioning and the visual tracker state identical. For A, resume the [12–10 player](../../artifacts/tt10/target-time-a-qualified-player.json.gz). For the general player keep the [TT8 integrated checkpoint](../../artifacts/tt8/span-after-four.json.gz), or the [registered control](../../artifacts/tt9/precontact-match-b-player.json.gz) when studying five-axis service. [Reverse transfer](../../artifacts/tt10/target-return.md)

**Next pass.** Test whether a player can select the inherited or cursor-trained timing model from its own visual evidence at table-tennis speeds, before scoring the fixture. Compare to both fixed models with equal probe budgets on fresh installations, using sustained returns and full games. Small gains in slow cursor prediction occurred on both A and B and failed to identify which model to use for striking. In parallel, test recalling and adjusting short strokes from the player's own previous legal returns, using observed ball flight and achieved blade motion. Compare on identical incoming replies, then qualify in independent full games. Mülling and colleagues learned a repertoire from human demonstrations; the open question here is whether strokes collected by the player itself can serve this role. [Robot table-tennis skill learning](https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Muelling_IJRR_2013.pdf)

Longer play with the existing updates produced the same total points and sustained exchanges as restarting between games. Keep its ten fixtures as a reusable control. Adjusting intercept depth and time won one five-axis game but reduced subsequent legal returns. Serve placement moved the opponent but reduced held-out points won on serve. [Five-game series](../../artifacts/tt10/series.md) · [13–11 first-person game](../../artifacts/tt10/five-axis-deuce.mp4) · [Experiment index](../../artifacts/tt10/README.md)

## Five-axis service and match retention on November 21

Twelve practice tosses per installed five-axis arm raised legal service on fresh throws from 15/24 to 23/24. Local ridge fits used camera-measured blade–ball displacement, including on complete misses. The learned sideways offsets had opposite signs on the two installations. Full alternating-service games with ongoing adaptation ended 5–11 and 5–11 versus fixed 4–11 and 6–11. The general starting checkpoint is still [the TT8 integrated player](../../artifacts/tt8/span-after-four.json.gz). For service and follow-up research, resume the [five-axis B player after a complete game](../../artifacts/tt9/precontact-match-b-player.json.gz) with standard `tt8_service_trial`. [Same-toss first-person replay](../../artifacts/tt9/five-axis-same-toss.mp4) · [All comparisons and reproduction](../../artifacts/tt9/README.md)

The next target is landing the **reply to a legal own serve**. On installation A the opponent answered all five of the learned player's legal serves, and the player landed zero follow-ups. A separate fixed-drive player hit seven of seven playable replies and landed three. Longer extrapolation and an earlier lift reduced legal returns on the same seven points from ten to eight. Test selecting the contact time and depth for each incoming trajectory using the camera-calibrated arm's predicted achievable blade velocity. The current planner largely aims at one depth, and a uniform depth shift already failed. Compare against the existing stroke on identical pre-point histories, then carry updates through full games on independent installations. Judge by legal follow-ups, later returns and points won. Koç et al. found trajectories generated at a fixed hitting plane can be infeasible and optimized striking time jointly with the arm trajectory. [Recovery evidence](../../artifacts/tt9/recovery.md) · [Earlier fixed-plane change](../../artifacts/tt7/contact-timing.md) · [Robot trajectory paper](https://www.ias.informatik.tu-darmstadt.de/uploads/Publications/Online_optimal_trajectory_generation.pdf)

On a third geometry the carried learner lost 3–11, as did the player that reset only the service correction. After this game the first arm scored 7–11 on its revisit versus 11–13 directly. Restoring only its original registration yielded 6–11. The continuous B–C–B checkpoint preserved all 180 native Pong tapes. The six-joint lift transfer and screened cart tests are also complete. [Installation retention](../../artifacts/tt9/installation-switch.md) · [Rotary games](../../artifacts/tt9/rotary.md) · [Cross-game transfer](../../artifacts/tt9/catch-transfer.md)

## Integrated Pong and table-tennis retention on November 21

Resume from the [full player after four arm games and intervening Pong practice](../../artifacts/tt8/span-after-four.json.gz). The hall port now carries the mini player's earlier motor-response-conditioned scores. One checkpoint kept the original native Pong actions on 180 starts, including 41 earned points, while allowing the table-tennis preference to change. It won 11–6 on the gantry, lost 3–11 on offset, won 11–8 on the returning gantry, played 32 disjoint native Pong points, and then reproduced the arm-only control's 10–12 game and 23 legal returns. All 76 played table point tapes matched their controls. The learner identifies the response span from visual motor probes. [Report and reproduction](../../artifacts/tt8/motor-span.md) · [Full first-person game](../../artifacts/tt8/full-game-first-person.mp4)

The next target is the five-axis serve and follow-up. Keep the original velocity-controlled serve as a strong fixed comparator. Its success ranged from 2/8 to 7/10 across soft/noisy toss cohorts. A six-parameter search succeeded 30/35 times on seven practice tosses and only 2/10 on new tosses. Across 41 learner serves in the two fixed-drive cohorts, the additional offset search control and the two frozen-drive games, four-frame optical departure speed was available on 18/20 legal serves and 10/21 failures. Use the blade and falling-ball tracks before impact as well, so missing outgoing evidence does not discard over half of the failures. Measure contact timing and outgoing flight from RGB and test whether these observations predict which corrections to make on new throws. Use distinct throws during learning and qualify on multiple installations with a full alternating game. A separate chronological transfer test found that current table-tennis bounce evidence helped cart catching versus shuffled evidence when the material continued (184/256 versus 149/256, generic prior 178/256). Unexpected material changes favored the generic player (167/256 versus 134/256). [Serve-search evidence](../../artifacts/tt8/service-search-offset.md) · [Chronological rebound transfer](../../artifacts/tt8/rebound-transfer.md)

## Serving, receiving and returning to the same physical arm on November 21

In a first-person alternating-service game, the gantry won 11–6 after 18 practice serves, versus 4–11 when it began service learning in the game and 3–11 for the receiver without a service stroke. Fixing the stroke to the one chosen most often in practice reproduced the win. The complete carried player lost 3–11 on a five-axis arm, then won 11–8 on returning to the original gantry, recalling its motor map in two probes. The five-axis player made zero of six legal serves with the original set of strokes. The next control target is adding enough forward speed on this body without losing the ability to sustain an exchange. [Full first-person game](../../artifacts/tt8/full-game-first-person.mp4) · [Reproducible comparisons and complete player checkpoint](../../artifacts/tt8/README.md)

The first full game's portable state preserved all 180 native Pong command tapes. After three games, a different shared interception relation was preferred. Native earned points fell from 41 to 36 and later returns from 94.7% to 87.4%. A bounce coefficient learned from table-tennis stereo helped catch a ball in a different game when the physical material continued (300/384 versus 289/384 for the strongest target-only control). Unannounced material switches erased the advantage. [Rebound catching and controls](../../artifacts/tt8/rebound-transfer.md) · [Native Pong after three games](../../artifacts/tt8/pong-after-three-games.json.gz)

Under changing table compliance, the RGB estimator distinguished all four physical surfaces, but using the fit won fewer total legal returns than keeping the old trajectory planner (38 versus 45 on 48 points). The six-joint adaptive lift improved on cold timing and the aligned fixed lift under mixed conditions, but did not beat the strongest fixed early lift in both games. [Surface comparison](../../artifacts/tt7/surface.md) · [Six-joint mixed timing](../../artifacts/tt7/contact-mixed.md)

## Six-joint soft-racket games and transfer to catching on November 21

A near-contact lift selected from camera and point-reward experience returned 14/20 openings after 24 practice points, versus 10/20 starting timing learning on the test course, 12/20 for a fixed aligned lift and 2/20 for the earlier receiver. On two independent six-joint stations the practiced player scored 7–11 and 11–9, versus 5–11 and 3–11 for the inherited player. Across bodies, correcting stroke predictions with measured outgoing flight won two gantry games, but on a fresh set of installations the fixed correction had the best five-axis score. Using camera-updated delay in a new fruit-catching game gained 21 catches in 432 falls over freezing it. The table-tennis timing prior tied a target-only start. Protecting the 42-point checkpoint while keeping three later motor maps saved 54% of probe commands when those stations returned, with equal wins and fewer sustained rallies.

The next pass tests learned timing against the fixed lift under mixed conditions, hidden table compliance, transferable bounce physics and physically serving from a near-side toss. The game scores above are receiving games in which the opponent starts every point. [Evidence, checkpoints and reproduction commands](../../artifacts/tt7/README.md) · [First-person rally](../../artifacts/tt7/contact-soft-five-returns.mp4)

## Continual arm, timing and racket adaptation on November 21

The saved player returned 31/42 openings and sustained 14 points on six new gantry and offset-arm installations after varied practice. Equal-duration ordinary practice produced 21/42 openings and ten sustained points. Mini-only produced 25/42 and seven. At 84 practice points performance slipped to 27/42 and nine sustained. On two genuinely new five-axis geometries with 13–19° tilts, the 42-point player returned 8/14 versus 3/14 for mini-only. Recurring actuator wirings needed 39% fewer in-game probe commands, and visually adapting to a five-frame delay produced six legal openings in eight points versus zero from fixed timing. The experienced player still lost its two extended receiving games, 9–11 and 4–11. Six-joint mixed-condition play is the hardest remaining test. [Report, four-arm figure and replay](2026-11-21-continual-arm-adaptation.md) · [Carried checkpoint and runnable comparisons](../../artifacts/tt6/README.md)

## First-person receiving games in the MuJoCo hall on November 21

The arm returned the opening ball on 50/50 held-out points and made a second legal return on 39. The previous controller produced 14/50 and one sustained point on the same serves. A body model fitted from commanded movement and first-person stereo was essential. On a separate four-installation course, adaptive calibration took 3.4–5.7 seconds of simulated interaction and the arm returned 24/24 openings. Two further first-to-11 receiving games ended 11–2 and 11–5. Shared preferences extracted after one complete hall course passed the existing native Pong retention test on 180 starts. The Pong-only donor nearly tied the original mini donor in the main qualification. The stroke planner, optical selection and landing objective have substantial table-tennis-specific structure. The next transfer target is learning how to change the outgoing stroke after an unannounced change in racket or table response. [Report and first-person rally](2026-11-21-first-person-hall-learning.md) · [Checkpoint and commands](../../artifacts/tt5/README.md)

## Mini table to a four-axis MuJoCo racket on November 21

Both tables now have player-eye stereo. Original mini practice improved first-person projected-RGB receiving play to 45/72 points with a legal return, versus 25/72 for Pong-only, with 12 versus three sustained points. In the stricter shaded practice hall on five fresh installations, the same player landed 9/40 openings versus 4/40 for Pong-only. The default first-person mini player also landed 4/40. A separate matched hall tape shows that player's two-return winning rally. Eight target-side motor probes preceded each shaded match. The next target is to preserve useful stereo tracks through approach, bounce and blade overlap and to improve legal landings after contact. Use the furnished hall, frozen projected reference and held-out installs to test changes. [Research report and POV replays](2026-11-21-realistic-table-tennis-transfer.md) · [Checkpoints and commands](../../artifacts/tt4/README.md)

## From native Pong to a 3D table-tennis arm on November 21

An anonymous RGB tracker and a numeric, score-trained spatial policy now transfer from native Pong into a simulated three-joint racket arm. With the learned reward-time model and unrestricted meeting times, the final player landed 132/160 openings and recorded 10 sustained rallies versus 41/160 and four from scratch. All eight trained players achieved a legal return within five points. It kept playing native Pong after table practice. On 240 matched points, first returns moved from 165/201 to 166/201, later returns from 246/262 to 247/263, and earned points from 48 to 49. After extending the same table course to 64 points it had landed 52 openings, sustained 11 rallies and won 13 points. Native Pong again passed its three retention tests. The remaining performance gap is controlling the outgoing shot on a fast opponent reply. [Research report and replays](2026-11-21-3d-table-tennis.md) · [Checkpoints and reproducible runs](../../artifacts/tt3/README.md)

## Pong through a three-joint arm on November 21

The carried player won 24/24 first-to-three simulated Pong matches through two physical touch buttons, saving 93/100 incoming balls and earning 56 points after a return. With the same arm and timing rule, a fresh Pong learner won 12/24 and reached the 90-second cap on the other 12. The successful arm controller uses the estimated time of ball arrival and continued paddle travel while a finger is leaving a switch. In a separate challenge the learner independently identified both button effects in 0.73 s and recovered from a hidden reversal in 1.18 s (medians, 16 streams). On 12 additional starts, the practiced arm won 11 with 120 Hz arm control and 10 when both cameras and control ran at 30 Hz. The fresh Pong learner won four at 30 Hz. The historical scoring checkpoint had no measurable scoring advantage on its small arm course. All results are in simulation. [Research report and two-camera replays](2026-11-21-pong-through-an-arm.md) · [Checkpoints and game records](../../artifacts/arm-pong/README.md)


## Air hockey, changed Pong and an RGB camera on November 20

Four hundred air-hockey practice points improved final-touch goals from 166 to 228 on 288 held-out starts. A further 360 points against equally mobile rivals took held-out goals–concessions from 14–22 to 40–9, with 143/192 timeouts. Pong-derived wall estimates improved hockey trajectory forecasts. Varied hockey practice also taught a faster adjustment rate that helped actual native Pong when a camera cover hid an incoming rebound. Two independent changed-wall tests saved 225/256 and 222/256 first balls versus 21/256 and 10/256 after ordinary Pong practice. Diverse Pong walls also helped. Clear-camera game outcomes largely tied. On the arm, online RGB recalibration reached 191/192 physical goals after camera and actuator changes, compared with 34/192 under a stale calibration. The next major question for the arm is a realistic calibrated joint-target interface with depth and occlusion. [Full report and replays](2026-11-20-air-hockey-and-camera.md) · [Playable recordings, learned states and evidence](../../artifacts/air-hockey-camera/README.md)

## Cross-body learning on November 20

The latest milestone moved beyond paddle games. In simulation, five coefficients learned from scalar reward on an XY carriage let a fresh three-joint arm hold the paying marker first on 57/64 trials versus 27/64 from scratch. A separate 64-stream course reached 53/64 after eight carriage practice reaches. A source motor map helped a related arm with new actuator dynamics. Copying it across the carriage-to-arm body change and importing a source-selected exploration schedule showed no clear benefit.

A follow-up detector that discards strongly contradicted reward estimates improved recovery from a full inversion, with no false resets on the stable courses. Its advantage on independent replacement rules was inconclusive. The current controller uses segmented marker images, learns within a supplied degree-two visual feature family and starts the receiving arm with fresh motors. The next hardware-facing target is a simulated SO-101-style position interface with delayed and degraded observations and hidden actuator changes, before considering real-arm operation. [Research report and replays](2026-11-20-cross-body-transfer.md) · [Trained states and experiment records](../../artifacts/body-transfer/README.md)

## Pong to Breakout and back on November 20

The carried Pong interception skill returned 153/153 incoming balls on an independent ordinary Breakout follow-up when offensive aiming was reserved for a visible opponent. After 12 Breakout practice lives, a visual-rebound learner cleared 12/12 new full boards versus 6/12 for the same model before practice. A single continuously updating controller then passed two 36-life alternating-game courses, including opposite button meanings and an extra six-command delay in Breakout. On re-entry to that conflicting course it saved 21/21 Pong and 82/82 Breakout balls, with nine own-return Pong wins and 80 bricks.

The practical next milestone is learning the scoring objective from disappearing objects and score changes. Today’s system still contains a hand-designed visual scene rule and target proposal. Start with an identical-looking board where the paying target bank changes. [Progression report and real-game replays](2026-11-20-pong-breakout-transfer.md) · [Checkpoints and loading examples](../../artifacts/m2/README.md)

## Pong remapping, timing and score learning on November 19

The three independent qualifications passed. The rapid controller returned 27/30 first balls after hidden mid-rally remaps. With hidden actuator delays, the timing learner returned 21/32 first balls versus 6/32 with the timing estimate frozen. The score-trained policy with the visual safety rule won 18/32 after an own return versus 0/32 centered, and maintained 65/66 later returns against the stronger opponent. Its camera and learned interception relation are inherited. Its score values come from 96 chronological practice points. Training used no backpropagation.

On two fresh 64-point courses the final controller carried shot values, motor and timing beliefs, and shot-risk estimates across all state changes. It won 56/96 scorable points after an own return versus 7/96 centered and saved 86/95 later balls in the strongest combined condition versus 94/101. All pooled criteria passed. Course A alone saved 44/50 later balls in that condition. The next milestone is to discover the paddle, ball and axis among distractors before testing transfer to Breakout.

Read the [progression report](2026-11-19-pong-transfer-progression.md) for the separate interventions, multi-state course, actual-game replay and limits. The [checkpoint and loading example](../../artifacts/m1b/README.md) and [transfer plan](../plans/2026-11-19-pong-skill-progression.md) identify what carries into the proposed Breakout experiment.

## Online button discovery qualified on November 19

Two independent learners each carried state through 128 games of normal, reassigned, differently reassigned, and restored controls. The visual button-effect controller passed all eight predefined late-opening and later-return checks. It returned 1,398/1,411 balls including every discovery loss. In the matched first remapping, learning returned 170/173 versus 3/35 frozen. One additional asymmetric-force test returned 129/131 versus 0/24 frozen. The model inherits its visual interception objective and learns the current button effects online without backpropagation.

At this earlier checkpoint the target was faster recovery when a change happens mid-rally. On 24 new matched switches the standard controller returned 8/24 next balls and a shorter-probe candidate returned 10/24. Both identified the new controls before that ball on 3/24 starts. Try acting on the first informative motion before completing a three-button calibration. Keep the qualified controller as the comparison and score the first incoming ball, simulated time, and ordinary-game reliability. Use Sol 6.2 XHigh for delegated research and independent challenge.

[Full results and game-by-game figure](2026-11-19-motor-discovery-results.md) · [Portable checkpoints and receipts](../../artifacts/m1b/README.md) · [Research plan](../plans/2026-11-19-replicated-motor-learning.md)

## Previous pause on November 5

All experiments stopped after the completed categorical motor-response pilot.
Read `2026-11-05-discovery-learning-pilot.md` for the result and next step.
Fresh online numerical learner: normal41/42, hidden three-action cycle78/82,
normal-restored30/33 (three initial losses then30/30). Acquired coefficient-frozen
control0/16 on matched cycle starts. All48 public replays, carried state boundaries,
source/assets and frozen controls checked clean. This is one promising development
stream, NOT independently qualified reliability or fullM1B.

Saved artifacts/checkpoints:
`/workspace/m1b-experiments/pong-categorical-motor-learning-20261105-r1`.
No qualification has been prepared or launched. Resume only on user instruction;
next is fresh replicated confirmation with fixed reliability/recovery criteria.
The active research plan is `../plans/2026-11-05-discovery-adjustment-learning.md`.
Historical completed scaffold and running notes below refer to older work.

## Latest completed scaffold — carried learning and motor adaptation passed

2026-10-30: two fresh chronological learners passed A→B→A with no learned-state
reset, no phase input and no backpropagation. Fixed last32 windows total
**1,063/1,071 returns**, first192/192; all384 live learning games including
exploration/recovery total1,719/1,815. One matched control pair: resetA11/43;
frozen acquired controller reversed0/32 and normal-again179/180. All544 planned
games completed; independent61source/assets and all receipt/tape/continuity audits
passed. Current process state: **no experiments running; stop at completed scaffold**.

Read `2026-10-30-carried-learning-and-adaptation-results.md` for the plot, exact
scope, controls and costs. Artifacts/checkpoints in
`/workspace/m1b-experiments/pong-carried-qualification-20261030-r1`.
The learner updates reward evidence during games and selects from16 inherited
feedback structures between games. This is not raw-pixel learning from scratch,
within-rally policy adaptation, arbitrary-game learning, winning or fullM1B.
Potential next target: learn numerical feedback/sensorimotor associations online
rather than only select existing structures. No next allocation is launched.
Historical running-task notes below are superseded.

## Active carried-learning work — 2026-10-30

User requests continued orchestration until carried learning/adaptation shown or
stop. See `../plans/2026-10-30-carried-learning-and-adaptation.md` for current
prospective decisions and exact interruption accounting. r2 pilot interrupted by
cloud restart at142/144 and137/144games; all279 completed receipts audited clean.
Both learned across games and adapted to reversal; capped-evidence false alarms
caused costly repeated exploration. Unchanged discounted learner selected for
fresh confirmation:2new streams64games/phase A→B→A, paired tail-only reset/frozen
controls onreplica0. Confirmation is now running in `/workspace/m1b-experiments/pong-carried-qualification-20261030-r1`,
spec SHA `fb058e6c8b71ee83e711b4c2bec766d5e1e423ff4227e04c25707cf8d578be4e`.
Fixed544games/544,000maximum transition cameras; live budgets2,760seconds each,
reset1,380seconds, frozen960seconds;4workers maximum. Both parent and independent
reviewer verified code/assets and384new distinct live starts plus matched controls.
Do not modify the pinned runner/learner or extend any started stream.
No pilot retry or silent extension. No milestone completion claimed yet.

Current source: `run_pong_carried_learning.py`,
`pong_carried_structure_learner.py`, `run_pong_carried_qualification.py`,
`report_pong_carried_learning.py` under `scripts/experimental`.
Pilot artifacts: `/workspace/m1b-experiments/pong-carried-learning-20261030-r2`;
`interrupted-report.json` audits source, reward tapes and carried phase boundaries.


## Latest result — longer frozen validation passed

User requested the longer evaluation after the clean pause. Completed all 128
fixed games in 11m23s: **1,436/1,437 returns**, including **1,075/1,076** in the
64 one-minute games. 63/64 long games reached the cap; late-bin returns were
354/354 across those 63 surviving games. Both independent endpoints passed their
predeclared first/later reliability targets. Source/assets stayed frozen.

See `2026-10-29-sustained-return-validation.md` and its committed JSON summary.
This establishes tested sustained frozen control, **not learning during games**
or full M1B completion. Next target: continuous online acquisition and retention
across games. No further experiment is running. Historical pause/status notes
below are superseded by this result.

## Paused at user request — 2026-10-29

Stopped cleanly after both independent reward-based acquisitions completed.
No experiment or agent is running. Final longer-play validation has NOT started;
its fresh cohorts remain unused. Resume only when the user requests it.

Both acquisitions selected `structure_04` independently on disjoint 16-game
training cohorts: **90 returns vs baseline 39** (initial random control 29), and
**90 vs baseline 44** (initial control 1). Both model, endpoint and summary hashes
were independently checked at pause. The selected model SHA is
`75530410f2c985a3037e9aa264447b8d172e5e9297a9e556bbb74d2c511dc7cc`.
Artifacts: `/workspace/m1b-experiments/pong-relational-categorical-replication-20261029-r1`;
`endpoints-frozen.json` records the completed freeze barrier.

Earlier fresh short-game validation achieved **179/180 returns**, including
32/32 first returns and 147/148 later returns. This is promising replicated
reward-based selection over a strong relational-controller prior and inherited
frozen vision, not learning from scratch or proof of long-match reliability.

Next on resumption: use the reviewed
`scripts/experimental/run_relational_replication_validation.py` to prepare and
pin the final spec, then run its fixed 128-game comparison (32 sixty-second and
32 twenty-second games per replica). Budget: 1,200 seconds, 256,000 transition
cameras, four workers. Inspect first/later returns and late time bins. Do not
refit, choose models from validation, or extend the fixed cohorts. The sustained
return milestone remains unconfirmed until this longer validation passes.
This pause supersedes the running-task descriptions below.

## Current position — learned feedback qualified; reliability still incomplete

**Newest result:** integrating the learned feedback with existing public-ID visual
memory, and using it from the opening, returned **180/180 TRAIN32 opportunities**:
32/32 first and 148/148 later returns; every game reached its 20-second limit.
Both new memory arms passed full serialized public-input/action replay (61,261
decisions). Current continuation artifact:
`/workspace/m1b-experiments/pong-relational-memory-20261029-r1/models/ungated.npz`,
SHA `75530410f2c985a3037e9aa264447b8d172e5e9297a9e556bbb74d2c511dc7cc`.
Frozen fresh63345 also passed: **179/180 returns**, first32/32 and later147/148;
31/32 games lasted20 seconds. All31,478 public-input decisions replayed exactly,
with source/assets frozen. Cost96.40 seconds. This is not yet long-match or
independent-acquisition qualification. `value_execution` owns a compact
streaming-audit worker and the final longer-play validation coordinator;
`learning_challenge` owns two new independent categorical acquisitions. Neither
acquisition nor final long validation is authorized to launch before parent source
review. See `docs/plans/2026-10-29-feedback-replication-next.md`. No further
gate/age/coefficient variations are selected. The historical result below explains
the learned binding and the preceding fresh improvement.

The milestone is **not met**, but the compact feedback learner has now produced
a substantial fresh-game improvement. Preserve `40069` as the historical baseline;
carry the new relational endpoint for further scaffold development.
The user has authorized continuing without further architecture approval until
reliable learned match-long returns are achieved or they stop work. Preserve the
no-backpropagation, camera-input and no-expert-label constraints.

The latest symbolic archive candidate improved TRAIN returns from 83 to 86, but
failed the frozen fresh comparison: **74 to 68 returns**, with first returns
unchanged at 26 and later returns falling from 48/69 to 42/64. No promotion.
Stop the symbolic grammar family; see the symbolic archive and fresh outcome notes.

The compact relational feedback search completed in
`/workspace/m1b-experiments/pong-relational-search-20261029-r1` under spec
`aa9b357ca0c2f354f1f078bcf41d25614024bd7cc25942d257dcd9da73c557c7`.
It learns anonymous object-pair binding, axis/polarity and five feedback
coefficients from actual own-hit rewards. Existing vision and the baseline opening
are retained. This is a stronger controller prior with a smaller search space,
not a new optimization algorithm. Fixed nested 4/8/16-game training cohorts avoid
the previous rotating-fitness problem; only complete 16-game fitness can replace
the incumbent. Total allocation: 900 seconds and 320,000 transition cameras,
including a final 32-game comparison reserve. Actual cost was 632.38 seconds,
231,307 transition cameras and 448 initial observations.

The selected controller learned choices `(3, 0, 1, 0, -1)`; all five numeric
coefficients remained zero. Its learned mechanism is position feedback between a
reward-selected anonymous visual pair. TRAIN32 improved **83 to 130 returns**,
with later returns **54/76 to 101/110**. Frozen fresh cohort 63313 improved
**90 to 131 returns**, with first returns unchanged at 30/32 and later returns
**60/84 to 101/113 (71.4% to 89.4%)**. Games reaching five returns rose from 8 to
18; the opponent missed no returns. All candidate decisions replayed exactly.
Sources/assets stayed frozen; the 32 fresh starts were independently checked
against all 86,815 declared exclusions. This is real improvement, not yet the
95% reliability target or independently replicated learning.

Endpoint `models/final-policy.npz` has SHA
`026ff8da73457238209708a03060bcaf10a5f94bbb83805d9f7a60c0644beedf`.
Next: compare the same learned controller from the first supported observation
against its current baseline-until-first-hit gate. Separately, a new persistent-ID
adapter will test using the existing vision system's carried entities during
brief observation gaps. No additional fits or coefficient sweep are selected.
Both changes require new runtime identities and fresh qualification; 63313 must
not be reused as an untouched evaluation cohort.

The new fitness runner uses an exact encoder-copy optimization. A paired native
proof reproduced every action, feature, encoded observation and runtime-state hash.
One timing pair suggests modest savings, not a general speedup percentage. Original
pinned runners are untouched. Helper commit `3e9081d`; search commits `7238313`
and `e144648`. The old unlaunched spec is preserved in the run directory.

Reused Sol 6.1 XHigh agents: `value_execution` owns execution and opening-gate comparison;
`learning_challenge` independently reviews the mechanism and next direction;
`runtime_efficiency` completed the exact-copy proof; `symbolic_control_challenge`
completed the negative fresh result. Parent owns synthesis, review and direction.
Older entries below are historical and must not override this current position.

Storage: two verified exact-byte hardlink passes preserve all artifact paths and
contents. The latest manifest is
`/workspace/m1b-experiments/storage-deduplication-20261029-r2.json`; it recovered
236,535,808 allocated bytes across 1,530 aliases, excluding active relational
artifacts. Treat completed artifacts as immutable. **Copy a linked file to a new
inode before editing it; never overwrite a shared artifact in place.** No SQLite
files were compacted and no unique experimental evidence was deleted.

## Active mandate

Continue orchestration until ball returns are learned consistently throughout a
match, or the user stops work. The goal is NOT met. No backpropagation, privileged
policy inputs or expert action labels. Latest user instruction APPROVES the tree
pilot and temporarily waives further architecture approval until this milestone
is complete or the user stops work. Parent review/bounded experiments continue. Bridge/viewer connectivity is
another thread's work. Earlier two-hour session limits are superseded.

Use parent direction/review and reused Sol6.1XHigh execution/independent challenge.
Previous workers unavailable at this turn (agent list only root); replacement
Sol6.1XHigh `/root/value_execution` owns approved tree implementation in MAIN.
Independent `/root/alternative_mechanisms` researches distinct next hypotheses.
Prior worker: `/root/rally_integration`, checkout
`/workspace/Fly-Connectome-m1b-pong-fast`. Reuse it via followup_task when idle.
Old `/root/pong_one_pass_research` became unresponsive and was parked.
Main checkout `/workspace/Fly-Connectome-m1b`, branch
`reconstruction/m1b-continuation`; Python `/workspace/Fly-Connectome/.venv/bin/python`.
Use `PYTHONPATH=src:scripts/experimental`, one BLAS/OpenMP thread and four native workers.

## Latest active update — October29

User has waived further architecture approval through this milestone; do not ask
again. No backpropagation, privileged inputs or expert action labels. Goal remains
95% first/later returns on fresh games with200+ later opportunities and independent
training replication. Current40069baseline remains primary;761secondary is unpromoted.

Completed pilots: fixed-treeFQI23→1; local option memory found2actual rescues but
23→19 deployed; past-reward context correction23→23. Original Ogma temporal/local
actor23→10, rejected. Its first changes all occurred on empty selected current
visual planes, while the baseline still had other camera context.

Guarded SAME Ogma weights also REJECTED on32games:82→74total,29→24first,
18→15second,53/76→50/67later. Source19c8bdf; recovery85f2b5b.
Screen8+1did not transfer. All16,441completed candidate rows replay exact;
64initial empty-prefix rows matchbaseline. Original180s interruption left17complete
receipts+166-row orphan. Separate120s recovery completed15missing in46.586s,
6167cameras+15initials. Saved work lowerbound16,607cameras+33initials PLUSunknown
interruptedwork. No fresh63185, no more temporalinput/Tvariants. Artifacts under
`pong-ogma-prior-20261028-r1/empty-current-correction/recovery-20261029-r1`.

Pairedforest9f282c6 REJECTED: all48sites acquired86.056s,28,139cameras+48initials;
16rescuepairs/12sites,19losthitpairs/15sites,109ties. Singlefit0.269s/exact2096row
sklearnparity. Actual8games23→9total,7firstunchanged,4→1second,16/21→2/9later;
114optionstarts/1656heldcams/1319changedactions.2590evalrows exact. Combined
30,729newcams+56initials; no32/fresh63201, no parametersweep. Source/forestfrozen.
Artifact `pong-paired-option-forest-20261029-r1`. Root requested<=3minoffline
mechanismreview fromvalue_execution, no newnative. Preserve compactN+1suffixdata.

Forestmechanismreview:18/21all-zero training sites stillpositive;37selectedholds
include12actualrescues+25ties.107/114deployedstartsrepeatwithin160cams(median16).
Root acceptedSol'schallenge: no fixedweights tailcorrection, stopforestfamily.

Active graph: `/root/learning_challenge` firstactualofflinefit converged13.56s,
1012iterations, residual9.966e-9/errorbound5.526e-6.27471eligibleobservations,
118hits,52naturalends;8censoredfinalsomitted.65capacityoverflows unsupported,
nohit/end evidence lost.20,080/22,233posthitrows baseline+alternative supported;
674/27301unknownsuccessorboundaries;13,343savedchoiceschanged (notperformance).
Graph4067f033ec767fa072ab4e9342d0d95b9f27b030cbcda536e16261fd7bdccb75 at
`pong-kernel-transition-graph-20261029-r1-recovery/graph.npz`.
Original60ssetupfailedbeforeanylearning(importpath,overflow); explicitseparate60s
recovery preservedoldreceipts/specs. Sevenfocusedtests+64queryserializationchecks.

Frozen graphnative69ea13a REJECTED23→10total,7firstsame,4→2second,16→3later.
2717cameras+8initials,10.913s;allchronologicalrows exact,1020changedactions,
1611/2020posthitrows supported. No32/fresh63217/radius/supportvariants.
Artifact `pong-kernel-graph-native-20261029-r1/summary.json`.

AdaptiveSMDPsource3d156b9 complete: physicalelapsedgroupinginvariant reward,
MCobserved-returnwarmstart+26FQIupdates168.313s, finalca99b349... selectedbyclock.
Notconverged; exact44,699training+2048serializedqueryparity. Heldmacros stopon
actualhit/naturalend/16; baselineoption1.17/32actualoptionleafpresenceguard,
independencediagnosticonly. Allsettingsfixedbeforefit; seeplan/resultsledger.

Initialnative201f517genericnewharness:23→12returns,7firstsame,4→2second,16→5later,
3150cams+8initials/18.591s. Allrowsauditexact. Onegain1→2andone5hitretained,
threebiglosses. No32/fresh63233/promotion. Artifact `pong-smdp-native-20261029-r1`.

Closed-loop C1 improvedMATCHED32 C0→C1:44→50total,29firstsame,7→13second,
15/42→21/49later. Baseline40069still82total/53later/18second. No63249fresh or
promotion; thisisTRAINadaptationonly. Comparison77.661s,21664newcams+56initials,
C0eightcachedreused; allaudits/frozenchecks passed. Artifact
`pong-smdp-cycle-comparison-20261029-r1/summary.json`.
Collectionall64complete71.734s,22876cams+64initials,86hits; exactmacroexploration.
C1refitdata65409usable/139starts;MC+16FQI166.942s(same180budget,C0had26updates).
Finale58a73f8ba9fe695aa7e1815fc268aaf43bef0759f8a65760fa58bcf8009f2cd;C0unchanged.
Exact65409+2048parity; notconverged. Olddataprefixes byteidentical. Seeledger/plan.

Second feedback cycle is complete and rejected; switching to learned dynamics/planning.
Collector2 used frozen C1 with unchanged exploration/options on 64 new TRAIN starts
(32 per phase), excluding all 142 prior/admitted starts. All games completed in
62.941s: 19,998 transition cameras +64 initials, 73 hits. These collection hit totals
are not paired performance because collection1 and collection2 use different starts.
All 56 new comparison traces (21,664 already-paid cameras) plus all 64 collector2
traces were appended; cached C0 eight were not duplicated. Prior array prefixes
and start IDs are byte-identical. Combined 100,286 usable windows across 206 starts.

C2 fitted in 168.054s, MC plus 12 completed FQI updates; endpoint
`785d35d48814633f253073f2e6f7804cb98b4293e99dadc632375ba2f4c7b43d`.
Exact query optimization preserves targets/support/scores; all 100,286 training
inputs and 2,048 serialized queries passed parity. More data still costs more:
C1 completed 16 updates, C2 12 in the same 180s limit. Neither is converged.
No settings, reward, exploration or action-duration changes. Sources c31de46.
Artifact: `pong-smdp-feedback2-refit-20261029-r1`; data spec d590d036b673bb099efe9c309cec799aceaa81ce97274d9109177718ae203a56.

Comparison2 (source a86d89c) used cached C1 games; only C2 played. Spec
92ae13da91aaf38de4a6d58e20ec7a85212492ee4f678ed99a93c7560c4f1481.
C2 failed the first-eight gate: C1 16→13 total, first7 unchanged, second5→2,
later9/15→6/13. Baseline still23total/16later. All3,270 rows audited exactly;
3,270 new transition cameras +8 initials in16.508s. Source/assets frozen.
No remaining24 or fresh63265 games; no promotion. Stop data-only SMDP cycles.

Learned-dynamics pilot completed and was rejected. Source18f09cb: ridge visual
changes +8 residual trees,16 event trees, H8/W8 primitive replanning, previous16
actual commands, frozenC1option3 tail. Prepared114,229 unique transitions from452
traces/206starts in25.04s; retained visibility/cardinality changes and actualprefixes.
One fit39.131s,1.47MB new model; model
171b9acabd3c4abc4b446b9f8c83a373bd36816aedfc8f96cebda80a075a189d.
Compiled exports2048rows/head and32 serialized beam plans exact; mean9.145ms/query.
Actual8 baseline23→11total, first7same, later16/21→4/11, second4same, max6→2.
2958transitioncameras+8initials in21.307s. AllrawQ/historyrows checked;402planner
queries replayed includingonecompletegame. Sources/assets frozen. No32/fresh63281.
Artifact `pong-delta-native-20261029-r1/summary.json`. Stop model/planner variants.

Symbolic search completed at literal zero: 749.693s,169402transitioncameras+
400initials,399completegames+1partial;94completechallengerfour-game evaluations,
11fullgenerations+partial12th. OwnTRAIN32bank baseline83total/29first/54of76later;
NOT the earlier82-returnbank. Localwinners failednewrotations andzerowasrestored.
Literalzero endpoint reusedexactcachedbaseline, noadditionalqualificationgames.
No fresh63297/no promotion. Spec fdc6b3b81e6398dca97f0c861c68fc2794fc5658c5e86803a16ff45a636ea949.
Resultnote `2026-10-29-symbolic-policy-outcome.md`, commitbac94e5.

CURRENT parallel tasks:
1. Posthoc frozenarchivequalification: root found exactlytwo previousincumbents
withnet+3hits across8actualmatchedfitnessseeds each, so lastrotationzero isnotproof
allarchivedpoliciesbad. a43dc09a175c...18vs15; f85f6d53fcda...16vs13.
`/root/symbolic_control_challenge` ownsapprovednewwrapper/source/spec
168d489cc9ff5a2e56f94fb5c9ea58ffedcc4a56e33464c62ebfc112da195acc,
artifact `pong-symbolic-archive-comparison-20261029-r1`. 300s/64ktransitioncap,
64newfullrawgamesinterleaved2policies/seed, sameown32bank, nofit/parameterchanges.
Budget300correctedBEFOREnativefrom180 usingmeasured226cams/s toavoidcensoring
best64×1000gameoutcomes. No fresh; parentallocatesifonequalifies. Rootreviewdone,
launchauthorized. This isseparateposthocselection, notchangedoriginalsearch.
2. Compactrelationalfeedbackservo implementation: `/root/learning_challenge` core,
`/root/value_execution` search. Five discretechoices n/i/j/axis/sign; five numeric
ci,cj,b,deadzone,hysteresis. Anonymousrelativeposition+velocityfeedback, rolesonly
selectedthroughactualreturns. Explicitgenericservo prior, notdiscoveredgeneral
control ornewoptimizer. Usefixednested4→8→16TRAINfitness withsameaccumulatedseeds;
stoprotating4singlechampion objective. Implementationonly, nolearningauthorizedyet.
3. `/root/runtime_efficiency` ONEactualzero-baselineprofilegame authorized≤1000
transitions+1initial at `pong-runtime-profile-20261029-r1`. Profilebeforeproposing
EXACTspeedups; noexistingpinnedsource edits. CurrentCPUquota4cores/16GBRAM.

One0.58s savedtraceworldplannerreview found stable-maskone-stepmotionreasonably
predicted, but8-step tailerror exceededchosenrootadvantages. Sol independently
advised NO-GO event-onlyH8variant: finaleventpredictiondoesnotproveearlyuseful
actionranking. Rootacceptedstop; preservelearnedmodelsforfuturefeatures/proposals,
notmoregreedyplannerparameter variants. Notesinlearned-delta-pilot.md(b2fcc33).
Userwaiverpersists; noarchitectureapprovalquestions. Continueuntilmilestoneorstop.
Bridge belongs toanotherthread.

Disk headroom: removed147MB completed unit-test temporaries, then hardlinked125
SHA-identical frozen transform copies, recovering1.267GB. All research paths/bytes,
checkpoints and receipts preserved. Manifest `storage-deduplication-20261029.json`
in experimentroot. Copy shared transforms before future in-place modifications.
Do not duplicate large metrics/checkpoints. Bridge belongs to another thread.

## Latest result / next action

Cloud connection recovered. Final broad-block fresh32 rejected bfb: 90→69
returns versus40069, first30 unchanged, later60/82→39/66, zero gains/eight losses.
Only six missing evaluation games were recovered in a separate13.575s extension;
original deadline expired, saved58 unchanged, unknown interrupted work preserved.
No promotion. TRAIN256's662→686 gain did not transfer. Final audit passed35,048fresh action/Q rows;14games changed actions,8lost
returns, with6losses first diverging after an earned hit. No identical hard-prefix
replay block. Reused rally_integration is implementing one bounded causal-burst
plus joint-output pass; see [plan](../plans/2026-10-28-causal-joint-return-learning.md).
Causal pass completed80.737s with9269cameras+11initials, no restartneeded.
One rewardedaction learned in5.33s; target2→3, but representative32total60→57,
first24same,second16same,later36/56→33/55. Candidate85b35d rejected; nofull/fresh.
TAG63057unused/reserved. Sourcec9295e1/main789648b;35focusedchecks passed.
User mandate continues. Full-coordinate actor completed786.97s,190265cameras+
384initials. FULL256662→566,first233→214,second161→144,later429/615→352/535.
Reject4a30; nofresh63073. Training-start greedy returns also151→141. Keep40069.

Smooth198-coordinate actor completed435.28s,114966cameras+250initials.
Screen128287→237returns,first106→91,second67→61,later181/262→146/222.
Both64strata lost25returns; rejectdc33740, keep40069. No fresh63089; no process
active. AllchecksPASS, no unknownwork. Actorfamily stopped aspredeclared.

Continuous residual pilot completed with a weak mixed signal, NOT a reliable
learner.14completed updates; calibration recovery kept the original clock and
reused all2048saved observations without replay. TRAIN32 total63→64,first24→22,
later39/59→42/59. Parent approved SAME frozenendpoint on32predeclared freshpairs
within original600s: total78→81,first30→28,second18→20,later48/72→53/76.
Paired mean-return difference .09375,95%bootstrap[-.625,.78125]. Preserve76126332
as a secondary research candidate, keep40069baseline; no automatic promotion.
Artifacts `pong-rally-continuous-actor-20261028-r1`; source5626846/df65c94/f5abc5d.
Parent firstbatchaudit2039rows exact Q/probability/eligibility/credit,0cameras.

Fresh Sol6.1XHigh `/root/fresh_learning_directions` independently researched
value learning, learned dynamics/planning, and temporal predictive hierarchies.
LSPI existing-runtime test completed4.613s then failed actual8games23→2returns.
Reject7195fd; no sweep. Artifact `pong-value-feasibility-20261028-r1`.
Root authorized new existing-policy epsilon.2 acquisition to address action
coverage BEFORE another valuelearner:32fixedTRAINstarts,16/phase,<=20kcameras,
120s ceiling; completed39.099s/11871cameras+32initials,49hits,23natural ends
and9quota partials. Complete N+1observations/rewards/timings, no model changes. Artifact
`pong-value-collection-20261028-r1`, spec
127f1c6c24fa5278938f5fe8b51d6eadbe8bf00ee4a9b1471987279b17e27c4b.

Next major proposal: [fixed-partition value learning](../plans/2026-10-28-fixed-partition-value-review.md).
User approved; implementation underway. User explicitly wants distinct parallel hypotheses,
outside evidence, and advancing/combining promising methods instead of exhausting
feature/actor variants. Further architecture approval temporarily waived; ordinary acquisition
and research remain authorized. No backprop, privileged inputs or expertlabels.
Goal remains unmet. Bridge belongs another thread.


## Current provisional model and actual evidence

Current practical research baseline is joint40069, not a fresh-qualified claim
of better later-return skill:
`/workspace/m1b-experiments/pong-rally-joint-output-20261028-r1/rbf-model.npz`
Bundle SHA `40069e2deae2b745b83b6c193389fa740450856b67443339b12f0dbb9bf3c438`;
W SHA `479472a37e9950246726cd90babf44f053467e3ae354783da2afbc35080ea15f`.
Frozen vision/map/transport, existing raw4389 plus fixed512pairwise outputs; both
output weight arrays learned without backprop. Feature assets unchanged.

* TRAIN192:493→523returns,174→175first,113→123games≥2,319/455→348/482later.
* Pooled existing64fresh (descriptive after interim decision): eb145→joint151
  returns,57→60first,39→42games≥2,later88/136→91/142. No convincing later-rate
  improvement. First/second coverage motivates a practical baseline, not success.
* Batch4c checkpoint retained as own-reward repertoire: TRAIN546, but fresh73
  versus joint86 on latest32, so not selected as the general-player replacement.
* Preserve eb at `pong-rally-pairwise-residual-20261028-r1/rbf-model.npz`, SHA
  `eb2cbcf10aaaf88ee5fd0877cbc966b828286e230a0782bd91c4dce9ddce9152`.
* Preserve 3ac at `pong-rally-earned-trace-merge-20261028-r1/proposal.npy`, SHA
  `3ac7f87fbac23abab14e633b38cfe1e50986a8f03bb38b049bed8f4cc840bd5e`.
* Preserve d7 at `pong-rally-guarded-curriculum-20261028-r2/selected.npy`, SHA
  `d7d7d872c1acde4b34748240c2ce9bf9d37af4c65ff2cd1f25d753f19d01fe61`.

All relative artifact paths are under `/workspace/m1b-experiments`. Current model
is a reset/branch-assisted own-hit-reward scaffold, not the final continuously
learning general agent. The original viewer export is older than these models.

## Active — joint existing-output learning

Root `pong-rally-joint-output-20261028-r1`, main7e53729. Learn both EXISTING
linear and fixed512 output weights, with frozen vision/features, no backprop.
Plan: [joint output learning](../plans/2026-10-28-joint-output-learning.md).
One60s fit verified74,623complete-earned-prefix rows in18.53s. Sourcefc's rewarded
201-row two-hit target4195433914 replaces its old one-hit prefix; all other earned
prefixes retained. Actual target1→4; representative32 87→93; full192493→523,
174→175first,113→123games≥2,later319/455→348/482. No perseed return losses.
Both output paths contribute; no necessity ablation performed.
New W SHA479472a37e9950246726cd90babf44f053467e3ae354783da2afbc35080ea15f;
bundle SHA40069e2deae2b745b83b6c193389fa740450856b67443339b12f0dbb9bf3c438.
Fresh32 TAG62993 interrupted at63games/30,923cameras. Parent recovered only the
missing carry game within ORIGINAL900s:5hits/1,000cameras/12.62s. All64receipts
complete:73→65returns,29→30first,21→19games≥2,later44/68→35/62.
No demonstrated transfer gain. All saved action/Q rows audited; all processes
exited. Native completion865.718s within original900s (fresh300substage exceeded
during interruption); documented≥105,587cameras+191initials, unsaved interrupted
work unknown. Joint40069 now practical research baseline as described above; oldeb/r5 preserved.

Completed diverse3-target batch `pong-rally-joint-batch-20261028-r1`:
agentdf9c09e/main3a5600a,27focused tests. One33.26sverifiedfit/80,839rows,
actualtargets1/1/2→3/6/6; full192523→546total,175→176first,123→129second,
later348/482→370/501. All192retained;11gains/no losses. But newfresh32:
eb72/28first/18second/44of68later;40069 86/30/23/56of80;candidate73/28/21/45of71.
No demonstrated fresh improvement. Batch4c0b preserved as TRAIN artifact, not
promoted. Parent/worker selected broader new TRAIN experience next. Implement minimal
manifest collector:64newstarts32/phase, fixed40069and4c,128games/600s,
no fitting yet. Sourcea3c2aee/main41d1ce6 reviewed;29focusedtests passed.
Collector launch authorized at `pong-rally-new-TRAIN-repertoire-20261028-r1`,
manifest at sibling `.manifest.json`, ROOT1943906127/TAG63025. Collector completed209.534s/600,63,910cameras+128initials,128complete,
all saved action/Q/source/raw checks PASS. New64:40069 139/58first/38second,
4c140/58/38;14better4c,16worse, same6firstmisses. Existingfresh remains out.
Plan: [broader own-reward experience](../plans/2026-10-28-broader-earned-experience.md).
Parent approved admission of64auditedscaffold40069receipts into uniformTRAIN256:
662returns/233first/161second/429of615later;100,069earnedrows/233cases.
Implement bounded3-updateblock,≤3targets/update,90sfit each,one1200sclock,
broaderGsameformulaonce.12compatiblepool; first1289501189L1→2,58405525R1→2,
3654232341L2→3;100,586initialhardrows. Sourcebeb99fc/maina8a0a84 reviewed,32focusedtests PASS; launched
`pong-rally-joint-learning-block-20261028-r1`, session98406. Single1200sclock,
implementation900sseparate; spec5759fd6f36eb4c536a6fcae7c9c4a09ec41433aca8130ada76e37fb83f32cfd3.
FreshTAG63041 reservedpre-fit, uniformnewbankr7. No activecodechanges.
Exact completeprefix certificates can preserve known-return LOWERBOUNDS between
fits without fullsuffixreplay; mixedsourceprovenance required, currenttargetactual
outcomes must be measured before replacement, full/freshqualification atblockend.
Use fixed8correctnesssentinel incl104fivehit, finalfull256/fresh32vs40069once.
Stopupdatesonfailedfit; preserve lastverifiedcandidate. No pruning/overrides now.
All3updates accepted by218.70s: fits20.62/27.59/28.55s verified100586/101837/103318rows.
Actualtargetgroups4→10,4→12,7→10; controls28→28/28/29. MemoryLOWERBOUNDS
662→668→676→680,233first/165second/447later; notfullnativecounts.
Finalmodelbfb0d2824c7eb943434f31ff5772bb1ceee48add957220bd0374f6c83e84d1f7,
Wf90abda40c83aa4a9fa9df25efdbf04db66df6f13c03519cb857f828def048a6.
Full256complete:662→686returns,233first,161→167second,429/615→453/636later;
13gains/0losses, all earned minima retained. Cloud outage left58/64freshgames
saved and no activeprocess. Recovery authorized ONLY missing frozen games
under separate180s allocation if originaldeadline expired; do not resetclock
or restarttraining. Reused rally_integration recovering; no activecodechanges.
No more blind small oldreplayfits.
No fresh62993or63009admission. Both r5eb and r6TRAIN-scaffold banks preserved.
Batch589.36s/900,160,146cameras+271initials, no interruption/unknown work.
Bundle4c0b6374f3e35fad8260d5bd07161f780383de99a8c536b67437169d12200811;
W3f2e4958de960c32d0ba30b61d25449e1d60d1344fe9c5bf669b12d08f1419aa.
All runs exited. Plan/results: [joint batch learning](../plans/2026-10-28-joint-batch-learning.md).
Goal remains reliable match-long returns, not merely passing a fitted constraint.

Current TRAIN bank `pong-rally-TRAIN-bank-20261028-r5/admission.json` has192games:
493returns/174first/113games≥2/319later of455. Earlier32fresh candidate receipts
were explicitly admitted after evaluation; subsequent fresh sets remain evaluation
only. Preserve old banks/checkpoints and reserved final streams.

Recent stopped routes (details in ledger):
* Residual-only historical replay improved TRAIN493→499 but fresh80→78 and
  forgot original104562292five-hit behavior5→2; unpromoted ef890acc.
* Eight projected random requests all failed bounded projection verification;
  zero native games, not evidence that their actual rewards failed.
* The SAME eight raw requests yielded two strong8game screens, but chosenfc
  failed full192493→404. Existing runnerup1+ failed32games92→59. Search family
  stopped; useful individual rewarded fc trajectories were retained as experience.
* Joint learning combines existing output paths to integrate that experience
  while retaining complete earned memory; no new architecture was introduced.

Detailed results: [ledger](2026-10-27-sustained-return-results.md).
Bridge remains another thread's work. Main through2f9ddc6 pushed; a8a0a84 and currentresults pending nextpush.
Disk has about1.5GB free; reuse the192MB metric.

## Completed block and failed routes

Four new burst recoveries were found on current-policy misses, each+1actual next
return (`pong-rally-new-bursts-20261028-r1`). Integration remains the difficulty.
Hard complete-sequence preservation did not verify within bounded fits. Full
empirical score covariance made fitting fast but harmed actual retained behavior:
target-only sentinel61→58; three selected guards improved sentinel61→68 yet
full409→391. Protecting every known successful opening gave full409→403,
143→144first but266→259later. All rejected. Do not resume blind guard growth.

User approved ONE fixed64-RBF residual comparison, now completed:
`pong-rally-local-rbf-20261028-r1`.2,048TRAIN samples/160episodes; exact zero-output
parity on92,162saved rows. Effective centered rank1.208, broad activations. An initial
heuristic screen stopped after4.81s; parent and Sol explicitly relaxed only that
heuristic and ran one fit on the exact saved basis/data/objective under the ORIGINAL
600s deadline. Fit6.18s: no verified candidate, solver-reported primal infeasibility
(approximate certificate, not exact impossibility proof). ZERO native cameras,
no new model, no bandwidth retry. No further RBF sweep is authorized.
Source agent7bb1847/191bde6 integrated main1baf03a/2ad26e7; eleven focused tests pass.
Implementation416.4s recorded separately. Outcome docs pushed in main12a18b1.

One passive vision diagnostic suggested short-history velocity estimates reduce
noise in a selected miss; another miss had accurate inferred ball tracking.
Neither establishes global vision correctness or improved control. Smoothing can
lag rebounds and change the head's input distribution. These motivate the active
bounded native pilot, not a declaration that vision is solved.

## Durable references

* [General scaffold plan](../plans/2026-10-21-continuous-learning-scaffolds.md)
* [Sustained-return plan](../plans/2026-10-27-sustained-return-learning.md)
* [Detailed results ledger](2026-10-27-sustained-return-results.md)
* [Completed RBF proposal and amendment](../plans/2026-10-28-local-nonlinear-residual-review.md)
* [Preserved full historical handoff](archive/HANDOFF-through-2026-10-28.md)

Report actual native outcomes separately from rewarded exploration, fitted actions
and selected checkpoints. Preserve prior checkpoints and charge all camera work.
Use exact deployed-policy episode certificates where valid; avoid re-running
unchanged games. Reserved final streams remain reserved. Do not declare M1B
complete merely because a Pong scaffold improves.
