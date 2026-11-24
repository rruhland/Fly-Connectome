# Full-size physical rally teacher

## Coupled speed–torque qualification

Two exact-state coaches each completed **100 alternating legal returns** on the regulation court under the same signed speed-dependent motor cap used for candidate simulation. Each point started from measured DeepMind launch 10542 and continued using only native contacts, servo commands, and the carriage's physical forces. The carriage varied among six forehand targets. Both command replays matched their full live-state trajectory digests. [Reward-gated recall receipt](rally-physics-envelope-100.json.gz) · [Fresh-search receipt](rally-physics-envelope-100-fresh-search.json.gz)

| Coach | Simulated point duration | Wall time | Peak joint rate | Minimum CAD clearance |
| --- | ---: | ---: | ---: | ---: |
| Recall with a fresh search when needed | 74.33 s | 152.58 s | 3.516 rad/s | 18.17 mm |
| Fresh search at each reception | 72.42 s | 849.77 s | 3.309 rad/s | 18.59 mm |

The recall coach began with empty memory and stored 50 movements after genuine legal landings. It selected 48 retrieved schedules and three fresh searches, including the plan in progress at the cutoff. It checked reused schedules with the coupled dynamics and used a broad playable-landing objective. The elapsed times include shared-machine conditions during the respective runs. The median three-dimensional departure speed was 4.53 m/s from the near blade and 5.65 m/s from the far blade, with a median 0.744 seconds between contacts. Replay found zero geometric penetrations across 35,680 solver samples. [Rewarded action and referee history](rally-physics-envelope-100.json.gz)

## Independent speed and torque screens

The simulated SO-101 and a force-driven carriage completed **100 consecutive alternating legal returns** in 70.25 simulated seconds. The point began with measured launch 10542 from the pinned DeepMind recordings. The launch was excluded from the total, and the point was still live at the measurement cutoff. The carriage varied its target across six forehand positions and pace settings. Every counted return came from a native racket contact and a legal landing across the net. [Rated-motion replay receipt](rally-physics-rated-100.json.gz)

The coach used exact ball state, numerical inverse kinematics, and isolated MuJoCo candidate trajectories. On the live arm it issued signed servo target rates with a 4.16 rad/s limit for the five stroke joints and a 3.35 N·m torque cap. It held each screened follow-through before planning the next reception. Replay reproduced the live position–velocity trajectory hash and every referee event. Across 33,720 solver samples, the closest audited arm geometry cleared by 18.59 mm. The largest sampled joint rate was 5.445 rad/s, below the STS3215's nominal 5.454 rad/s no-load reference. These independent limits do not model available motor torque as a function of speed. [Motor and geometry fields in the rated-motion receipt](rally-physics-rated-100.json.gz)

| Measurement | 10th percentile | Median | 90th percentile |
| --- | ---: | ---: | ---: |
| SO-101 ball departure, m/s | 4.91 | 5.23 | 5.65 |
| Carriage ball departure, m/s | 5.65 | 5.99 | 6.25 |
| Time between contacts, s | 0.617 | 0.677 | 0.819 |
| Pinned DeepMind launch speed, m/s | 4.22 | 5.50 | 7.18 |

Departure is the three-dimensional ball speed at racket separation. The rated-motion run took 546 wall-clock seconds including search and control, about 7.8 times the simulated point duration. On the earlier full-slew run, the near-side search took 228 of 376 wall-clock seconds across 51 calls. It ran 13,080 native trials from a generated pool of 42,600 candidates, with a median of 4.28 seconds per search call. The point pace therefore characterizes simulated play while the search timing characterizes an offline coach. [Pinned launch-speed reference](../tt25/recorded-secondary-reference.json)

The regulation fixture supported two more 100-return replays under independent motion screens. A 4 mm candidate CAD margin held at least 4.71 mm measured clearance through the forehand point, with a 6.34 rad/s peak actual rate. The physically fitted side-on grip on measured launch 3728 held at least 12.69 mm clearance through 36,504 solver samples, with a 6.47 rad/s peak rate. Its point lasted 76.05 simulated seconds. [Forehand margin receipt](rally-physics-margin-100.json.gz) · [Side-on grip receipt](rally-physics-side-on-100.json.gz)

The full-slew 10542 video preserves the underlying 100-return simulation and its pace. That run peaked at 7.71 rad/s and replay found two samples of racket–table overlap above the 3 mm comparison tolerance, worst 4.95 mm. An earlier 41-return tucked-arm test intersected the robot's own geometry and motivated the laterally shifted forehand interception. [30 FPS real-time replay](rally-physics-teacher.mp4) · [Full-slew diagnostic receipt](rally-physics-100.json.gz) · [Folded-posture diagnostic](rally-physics-folded-diagnostic.json.gz)

## Speed–torque sensitivity

A linear signed motor model used the compiled 3.35 N·m stall cap and the 5.454 rad/s no-load reference. It estimated normalized voltage as signed torque divided by stall torque plus signed speed divided by no-load speed. Torque and speed in the same direction consume voltage headroom. Opposing torque during braking retains the stall-current cap. This is an optimistic sensitivity model for simultaneous demand with no under-load motor measurements. [Signed motor audit receipt](rally-physics-torque-envelope.json)

Replaying the rated-motion tape under the original plant showed that 35.4% of motoring joint samples and 66.4% of positive mechanical work exceeded the linear speed-dependent bound. Its braking samples stayed inside the envelope. Applying the tighter torque cap at every step to the same fixed actions ended the rally after two legal returns. [Original and torque-derated replays](rally-physics-torque-envelope.json)

A separate coach planned with the same linear torque cap used by its live arm. It reached a prespecified **12-return cutoff** on launch 10542 in 9.2 simulated seconds. This was a continuous point with signed servo commands. Replay reproduced the full trajectory and referee events. It found zero CAD penetrations and zero violations among 22,080 signed speed–torque samples. The selected solver trials also reproduced the six live near-side contact times and landing centers. This shows that fresh planning can play through the capped drive model on the measured launch. The later endurance trials above extended this result to 100 returns. [Matched torque-envelope replay](rally-physics-envelope-12.json.gz)

The same readout on the verified 7271 stereo pupil trace counted 100 legal returns and 59.1% of motoring joint samples outside the linear envelope. The excess accounted for 66.8% of positive mechanical work. Its braking samples stayed inside the model. [Signed camera-action replay audit](rally-physics-torque-envelope.json)
