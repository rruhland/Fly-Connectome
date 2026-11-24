# Learning from the direction of a shot error

The arm pairs the selected motion's simulated outcome with the outgoing flight it sees through its two fixed RGB cameras. It triangulates the ball, fits its flight before the next impact, and estimates where it would cross the net and reach table height. A net hit ends the fit before the deflection, so a shot below the net still supplies a signed error. Each measurable shot adds the difference in lateral landing, landing depth and net clearance to a 64-shot memory. The planner scores verified candidates using a weighted average of errors from past shots that had similar incoming flights and predicted net clearances. [Optical updates](../../scripts/experimental/tt29_optical_outcomes.py) · [Live controller](../../scripts/experimental/tt29_action_memory.py)

Both controllers collect the same type of optical evidence. `--optical-outcomes record` keeps the nominal costs, and `--optical-outcomes learn` applies the signed corrections online. Both start with the same rewarded movement archive and continue adding successful movements. The two arms in each comparison use identical measured openings, opponent seeds, motor envelopes and cameras. The independent wide comparison changes the opponent schedule and starts with an empty optical memory. Scores below exclude the measured opening launch. [Point results, optical measurements and commands](optical-outcomes.json)

| Matched course | Alternating returns, record → learn | Arm returns | CPU seconds per alternating return |
| --- | --- | --- | --- |
| Wide opponent, original schedule | 22/2/8/6/4/4 → 30/10/8/2/4/32 | 23 → 43 | 2.79 → 2.88 |
| Combined changes in materials, arm load, camera pose, command and image lag, and command noise | 4/2/3/2/6/2 → 4/2/6/2/16/6 | 10 → 18 | 3.04 → 4.08 |
| Wide opponent, independent schedule | 4/0/4/12/22/4 → 4/0/4/6/24/4 | 23 → 21 | 4.22 → 5.16 |

On the original wide course, the first changed command was tick 431 of the first point. The optical learner reached the 32-return cap on the sixth point. The baseline reached a maximum of 22 returns. The record-mode trajectories reproduced all four action tapes in the earlier wide assay. For the 21 clean landings available to audit in the record course, camera-estimated placement had a mean absolute error of 1.5 mm laterally and 11.2 mm in depth. Referee events were used for this offline check. [Earlier wide-target assay](envelope-variety.json.gz) · [Matched results and landing audit](optical-outcomes.json)

The independent wide schedule reached a 24-return point with optical learning, at the course cap. Its overall score decreased from 46 to 42. Both controllers issued identical actions on the first three points, and the first changed point fell from 12 to six returns. The optical model requires an observable outgoing segment, so a ball that passes the paddle supplies no signed placement update. [Independent schedule and action hashes](optical-outcomes.json)

The two final gains in the combined condition raised the longest point from six to 16 alternating returns. The controller that used signed optical errors took 146.7 CPU seconds across all six points, compared with 57.7 for the record control. The camera fit put its first loss 6.5 cm below the net-clearance threshold. Both controllers received the same body fit, camera registration and two-tick encoder timing estimate from solo practice. Aligning shot timestamps to the private planner clock moved the median audit difference from 29 ms late to 4 ms early. All 12 action tapes and camera estimates reproduced exactly after the clock change. [Compound controls and calibration](tt29_outcome_feedback.md) · [Timing recheck](optical-outcomes.json)

The six points in the original wide record course supplied a separate bank of 24 measurements. It can be loaded with `--optical-checkpoint artifacts/tt29/optical-outcome-primer.json` and held fixed with `--freeze-optical-outcomes`. The record mode produces a checkpoint after every point, so a run can supply a bank without applying the corrections during its collection. [Frozen bank and donor course](optical-outcome-primer.json)

## Short and sideways deflections

The independent wide record course had three contacted misses with too little outward motion for a full placement estimate. A stereo replay reproduced every referee event and exposed one short outward descent and two lateral deflections. Each gave an estimated upper bound on forward reach at least 1.09 m short of the net. The additional method uses either a sustained outward descent near table height or a visible velocity change near the planned strike. It abstained for both legal returns that lacked a full placement estimate and for the observed attempt that had no contact. [Stereo samples and offline contact audit](optical-censored-reach.json)

The optional `--censored-optical-outcomes` mode uses these partial observations to raise the cost of later verified movements with similar trajectories and insufficient predicted forward reach. Each sample provides an upper bound on the depth error. The replay informed the detector design, so the same schedule serves as a development comparison for this extension. [Live option and bounded scoring](../../scripts/experimental/tt29_optical_outcomes.py)

| Mode on the development schedule | Alternating returns in order | Arm returns | CPU seconds per alternating return |
| --- | --- | --- | --- |
| Record | 4/0/4/12/22/4 | 23 | 4.22 |
| Learning from full flight estimates | 4/0/4/6/24/4 | 21 | 5.16 |
| Learning from full estimates and forward reach | 4/0/4/24/16/22 | 35 | 4.70 |

The arm observed one partial shot online, on the second point. Its first three action tapes matched those of the full-estimate learner. The first action change came at tick 89 on the fourth point, which reached the 24-return cap. Total CPU time rose from 216.8 to 328.8 seconds as the full course grew from 42 to 70 alternating returns. [Development runs and observation timing](optical-censored-reach.json)
