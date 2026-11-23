# How much bouncing practice carries into a regulation game?

[Practice and transfer curves](bounce-practice.png) · [Watch the wrist-camera and external replay of the same held drop](wrist-bouncing.mp4)

The same simulated SO-101 practiced with either a twisted handle or a canted paddle. Each course used one continuing stream of wrist-camera-controlled solo drops, with checkpoints before practice and after 4, 16, 64 and 192 drops. Inputs during each drop were wrist RGB and encoders. A calibrated stationary camera pair observed the flight before and after the first visible impact and helped estimate the grip correction between drops. The scalar rebound reward updated a second parameter controlling how much the wrist policy could move the paddle center. Both tasks started with previously trained visual policies. At the table only the rotation estimate transfers between the two policies. No experiment in this pass used backpropagation. [Source and target protocols](dose.md)

The solo task itself showed a substantial response early. Across 24 matched fresh drops, the physical referee counted 21 rebounds before the new practice, 216 after four drops, 136 after 16, 126 after 64, and 175 after 192. The two 192-drop courses together used 48,666 control ticks and produced 2,071 qualified practice rebounds. On a separate sequence of regulation practice shots, a receiver at the 16-drop checkpoint made 44/96 legal returns on fresh variations after 56 target trials per mount. The 192-drop checkpoint made 45/96. Their frozen counterparts achieved exactly the same held totals. This experiment transfers a blade calibration between existing policies. It offers little evidence of learning a general rally strategy. [Matched solo trials, target learning and costs](dose.md)

## Receiving before any new table practice

| Paired receiving test | Starts | 0 drops | 4 | 16 | 64 | 192 | Source-only stop |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| New variations of practiced shots | 96 | 29 | 30 | 32 | 32 | 30 | 31 |
| Nearby original recordings | 152 | 2 | 5 | 5 | 4 | 5 | 4 |
| Unrestricted original recordings | 192 | 1 | 1 | 1 | 1 | 1 | 1 |

The custom stopping rule ended after seven drops on both mounts. Source-only stop uses that checkpoint without seeing any target reward. Starts here count the same launch once per tool. On the unrestricted cohort, going from four to 192 drops gained 0 previously missed returns and lost 0 previously successful ones. [Paired outcomes with donor-clustered uncertainty](assessment.json) · [Switch on the handle](switch-handle.json.gz) · [Switch on the canted tool](switch-canted.json.gz)

## Does the next ball come back in a game?

Both arms played each point to completion on the regulation table. Each checkpoint had the same 27 launches per grip: seven originals whose opening had previously been returned, eight fresh variations of practiced openings, and 12 unrestricted recordings. The near arm received RGB from two fixed cameras, its own commands and encoders. Both its opening and its reply controller received only the grip estimate learned in solo practice. The opponent independently chose its replies using simulator state. All 162 points reproduced their referee events when both physical arms’ command tapes were replayed. [Full-game runner](../../scripts/experimental/tt28_live_dose.py) · [Handle at 192 drops](live-handle-192.json.gz) · [Canted tool at 192 drops](live-canted-192.json.gz)

| Across the same 54 full points | No solo practice | 16 drops | 192 drops |
| --- | ---: | ---: | ---: |
| Openings returned, all cohorts | 15 | 20 | 20 |
| Opponent sent the opening back legally | 7 | 16 | 16 |
| Arm legally returned the opponent's reply | 1 | 2 | 1 |
| Openings returned in the 24 unrestricted starts | 0 | 1 | 1 |
| Longest exchange, both arms' legal returns | 4 | 4 | 4 |

The 16-drop correction helped the opponent get playable balls. Increasing solo exposure twelvefold left the arm with one second return on those same 16 reply opportunities. That is still a long way from consistent 10–20-return rallies.

## Choosing when to switch tasks

A second rule made the switch using only the visible source impacts, before any receiving results. After at least six observed impacts it resampled the impact directions 400 times. It required the 90th percentile angular deviation of the resampled median to fall below 1.5°, and the last three observations' median to lie within 2° of the cumulative estimate. Two consecutive new impacts had to pass. The thresholds were fixed before evaluating the selected checkpoint. This tests convergence of the measured grip orientation on two stationary mounts. [Stopping rule and physical comparison](../../scripts/experimental/tt28_when_to_switch.py)

The equal-budget test allocated 192 episodes among solo practice and receiving, then froze each receiver. Half the table-practice shots were variations of previously demonstrated openings and half were variations of a separate set of measured trajectories. Online table practice rewarded only legal landings and used seven available joint adjustments. The held assessment tested 48 new variations of the original demonstrations, 38 different nearby recordings and 38 unrestricted recordings per mount. Training episodes lasted different lengths, so this comparison equalizes the number of episodes. [Experiment](../../scripts/experimental/tt28_practice_budget.py)

| Allocation, solo + regulation practice episodes | Held variations of practiced openings, out of 96 | Held nearby recordings, out of 76 | Held unrestricted recordings, out of 76 |
| --- | ---: | ---: | ---: |
| 0 + 192 | 29 | 3 | 0 |
| 16 + 176 | 36 | 2 | 0 |
| 64 + 128 | 41 | 3 | 0 |
| 192 + 0 | 37 | 3 | 0 |

The intermediate allocation had the highest score on variations of practiced shots. Its advantage over all-table practice was concentrated on the twisted handle (26 versus 11). On the canted tool that comparison was 15 versus 18. The independent recorded trajectories provide no evidence that this practice allocation solved general receiving. [Handle mixed](budget-handle-mixed.json.gz) · [Canted mixed](budget-canted-mixed.json.gz) · [Handle solo only](budget-handle-long.json.gz) · [Canted solo only](budget-canted-long.json.gz)

## What changes when a toss arrives from the side?

Three additional practice streams exercised the same inherited wrist policy. Each tried a new small five-joint displacement on a fresh physical drop. The fixed cameras and encoders estimated incoming and outgoing velocity at the first optical reversal in the gripper frame. A regularized local fit estimated which joint combinations changed the outgoing velocity most strongly after accounting for the incoming one. This experiment used the first 32 ticks of each release to study individual impacts. The 192-drop courses above separately measured sustained bouncing. [Solo experiments and fitted directions](../../scripts/experimental/tt28_toss_prior.py)

| Training drop | Releases | Visible first reversals | Median horizontal incoming speed at a detected impact |
| --- | ---: | ---: | ---: |
| Ordinary drop | 96 | 90 | 0.04 m/s |
| Oblique toss | 96 | 87 | 0.61 m/s |
| Increasing toss speed | 96 | 62 | 1.37 m/s |
| Successful historical regulation openings | 19 | 19 visible approaches before the demonstrated contact | 4.18 m/s |

In the increasing-speed course, first-reversal detection across four successive blocks of 24 was 23, 23, 11 and 5. Only two rebounds passed the physical solo referee’s scoring rule in the fastest block. The launcher progressively matched more of the regulation speed distribution while providing fewer examples of a controllable rebound. Recorded regulation speeds were fitted from the final five valid precontact stereo samples. These 19 trajectories were selected by earlier demonstration success. [Source speed distributions](motion-range.json) · [Graduated tosses and motor tapes](toss-staged.json.gz)

Regulation performance was tested by giving each branch one recorded opening and seven new variations from the same donor, followed by four further variations with the correction frozen. After a legal return the learner retained the adjustment and applied it to the posture selected from the next ball's first four stereo frames. Otherwise it explored positive and negative directions in order. Controls included fixed camera-selected goals, three arbitrary orthogonal directions, manually chosen wrist/arm axes, and a source model fitted after shuffling the motor commands relative to their effects. The held courses were run separately on different original recordings. A further scaffold gave all branches an earlier known contact posture on 16 donors whose 56 stored actions had all failed to make a legal return. These 16 are reported separately. [Transfer implementation](../../scripts/experimental/tt28_toss_transfer.py)

| Held legal returns on new variations after eight regulation practice shots per donor | First roster, 152 feeds | Independent roster, 152 feeds |
| --- | ---: | ---: |
| Frozen receiving policy | 4 | 14 |
| Three arbitrary orthogonal motor directions | 9 | 13 |
| Hand-selected wrist/arm axes | 8 | 15 |
| Directions fitted on 24 ordinary / 96 ordinary drops | 7 / 10 | 14 / 13 |
| Directions fitted on 24 oblique / 96 oblique drops | 8 / 10 | 14 / 15 |
| Directions fitted on 96 progressively faster tosses | 10 | 14 |
| Directions after shuffling the 96 oblique motor labels | 6 | 13 |

The first roster suggested exploration could rescue some new trajectories. Its gain over frozen largely disappeared on the next 38 recorded donors. Across the 16 historical contact-only donors, the 96 ordinary, oblique and increasingly fast practices yielded 4, 5 and 1 legal returns out of 64 new variations respectively, versus 0 frozen. This experiment did not establish a transferable advantage from wider toss directions. [First roster](toss-target.json.gz) · [Independent roster](toss-target-confirm.json.gz) · [Increasing-speed confirmation](toss-target-staged-confirm.json.gz)

## Outgoing flight and the next ball

A separate test addressed what the arm can learn after making contact. Every branch began with an earlier known contact posture for the current donor. Reading the first three outward camera positions after a wrist adjustment provided a rough landing estimate even when the physical referee gave zero reward. The strongest gain was with three practice probes on a blade rotated in one direction. Additional probes let the reward-only controller catch up. The test used a stationary opponent after the feed. Related robot work by Tobuschat and colleagues measured how changes in the arm at interception affected ball placement. Their reported two-to-five iterations come from a robot already equipped to intercept the ball. Here, the separate unfiltered regulation assessments also count failures to reach it. [Full held comparison](outgoing.md) · [Tobuschat et al.](https://arxiv.org/abs/2308.14562)

We also physically combined the skills. Twelve earlier contact-supported donors each contributed six new trajectories under both mounts. After 16 solo drops, the learner used its observed blade correction to choose the starting posture. It then tried three target shots before the six fresh ones. The combined learner using outgoing flight landed 17/144, equal to 17 using outgoing flight alone. The solo checkpoint alone landed 16, the solo checkpoint plus score-based updates landed 18, and the unadjusted receiver landed 14. The handle improved from 2/72 unadjusted to 9/72 combined, while the canted tool fell from 12/72 to 8/72. The fresh games here evaluated placement after coached contact. They do not establish broad rally transfer. [Both mounts, gains and losses, and replays](combined.md)

## Reproduction

Use the pinned regulation launch dataset and the project’s MuJoCo environment. From the repository root, set `MUJOCO_GL=egl`, `PYTHONPATH=.:src:scripts/experimental`, `LP_NUM_THREADS=1`, and `OPENBLAS_NUM_THREADS=1`. The solo courses take enough time to be worth reusing. The checked-in checkpoints contain the fitted parameters. [Source and target commands](dose.md#reproduction) · [Launch dataset](../tt25/recorded-benchmarks.md)

```bash
for kind in ordinary oblique staged; do
    python -m scripts.experimental.tt28_toss_prior "$kind"
done
python -m scripts.experimental.tt28_motion_range
for option in "" "--confirmation"; do
    python -m scripts.experimental.tt28_toss_transfer /path/to/rallies.json $option
    python -m scripts.experimental.tt28_toss_transfer /path/to/rallies.json --staged $option
done
for tool in handle canted; do
    python -m scripts.experimental.tt28_when_to_switch /path/to/rallies.json "$tool"
    python -m scripts.experimental.tt28_practice_budget /path/to/rallies.json "$tool"
    python -m scripts.experimental.tt28_practice_budget /path/to/rallies.json "$tool" --long-only
    for dose in 0 16 192; do
        python -m scripts.experimental.tt28_live_dose /path/to/rallies.json "$tool" "$dose"
    done
done
python -m scripts.experimental.tt28_audit
python -m scripts.experimental.tt28_dose_figure
python -m scripts.experimental.tt28_bounce_replay
```
