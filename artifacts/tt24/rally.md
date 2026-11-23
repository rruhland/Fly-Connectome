# Direct joint feedback in a physical rally

The pupil reads the ball silhouette in two live, fixed RGB cameras. It combines each visible center with recent image motion, missing-sighting history, measured arm joints and its own issued joint rates. A weighted least-squares fit over nearby demonstrations estimates the next actuator position from the current observation. The pupil converts the estimate to one frame of normalized joint rates and runs the same calculation on the next observation. All five arm joints participate. The sixth servo keeps its grip. The fitted lookup operates on individual observation/command pairs. It has no episode index or strike clock, and it reads no ball state, contact forecast, future command or referee signal during play. [Feedback law](../../scripts/experimental/tt24_rally_policy.py)

The demonstration bank reconstructs native RGB and proprioception by reproducing twelve archived TT23 teacher tapes from five release seeds. Demonstration suffixes end immediately after the teacher's last own legal return. The archive's teacher owns those trajectories and their physical planning. Its actions are fit labels only. For each shared-opening assessment release, every condition receives the same 25 teacher commands. The first contact is made by that prefix, and the referee usually credits its legal landing after the handoff. The live far partner is the same physical learned coach throughout, including after the pupil changes the ball trajectory. [Teacher archive](../tt23/tt23_sequence_coach.md) · [Replay and scoring](../../scripts/experimental/tt24_rally_trial.py)

## Held continuation with a shared legal opening

Nine archived launch seeds were withheld from the pupil demonstrations. These archived releases were selected for a usable teacher opening and far reply. The table counts all nine in each condition. A second near return means the pupil earned one additional legal landing after the supplied opening. The visual mismatch delays all visual channels by six frames while leaving the live proprioception and issued-command history current. The zero-demonstration fit returns an identically zero command, which makes it equivalent to the separately implemented hold control.

| Condition | Demonstration releases, tapes | Teacher frames, including prefixes | Continuation fit samples | Second near return | Fifth near return | Tenth near return | Longest alternating run |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Hold | 0, 0 | 0 | 0 | 0/9 | 0/9 | 0/9 | 2 |
| Empty fit | 0, 0 | 0 | 0 | 0/9 | 0/9 | 0/9 | 2 |
| Delayed RGB | 5, 12 | 1,999 | 1,699 | 0/9 | 0/9 | 0/9 | 2 |
| One release | 1, 2 | 315 | 265 | 1/9 | 0/9 | 0/9 | 6 |
| Three releases | 3, 8 | 1,558 | 1,358 | 5/9 | 0/9 | 0/9 | 6 |
| Five releases | 5, 12 | 1,999 | 1,699 | 4/9 | 0/9 | 0/9 | 8 |

One release sufficed for three counted near returns on one later game. More teacher experience raised observed second-return frequency through the three-release budget, with a decline at five. The one-, three-, and five-release fits reached their first second return in 44, 40, and 42 post-handoff frames respectively, each starting on the first held point. On seed `1866323335`, the five-release pupil made legal landings at frames 70, 99, and 139 after the teacher's frame-36 landing. The opponent landed four legal replies. In overlap with the corresponding hold points, the five-release fit changed shoulder pan, shoulder lift, elbow flex, wrist flex and wrist roll on 182, 180, 196, 198 and 191 frames respectively. The delayed-RGB branch changed all five channels and earned no second return. All 54 complete physical command tapes reproduce the strict referee events and terminal outcomes. [Demonstration pairs](rally-demonstrations.npz) · [Scored command tapes](rally-joint-feedback.json.gz) · [Eight-return video](rally-joint-feedback.mp4) · [Video verification](rally-joint-feedback.json)

## Independent shared-opening confirmation

A pre-existing schedule generated 24 new C/D releases. Each screened release tried the neutral teacher opening first, followed by the duration teacher opening if needed. Both attempts played live with zero continuation after frame 25. Thirteen produced an opening contact by frame 25, a legal near landing and a legal far reply. All thirteen qualified for the paired pupil comparison. The same frozen five-release fit earned a second near return on five of thirteen points. Hold and delayed vision each earned zero second returns. The fit earned 21 total near returns, compared with 13 per control. Its longest point reached four near returns and four far replies on seed `1838440417`. Every evaluation tape reproduced the full physical point. [Fresh course generator](../tt23/tt23_sequence_replication.json.gz) · [Screened and scored starts](rally-fresh.json.gz)

## Joint feedback from first observation

The from-release fit adds 300 opening transitions from the same twelve teacher demonstrations, increasing the fit to 1,999 transitions. More weight on body state helps distinguish initial and subsequent receiving postures within the same feedback law. The pupil receives the first visible RGB and joint state at release and issues every near-arm command through the terminal point. In the nine archived held launches, it scored a first legal return in eight and a second in two. Hold and delayed vision scored zero on both endpoints. Both points that reached the second near return had four alternating legal landings. The first second return occurred at frame 73 of the third point, after 197 cumulative frames of target play. All 27 physical command tapes reproduce the referee events and terminal outcomes. [From-release experiment](../../scripts/experimental/tt24_rally_from_release.py) · [Opening fit data](rally-opening-demonstrations.npz) · [Held points](rally-from-release.json.gz)

On all 24 fresh C/D releases at 320×240 per camera, the same fitted law earned a first legal return on ten points and a second on three. This cohort includes every scheduled release and takes every near-arm command from the policy. Both controls earned zero first or second returns. At the first fresh second return, on seed `1822090699`, the pupil landed frame 40 and frame 82 with two legal far replies. The run reached four alternating returns. Its second near landing took 342 cumulative frames of fresh assessment, or 11.4 seconds. All 72 complete tapes passed strict physical replay. [Full fresh course](rally-from-release-fresh.json.gz) · [From-release video](rally-from-release-fresh.mp4) · [Video verification](rally-from-release-fresh.json)

## Reward refinement with a shared legal opening

At 240×180 per camera, the five-release model receives an additional linear residual over three current image features for each of the five active joint rates. Nineteen practice points reuse one development release and its shared teacher prefix. Each part of a mirrored perturbation pair plays a complete point. Once both choices have finished, the residual with the best score so far becomes the center of the next pair. The score uses strict legal returns, with smaller credits for touches and alternating exchange length. The only post-handoff action supervision comes from the earlier stored demonstrations. The refinement consumes each practice point's own outcome after the point ends.

The unrefined first attempt scored only the opening return. Attempt seven landed near returns on frames 34, 75, 118, 153 and 192, of which the teacher supplied the first. Ten alternating returns preceded the terminal fault. The fifth near landing took 503 cumulative post-handoff control frames across the seven attempts. This is 16.77 simulated seconds of pupil activity, or 22.60 seconds including the seven shared prefixes. [Reward search](../../scripts/experimental/tt24_rally_reward.py) · [Ten-return practice video](rally-reward-practice.mp4) · [Video verification](rally-reward-practice.json)

The full practice schedule took 1,533 post-handoff frames, or 51.1 simulated seconds. On the nine matched held releases at the same resolution, the frozen refined model reached a second near return in five points, the unrefined model in three, and hold and delayed vision in zero. The refined model earned 15 total near returns compared with 13 before refinement. Both learned conditions peaked at three near returns and six alternating returns on held points. All 19 practice and 36 held tapes passed strict command replay. [Reward refinement receipt](rally-reward-refinement.json.gz)

A separate experiment expands the fit to 48 archived demonstration releases and evaluates every frame of 24 new E/F points. Its pupil reaches five near returns with five far replies on two independently drawn held launches. [Broader demonstration study](rally-bank.md)

## Reproduce

The repository's table-tennis environment uses MuJoCo, SciPy, Pillow and the existing Pong native component build. Use the table-tennis environment and build command from TT23. [Environment setup](../tt23/tt23_sequence.md#reproduce)

```sh
python -m scripts.experimental.tt24_rally_trial --collect
python -m scripts.experimental.tt24_rally_trial
python -m scripts.experimental.tt24_rally_fresh --workers 1
python -m scripts.experimental.tt24_rally_from_release --workers 1
python -m scripts.experimental.tt24_rally_from_release --fresh --size 320 --workers 1
python -m scripts.experimental.tt24_rally_reward --budget 19
python -m scripts.experimental.tt24_rally_movie
python -m scripts.experimental.tt24_rally_movie --practice
python -m scripts.experimental.tt24_rally_movie --from-release
```
