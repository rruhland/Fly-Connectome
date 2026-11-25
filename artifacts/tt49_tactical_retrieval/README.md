# Goal-conditioned recall of physically earned arm motion

The tactical flight envelope has experimental causal support, but grading private trajectories with it can still choose actions outside the physical skill. This trial guides the original search with archived joint sequences that have already produced the right real flight. It leaves the history readout, outgoing risk ranker, original online motor credit, actuator governors, private search budget and opponent unchanged.

The envelope is refitted after removing every release in either list of the official season schedule. The selected analytic flight region has projected bounce depth 0.34–0.76 m, lateral x within ±0.16 m, and speed before the projected bounce 2.10–2.65 m/s. Four of 351 physically achieved flights across 66 independent practice releases had a missing or low next delivery. None of the 22 trials after a hard current ball did. No next opponent target participates in fitting any motion or in selecting any live arm movement. The allowed ranges are goals on the arm's own ball, not restrictions on the responding opponent. [Disjoint learned goal](envelope.json)

A motion qualifies from the original marked physical command archives only if it landed, its own achieved outgoing estimates fell in the learned envelope and the actual following reply was legal and not low. At a new trial, the first three proposed joint sequences come unchanged from the original state-conditioned learner. The actor can reserve at most two more slots for locally similar qualified movements from independent archived strokes. Stock native screening still decides the issued commands. This is prior selection rather than an extra population or a late search. Changed commands can cause different actual deliveries despite identical randomly requested settings.

The online bank uses actual legal arm landings to credit its original joint fragments. An in-envelope outgoing flight estimate from at least three independent causal stereo exposures can give those new fragments a pending tag. Two fresh stable measurements of the following bounced incoming arc must support a height of at least 1.002 m before those fragments are eligible for reserved proposals. Tags persist in the same bounded motor checkpoint. An unresolved label cannot promote a motion. The new tag is not a prediction of a target before the opponent chooses it.

The live protocol uses eight unfiltered initial season starts with the exact same starting checkpoint and complete points under the original continuous-wide TT37 opponent. The comparison is the earlier complete history-and-risk season, which also learned inside and between the games. Already captured season outcomes do not enter the fit, goal annotations, or live policy. [Frozen official schedule](official.json)

On the first official release, 8831, recall extended the point from 27 to **40 legal arm returns**. The candidate made 17 in-envelope strikes versus eight for the baseline, and neither arm saw a low tracked reply on this release. The 40-return tape contains 41 opponent returns and lasted 65.36 simulated seconds. Command-only replay reproduced every event and opponent draw. All 31,372 full-CAD solver states had positive clearance. This is an individual whole-game improvement, not evidence that every game improved or a new across-study record. [40-return tape](official-8831-tape.json.gz) · [Exact native replay](official-8831-cad.json)

## Is the next ball actually returned?

A separate diagnostic freezes three already documented pairs of actual achieved strokes. It includes a hard incoming case, an ordinary case with a lower next-height tail and one where the lower-risk shape had a lower mean reply height. Each matched pair receives four fresh common draws from the original independent continuous target law. The prefix until after the earlier physical outgoing separation follows the recorded actions, keeping the actual different body states both strokes leave. The same ordinary stereo/encoder direct learner takes over and attempts the next own legal return. It starts with the same frozen motor bank in both worlds and never reads the draw, the recorded hit time or the branch label. The evaluator's timed handoff is not a new live cue. This is not a full-game or whole-policy test. [Frozen receiver comparison](receiver-plan.json)

The ordinary receiver landed its next legal arm return on **12 of 12** matched offers after the lower-risk stroke and **10 of 12** after the higher-risk stroke. The same four fresh target vectors were reused on the three physical pairs. Both choices preserved the preceding legal arm return. Every opponent draw produced a legal next ball. The two paired gains were on 6619 and 2920, and 8627 tied at four each. All four branches in the two discordant pairs reproduced exactly from their issued commands, with no penetrations in 6,733 full-CAD solver states. This supports a controllable benefit for actual receiving success in these tested contrasts, including the body state each stroke left. It does not show that the learned region protects against every future target. [Paired physical receiver results](receiver-report.json) · [Actual issued suffixes](receiver-live.json.gz) · [Exact native replay](receiver-cad.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt49_goal_retrieval plan
python -m scripts.experimental.tt49_goal_retrieval run
python -m scripts.experimental.tt49_goal_retrieval report
python -m scripts.experimental.tt49_tactical_compare
python -m scripts.experimental.tt49_tactical_receive plan
python -m scripts.experimental.tt49_tactical_receive run
python -m scripts.experimental.tt49_tactical_receive report
python -m scripts.experimental.tt49_receiver_audit
```
