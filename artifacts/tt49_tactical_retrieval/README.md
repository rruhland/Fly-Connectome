# Goal-conditioned recall of physically earned arm motion

The tactical flight envelope has experimental causal support, but grading private trajectories with it can still choose actions outside the physical skill. This trial guides the original search with archived joint sequences that have already produced the right real flight. It leaves the history readout, outgoing risk ranker, original online motor credit, actuator governors, private search budget and opponent unchanged.

The envelope is refitted after removing every release in either list of the official season schedule. The selected analytic flight region has projected bounce depth 0.34–0.76 m, lateral x within ±0.16 m, and speed before the projected bounce 2.10–2.65 m/s. Four of 351 physically achieved flights across 66 independent practice releases had a missing or low next delivery. None of the 22 trials after a hard current ball did. No next opponent target participates in fitting any motion or in selecting any live arm movement. The allowed ranges are goals on the arm's own ball, not restrictions on the responding opponent. [Disjoint learned goal](envelope.json)

A motion qualifies from the original marked physical command archives only if it landed, its own achieved outgoing estimates fell in the learned envelope and the actual following reply was legal and not low. At a new trial, the first three proposed joint sequences come unchanged from the original state-conditioned learner. The actor can reserve at most two more slots for locally similar qualified movements from independent archived strokes. Stock native screening still decides the issued commands. This is prior selection rather than an extra population or a late search. Changed commands can cause different actual deliveries despite identical randomly requested settings.

The online bank uses actual legal arm landings to credit its original joint fragments. An in-envelope outgoing flight estimate from at least three independent causal stereo exposures can give those new fragments a pending tag. Two fresh stable measurements of the following bounced incoming arc must support a height of at least 1.002 m before those fragments are eligible for reserved proposals. Tags persist in the same bounded motor checkpoint. An unresolved label cannot promote a motion. The new tag is not a prediction of a target before the opponent chooses it.

The live protocol uses eight unfiltered initial season starts with the exact same starting checkpoint and complete points under the original continuous-wide TT37 opponent. The comparison is the earlier complete history-and-risk season, which also learned inside and between the games. Already captured season outcomes do not enter the fit, goal annotations, or live policy. [Frozen official schedule](official.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt49_goal_retrieval plan
python -m scripts.experimental.tt49_goal_retrieval run
python -m scripts.experimental.tt49_goal_retrieval report
```
