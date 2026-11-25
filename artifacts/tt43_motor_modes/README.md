# Nonparametric motor readouts

The added readout did not improve this controller. On all 12 paired
broad-random games, it issued exactly the same bounded commands as stock,
reproducing all physical events and all 63 target draws. Both arms scored
51 learner returns in total and peaked at 19 in one point. None reached
the target of 100 learner returns in one rally. [Paired physical tapes](held-motor.json.gz)

At 38 replans the branch produced 1,226 additional short-horizon candidate
trials. The private model detected contact in 393, but none of the
shortlisted candidates passed the longer nominal landing trial. Thus the
new modes never displaced the stock behavior. On 9550's terminal offer,
even the three eligible rescue windows found no candidate contact when
stock's closest nominal separation was 9–10 cm. This positive-action
library did not span that low-ball approach. Search increased measured
CPU time from 157.8 to 173.1 seconds across the 12 games.

## Method

The stock controller tests remembered direct-joint fragments, differentiated
paddle corrections and short random slips. This branch asks if a continuous
proposal fit across several *different* legal strokes exposes a good action
the copied fragments do not provide.

The motor readout uses stereo motion and own joint/reference history. A
fixed depth provides a shared **phase coordinate**, not a strike plane or
a required contact location. Each earned fragment's 21 issued-reference
ticks are resampled at 11 phase knots. The search takes at most one
fragment per physically credited stroke. It fits up to three movement
families separately by weighted local ridge. The mean, fitted extrapolation,
changes in timing, and two learned residual directions provide novel
sequences. The public readout does not use a target placement or actual
contact time. There is no backpropagation.

Stock search runs first. The new branch requires an observed bounce, a
nearward optical estimate between y = −0.76 and −1.53 m, and no robust
stock choice. The causal private motor/ball model tests each new guess
and a few geometrically refined variants. A candidate must land in the
nominal prediction and at least one of the two perturbation trials before
it can be selected. The live arm still issues the same bounded incremental
joint references on the same court. In live mode, real legal returns
refresh the existing carried optical/action bank. Terminal miss fragments
of actually issued commands provide a bounded action-similarity penalty
inside that new branch. The held comparison freezes both memories to
isolate the effect of the readout.

The frozen schedule is the first 12 unfiltered cards of the prior prospective
assessment, with the previously saved 24-game learning checkpoint. Both
arms use the exact same measured physical launch for each card, as well as
the same independent random opening and reply placement seeds. The reactive
opponent solves each delivered return from the actual physical path, so
a different arm stroke can change later paths without changing the target
draws. The matched stock arm was already executed and is included byte for
byte. [Frozen schedule](held.protocol.json) · [Archived stock](held-stock.json.gz)

The joint-control tape stores the bounded increments, exact random draws,
referee events, motor limits, and nominal search diagnostics. The checkpoint
stores the optically credited joint-reference fragments and the terminal-miss
memory. The deterministic readouts are fit on request from those fragments.
No held-game score or terminal event enters a nominal trajectory candidate.

A separate three-game live exercise starts from a clone of the same
checkpoint and takes the next practice cards, not the assessment cards.
It limits each game to six alternating returns to verify memory wiring
without making a durability claim. The live exercise excludes post-miss
tracks already behind the arm when finding the reachable incoming window,
while retaining their actual issued commands in the tape. Games 7715 and
4214 each landed three learner returns and added 14 and 12 ordinary
positive fragments, respectively. The physical miss on 15093 between
them added three negative action windows that remained in the checkpoint.
[Live commands and outcomes](practice-updates-live.json.gz) · [Live checkpoint](practice-updates-live.npz)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
out=artifacts/tt43_motor_modes
python -m scripts.experimental.tt43_motor_modes motor "$out/held-motor.json.gz" --stock "$out/held-stock.json.gz"
python -m scripts.experimental.tt43_motor_modes_report "$out/held-motor.json.gz" "$out/held-stock.json.gz" "$out/held-summary.json" --replay
python -m scripts.experimental.tt43_motor_modes_practice "$out/practice-updates-live.json.gz"
python -m pytest -q tests/test_tt43_motor_modes.py
```
