# Prepare softly, then let the optical controller choose the strike

The +12 cm clamp creates useful flat-face contacts, but TT69's early
waveform replay lost routine hits and its late-only learned search
never issued a replacement. This candidate learns only a preparation
target from the real earned joint references. It sends a bounded fraction
of that target into the normal controller before the visible bounce.
The ordinary late selector and the TT69 learned candidates decide if and
how to strike. The new early policy never issues a stored swing.

An optical height correction uses actual RGB and encoder traces on the
independent TT69 physical school, including both scored arrivals and
receiving failures. A low-likelihood estimate comes from current stereo
features and release-disjoint calibration error. Rather than assume a
specific landing is known exactly, the preparation readout mixes useful
reference targets across plausible lateral and vertical arrivals. It
uses only nearby earned poses with a compatible joint configuration.
Two consecutive fresh optical estimates must agree before any preparation.
Late controls use only RGB, own encoders and known fixture calibration.
The known TT68 success on 10544 is not a training or held case.
[Locked real opponent scopes](protocol.json)

The 72 neutral headless scouts each use the ordinary unmodified
continuous random far draw. They fix one actual sub-1 m first delivery
and one ordinary control on each of six previously unopened releases.
Shot classification only stratifies the evaluation. The actor does
not see it. The matched comparison keeps a registered stock arm, an
equal late-only arm, and the same late arm with the learned early
preparation. A new physically answered low return with no lost standard
hit is required to open sixteen new full unfiltered points. Both arms
would retain ordinary live reward credit inside each point.

## Result

All three arms returned the same 1 of 6 low and 5 of 6 routine offers,
each duly answered. Every incoming far contact and first bounce matched
the native scout (36 comparisons). Preparation changed commands before
the bounce on all six low offers and no routine first offer. Two routine
points received preparation only against a subsequent incoming ball. No
audited tape incurred CAD penetration. The equal late-only and preparation
arms screened 150 and 148 joint paths respectively. Neither issued a
learned late replacement. Zero gained legal low returns and zero lost hits
failed the prospective gate, so the full points were not run.

The exact same offer on release 9550 changed from no contact to a rim
contact at a 75.75 mm radius with face alignment -0.45. The ball still
traveled toward the learner at 2.01 m/s after separation. That extra
reach did not create a usable flat-face impulse. The independent stock
success on 10544 used the negative face with shoulder rotation 0.176
rad and wrist flexion -0.383 rad from registered ready. Its credited
search first chose a movement only after the bounce. This witness never enters
the policy or qualification counts.

The height residual fit did not improve release-disjoint estimates on
the training archive (215 frames across 27 physically exercised
release IDs). Depth RMSE increased from 35.15 to 36.10 mm. The low-event
Brier score worsened from 0.141 to 0.161. The frozen fit was evaluated
without changes after looking at those diagnostics. This study rejects
promotion of this small corrective preparation and late selector, not
the physical feasibility of clean low returns already shown by TT69.

[Optical calibration frames](calibration.json.gz) · [Frozen learner](learner.json) ·
[Held commands and referee tape](held.json.gz) · [Audit and movement report](report.json)

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt81_conditional_preparation plan
python -m scripts.experimental.tt81_conditional_preparation capture
python -m scripts.experimental.tt81_conditional_preparation fit
python -m scripts.experimental.tt81_preparation_play first
python -m scripts.experimental.tt81_preparation_play games
python -m scripts.experimental.tt81_preparation_report
python -m pytest -q tests/test_tt81_conditional_preparation.py
```
