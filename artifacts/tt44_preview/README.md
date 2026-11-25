# Real-return learning of early optical preview

The delays learned from physical rewards transferred to **different joint
commands, not more returns**. All 18 practice pairs were exact command
matches under the shared exploration calendar and both arms scored 9.
The learner accumulated 26 physical outcomes, including 17 misses.
In 12 fresh frozen games it changed commands on four releases, using
two- and four-tick delays for a total of 59 changed early reference requests.
Each release scored the same number of learner returns as zero delay.
Both arms scored 7 total, with a peak of 4 on one rally. On release 6411,
the learned four-tick preview shifted the first arm contact by 4.46 mm.
The unchanged reactive opponent then delivered a different path toward the
same matched target draw. This is real physical control transfer, but
does not establish better return performance. The longest changed point,
8056, replayed exactly with ten native racket contacts, one launch, and
zero penetrations in 3,819 full-CAD solver samples.
[Adaptive tape](adapt.json.gz) · [No-update and zero-delay tape](control.json.gz) · [Verification](summary.json)

## Early readout

This experiment keeps the existing direct stereo-and-encoder joint field and
changes how far ahead that field looks in the early inbound phase. One delay
of 0, 2, 4, or 7 control ticks is chosen for a visually confirmed approach,
using three consecutive nearward optical estimates, velocity and own joint
displacement. The preview advances the optical position and velocity by a
short fixed time. It does not choose a table intercept or minimum distance
to a strike pose. The issued control remains a bounded six-joint reference
increment. The early preview ends at y = −0.53 m on the measured path.

The learner stores one context, delay and reward per physically delivered
attempt. The normal referee's actual legal landing counts as one and a
physical loss after the far side's legal delivery counts as zero.
Undelivered or capped attempts are censored. An analytical, local,
Beta-smoothed estimate of the success rate selects a delay with support
near the current context. The updater has no backpropagation, private
geometry access, opponent seed, replay, future observation, or hypothetical
reward. Where support is insufficient the actor uses its zero-delay
reference behavior.

## Prospective comparison

The first 18 previously registered broad-random practice cards use two
policies on the same measured initial shots and independent placement seeds.
Both use the same 55% exploration calendar and the same forced delay
whenever they share an approach ordinal. In exploitation slots the control
always uses zero preview and the learner uses its current actual reward
history. Arms may reach different total numbers of later approaches. The
trained learner then freezes its parameters for the twelve distinct later
assessment releases. The held control uses zero preview and both policies
stop updating or exploring. All arms use the unchanged continuously
reactive broad-random partner. The caps are eight alternating returns per
practice point and sixteen per held point, so this is an early-feedback
transfer experiment, not an assessment against the 100-return goal.
[Frozen readout and course](course.json)

The primary comparison uses raw direct feedback in both arms, without
the late private MPC. A separate four-card transfer kept the same frozen
late MPC and carried motor checkpoint. The established stock tapes for
all four ended before the transfer cap, so they are directly comparable.
Two games were unchanged at zero. The other two moved from 3 to 1
learner returns on 6411 and from 6 to at least 8 on 3786, which reached
the 16-alternating-return cap. Both arms had 9 observed learner returns
across this capped comparison. Early timing can change how the later
controller plays an offer, but the mixed effects do not support a general
improvement. [Late MPC tape](late.json.gz) · [Late comparison](late-summary.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
out=artifacts/tt44_preview
python -m scripts.experimental.tt44_preview adapt "$out/adapt.json.gz"
python -m scripts.experimental.tt44_preview no-update "$out/control.json.gz"
python -m scripts.experimental.tt44_preview_report "$out/adapt.json.gz" "$out/control.json.gz" "$out/summary.json" --replay-strongest
python -m scripts.experimental.tt44_preview transfer "$out/late.json.gz" --checkpoint "$out/adapt.npz"
python -m scripts.experimental.tt44_preview_report "$out/late.json.gz" artifacts/tt43_motor_modes/held-stock.json.gz "$out/late-summary.json" --archived-late
python -m pytest -q tests/test_tt44_preview.py
```
