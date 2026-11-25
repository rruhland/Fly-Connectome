# Signed motor practice from RGB-observed misses

On one matched physical offer, RGB feedback from four misses found a
new bounded stroke that landed. The same-budget blinded, reversed,
and prior-only final choices all missed. Across the 17 fixed forks,
however, guided practice and the blinded control each rescued eight.
The isolated success did not justify a new live policy or the reserved
fresh-offer trial. [Full verified results](report.json) · [Native joint and referee tapes](trial.json.gz)

| Method | Legal rescues out of 17 | Additional rescues after shared practice |
| --- | ---: | ---: |
| Unchanged failed stroke | 0 | 0 |
| Signed RGB response | 8 | 1 |
| Blinded final choice | 8 | 1 |
| Reversed RGB response | 7 | 0 |
| Nearest prior only | 7 | 0 |

## What was learned from a miss

The independent matched-motion archive provides 17 previously
collected exact decision forks across 13 measured releases. Each
assessment uses only the failed issued command stream and ordinary
past camera and encoder readings. Its two-dimensional change in
joint reference is learned from positive-minus-negative physical
motion contrasts at other releases only. The entire current release
is excluded, including other decisions within the same rally. Timing
uses the depth and pace already observed in stereo, not the phase
or contact time of the withheld scored movement.

Four native practice proposals test both signs of both directions at
the same intermediate amplitude. Each arm gets the same four attempts.
The learner fits no native trajectory or hidden ball state. The signed
choice compares RGB-observed outgoing paths from the failed probes
using projected net clearance and landing. It makes one larger move
along the clearly better measured direction, or falls back to the prior
if both optical differences are unsupported. The blinded, sign-reversed,
and prior-only controls each get the same one further move at the same
amplitude. The blinded control is sampled from the identical correction
subspace, so it controls for the signed update, not for the choice
to learn the subspace from outside releases.

Seven forks already yielded a real score on a shared probe. Actual
RGB supplied a departure estimate on 55 of 68 probes, including all
11 scored probes, and 12 of the 17 forks supported a signed update.
As a diagnosis only, projecting the actual withheld winning change
into the two learned directions retained a median 50% of its reference
change norm. That projection uses privileged assessment data after
the test. It is a clue to limited representation, not proof that other
playable movements are unreachable. [Original independent forks](../tt52_matched_movements/pairs.json.gz) · [Their motor contrasts](../tt52_matched_movements/lines.json.gz)

## The isolated 6458 rescue

None of the original or four practice motions scored. For the second
axis, the ordinary RGB projected-flight values were −0.33 in the
positive direction and −3.54 in the negative direction. The learner
issued the positive larger change and landed the identical incoming
offer. The three control choices missed. Independent replay reproduced
both the original failure and the new scored stroke from launch,
including every referee event. Neither tape recorded a clearance
penetration across a combined 1,775 full-CAD simulator samples.
The blinded control's one additional rescue was an earlier fork on
3401. A later fork on that same release already scored in shared practice.

The reserved fresh roster uses eight ordinary measured assessment
releases with no known scored alternate in the training materials.
It was not executed. The guided update did not improve upon equal-
budget exploration on the fixed first screen. [Unrun reserved roster](fresh.protocol.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt53_signed_practice
python -m scripts.experimental.tt53_signed_practice_report --span --verify
python -m pytest -q tests/test_tt53_signed_practice.py
```
