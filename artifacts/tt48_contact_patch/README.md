# Same-ball edge rescues on the stock arm

Three exact-state repairs turned four archived low, touched terminal misses
into three legal returns that the unchanged opponent answered. All three
rescues still used the outer edge of the 76 mm blade. The only tested
inner-face contact was at 57 mm and still failed. Steering the impact was
useful, but forcing it toward the blade center was not a prerequisite.
[Exact-state practice, bounded commands and referee events](causal.json.gz)

An unchanged-grid replication on four later, distinct measured releases
added one new same-ball rescue on 9114. It moved the native alignment
from .74 to .997 while keeping the contact at roughly 76 mm radius.
The fixed opponent answered the newly scored shot. Both control tapes
replayed exactly from launch with no penetrations in 6,610 full-CAD
solver samples. The other three probes still missed. This replicate was
not added to the three deployed treatment donors.
[New offers](new-offers.json.gz) · [Independent replay](new-offer-audit.json)

| Previous missed release | Projected height | Stock contact radius | Tested edge radius | Edge response |
| --- | ---: | ---: | ---: | --- |
| 9230 | .998 m | 73.8 mm | 74.9 mm | Legal and answered |
| 15456 | 1.002 m | 73.1 mm | 75.4 mm | Miss |
| 11809 | .991 m | 75.2 mm | 75.4 mm | Legal and answered |
| 2707 | 1.014 m | 75.2 mm | 76.0 mm | Legal and answered |

9230 and 11809 struck roughly 31–38 mm deeper than stock and increased
alignment of the native impact normal with the blade-face normal from
.38 and .45 to .94 and .91. The alignment already exceeded .97 on the
2707 stock miss. There the useful edge alternative changed the movement
and outgoing path, not an already good alignment. Every altered suffix
was sampled in the full CAD scene without changing its dynamics. No
penetrations were measured.

## Controlled practice and knowledge boundary

Each branch starts on a copy of the same physical solver state following
the far hit and before the target incoming bounce. The archived command
prefix and original failed suffix reproduce the archived referee events.
The grid varies aim point on the disk while sharing the incoming contact
phase, desired orientation families and target rates. Requested radii
do not necessarily equal actual radii. At each planned phase the exact
practice plant screens bounded direct reference changes relative to the
old tape. One physically tested candidate per observed radius band is
selected on the same nominal rollout cost. Bands are inner face below
59 mm and rim at 63–82 mm, not exact-center and not a proof of the
continuous reachable set. The comparison is a finite constructive
test on historically selected misses. Unchanged target draws, far
contacts and near-side bounces are checked against the original physical
delivery. Ball state and actual collision times in this diagnostic are
privileged training/evaluation inputs, not signals passed to a deployed
player.

## Optical transfer

The transfer proposes only a direct joint residual derived from the
actual issued contrast on one of the three legally answered practice
corrections. It does not replay any source tape. The ordinary season
controller runs first. If a stock legal route exists it remains untouched.
If the camera-initialized independent plant predicts a weak near-rim
departure, an additional native constraint inspection reads relative
impact-normal alignment in that independent plant. A few bounded time
ramps cover the learned motor changes. Only legal candidates that retain
normal alignment and survive at least one perturbed optical hypothesis
replace the original action. A bounded lease retains the motion through
its predicted contact; a measured outgoing ball restores ordinary
feedback. No true current opponent aim, simulator ball, live concealed
body state or future observation enters the actor.

The fixed evaluation uses all 24 season assessment assignments, eight
untrained measured releases and three independently redrawn broad streams
per release. The reference is the season-before matched frozen stock actor
with the same parent-48 memory, scene and rules. A cap of 240 alternating
returns is unchanged. [Fixed assessment and physically earned deltas](held.protocol.json)

## Reproduce

```sh
export PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1
python -m scripts.experimental.tt48_contact_patch
python -m scripts.experimental.tt48_contact_patch --source artifacts/tt46_season/learning-48.json.gz --output artifacts/tt48_contact_patch/new-offers.json.gz --indices 27 37 40 43
python -m scripts.experimental.tt48_contact_patch_report --practice-witness artifacts/tt48_contact_patch/new-offers.json.gz --archive artifacts/tt46_season/learning-48.json.gz --out artifacts/tt48_contact_patch/new-offer-audit.json
python -m scripts.experimental.tt48_rim_recovery plan
python -m scripts.experimental.tt48_rim_recovery run
python -m scripts.experimental.tt48_contact_patch_report --verify
python -m pytest -q tests/test_tt48_contact_patch.py
```
