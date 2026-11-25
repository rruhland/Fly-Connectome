# RGB-only low teacher reverse course

The two physically rewarded low teachers are converted to motor templates using
only recorded RGB ball estimates, own joint encoders, issued references and
installed paddle geometry. Scalar physical legal-return rewards authorize them.
Neither native contact coordinates nor native receiving-line coordinates enter
template extraction or retrieval. Three focused tests cover this boundary.

| Declared physical practice | Cases | Cold | Frozen two teachers | Reward-growing |
| --- | ---: | ---: | ---: | ---: |
| Exact teacher reproduction | 2 | 0 | 2 | 2 |
| New goal neighbors | 8 | 0 | 7 | 7 |
| Six control ticks earlier, new goals | 4 | 0 | 2 | 2 |
| Untouched broad reply streams, late handoff | 4 | 0 | 0 | 0 |

Every counted return was also physically answered by the opponent. Paired incoming
far-contact and near-bounce records matched in all 18 triples. The six-tick earlier
handoff changes the observed arm by only .147/.221 maximum joint radians; this
does not establish a normal-start skill. Nine newly earned templates were retained,
but aggregation made no causal difference relative to the original two templates.
The independent broad draws failed. No official full-game promotion is claimed.

Practice scaffolding physically replays an actual teacher command prefix to a
declared handoff, then uses ordinary RGB/encoder control. Neighbor goals change
the actual far paddle's goal; no ball or arm teleport is used. Prefix 3999 is paid
once; each subsequent practice prefix and each of three actual suffixes is paid.
Total: 9062 actual control ticks, 632.016 seconds wall and 164.352 seconds CPU
(last completed-row lower bounds). The fixed bound was 900 seconds/12000 ticks.
Suffix CAD clearance is recorded per branch. The altered prep prefix was not
independently CAD-audited, so this artifact does not certify an entire new course.

`course.json.gz` retains all actions, RGB/body observations, actual offers,
positive and negative outcomes. `executed_source.py` is the exact frozen source
that ran. Its final hash-report expression failed after all 18 rows were saved
because it passed a string instead of a Path. `tt51_low_reverse_report.py`
reconstructed only final metadata and the already earned template list, with
zero physical steps; this repair and its source hash are recorded in the artifact.
The current runnable source fixes that type error and accepts JSON template rows.

Reproduce the native course with the documented single-worker EGL environment:

```sh
PYTHONPATH=.:src MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 LP_NUM_THREADS=1 /workspace/Fly-Connectome/.venv/bin/python -m scripts.experimental.tt51_low_reverse_curriculum --out artifacts/tt51_low_reverse/reproduction.json.gz
```

`RGBTaught(rows).proposals(forecast,path,x)` is the shared candidate API. Rows
contain `p=[x,z,vx,vy,vz]`, RGB-derived `depth`, relative phase and actual next
reference arrays. Candidate acceptance still requires the normal optical/body
forward checks; template identity is not a live input.

## Encoder-indexed body bridge pilot

`body-bridge.json.gz` tests a genuinely earlier actual body. Maximum joint
displacements from the successful handoff are .509/1.215 radians for the first
teacher and .517/1.241 for the second. A continuation maps own observed q/ref to
the next reference in an actually rewarded preparation trajectory, then holds its
final issued reference. It receives no source tick or teacher identity.

| Fixed course | Cold | Late templates only | Body + same templates |
| --- | ---: | ---: | ---: |
| Four exact incoming, earlier body | 1 | 2 | 2 |
| Eight new physical goals, earlier body | 5 | 2 | 3 |

All 12 incoming triples matched. The bridge reached the first source's original
successful q exactly, but the inherited late actor cleared its strike state
during preparation and never queried the taught finish. The second source had a
genuine new gain: case9, deep x=-.204363/z=.950585 m, cold miss and late-only miss,
bridge legal and answered, zero recorded suffix CAD penetrations. The historical
actor bank is the inherited after-two bank. Across this whole practice pilot there
were three bridge gains and four losses against cold. It is not promoted.

The completed pilot charged 7982 control ticks, 375.890 wall seconds and 132.356
CPU seconds. Source and dependencies remained frozen. An earlier implementation
was interrupted during its shared prefix when an endpoint reference bug was
caught; `body-bridge-abort.json` charges another conservative 3999-tick ceiling,
70 wall seconds and rounded16 CPU seconds. No completed practice comparisons
were discarded. Frozen executed source snapshots and logs are retained.

## Stock-search phase handoff and wider body bank

`stock-bridge.json.gz` corrects the isolated handoff problem by using the existing
ordinary stock search and its supported extra-option slot. Before the observed
body trajectory completes, the actor only observes RGB/encoders and issues the
learned continuation. It holds the final actual reference until the ordinary
incoming RGB window opens at y<-.49. It then chooses normal stock and learned
candidates through the same search. There is no inherited one-off strike gate.

At all four real earlier-body starts, body+two frozen templates returned and was
answered: **4/4 versus cold0/4 and late-only1/4**. The earlier starts are the same
.509/1.215/.517/1.241-radian displacements. The strong original `season.CARRY`
memory is restored unchanged, ordinary credit stays frozen, and growing matches
the frozen two-template controller. All paired incoming records match.

These are matched practice actors: they retain the existing attained-body helper's
constant home base supplied to `choose`, whereas official `mpc.play` supplies the
camera-derived raw base. They also begin after a physically replayed teacher
prefix. They establish causal body/late-memory transfer, not superiority over a
normal-start official baseline.

The declared central curriculum changes the actual far target x with requested
height/pace fixed. The first two resulting deliveries were instead physically
higher, about1.0477/1.0475 m at the evaluator's deep receiving projection, and all
four actors missed. The two-miss stop rule fired. This provides **no central-low
test** and no new reward-admitted templates. Nothing was tuned after those misses.
The course costs6538 actual control ticks,198.143 wall seconds and89.118 CPU seconds.

`stock-bridge-audit.json` independently replays all three causal body gains
(cases0,1,3) plus every corresponding control:12 branches with the shared3999-tick
prefix paid once. Entire native events match exactly; full solver-state CAD
checks include the previous prefix, preparation and strike, with zero penetrations.
Replay costs5683 control ticks,99.190 wall seconds and61.099 CPU seconds.

`observed-prep-bank.json.gz` exports the same two actual rewarded preparation
trajectories, extended backward to the first recorded RGB outbound crossing
y>-.99. They begin at observed4010 and contain70/53 rows. `EncoderPreparation`
in `tt51_low_prep_bank.py` offers read-only `preview(q,ref)` and stateful
`reset()/command(q,ref)`. Live queries read only stored body/reference arrays;
source clocks remain separate provenance. The extra earlier entries are recorded
experience and still need a causal activation test from ordinary games. The lead
deployable wing-control agent has this bank; duplicate full-game qualification is
intentionally avoided. Five focused tests verify privacy, admission and body-state
continuation contracts.

`tt51_low_early_neutrals.py` is prepared for the lead integration's `install`
factory. It fixes the existing five TT50 neutral draws, preserves source controls
through4003, restores legal camera/readiness history and uses the official raw
camera base with unchanged original carry. It has not yet run; there is no neutral
transfer result or duplicate full-game qualification in this artifact.
