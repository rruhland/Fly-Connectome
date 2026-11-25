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

## Integrated V1 on the complete existing neutral cohort

The lead `tt54_live_wing.install` controller was evaluated from the original
early native source state with normal camera-guided arm decisions after4003.
The five cases are the unchanged TT50 neutral roster, not newly selected goals.
Every requested far reply remains the ordinary broad TT37 draw. Each controller
gets the same actual prefix, original carry memory and official raw camera base.
Readiness is rebuilt from the actual camera/encoder history, not native labels.

| Reply stream | Deep projected x/z (evaluator only) | Stock | Cold | Warm |
| --- | --- | ---: | ---: | ---: |
| 791162625 | .0171 /1.0089m | 1 | 1 | 1 |
| 515671963 | -.2146 /.9643m | 0 | 0 | 1 |
| 767799421 | -.2947 /1.0789m | 1 | 1 | 0 |
| 643968721 | .1946 /1.0036m | 1 | 1 | 1 |
| 880742181 | .1998 /1.0467m | 1 | 1 | 1 |

All counted returns were answered. The low gain is offset by a higher wing loss:
**4/5 total for all three actors**, with one gain and one loss. At the low gain's
first public stock window4078, warm q is
[-1.405,-1.676,.515,1.589,1.275,-.1], versus cold
[.543,-1.628,-1.335,1.085,-1.140,-.1]. The acquired physical workspace was reached
from the earlier ordinary arm state. Normal search accepted the frozen
`low/training-new-left-reply` at4078 with support2. This establishes a causal
transfer, not an aggregate improvement or full-game qualification.

All five actual incoming triples matched. Original controls through4003 were
exact. All changed-prefix/strike CAD records have zero penetrations, and both
controller/evaluator hashes stayed frozen. The complete course costs6393 actual
control ticks,394.611 wall seconds and128.974 CPU seconds. Raw action/observation
receipts and the evaluator-only posture report are retained.

`live-wing-positive-replay.json` verifies the low gain and both controls with
newly constructed controllers and risk predictors. Every policy action, RGB
estimate and native event reproduces exactly. All three far-gate signals
(4022/-.45,4030/0,4042/.45) exactly match forecasts freshly generated in that
variant, ruling out stale same-tick predictions. Full prefix/preparation/strike
CAD has zero penetrations. Verification costs4389 actual ticks,255.653 wall
seconds and79.324 CPU seconds.

## Frozen V2: adverse partial comparison

The separately frozen V2 retained already-made preparation through ambiguous
camera updates and recovered its original low training challenge. Its full
five-case neutral attempt reached the predeclared450-second bound. Four complete
triples scored stock3/4,cold3/4,warm2/4. The registered first-three mixed subset
scored2/3,2/3,1/3. It retained the low-wing gain but lost both791162625 and767799421,
which stock/cold returned. No promotion is justified.

The fifth stock branch completed successfully; its cold branch was censored and
warm unstarted. Neither missing result is counted as a miss. The source artifact
retains that partial trajectory, and the report marks the cohort incomplete.
All completed incoming pairs matched, protected earlier controls remained exact,
CAD had zero penetrations, and source/evaluator stayed frozen. Cost6103 actual
control ticks,450.914 wall seconds and123.794 CPU seconds. The executed evaluator
snapshot is retained before adding the explicit registered first-three switch.

A V3 first-three mechanism diagnostic is now running under the lead agent's
request. Those labels have been inspected; it is not new held evidence. The lead
agent owns the separate unopened full-game qualification.
