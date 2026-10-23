# Minimum-phase first-interception fitness

This executes the [selected objective test](2026-10-23-worst-phase-selection.md).
The preceding mean-balanced search learned LEFT and damaged RIGHT; a subsequent
fresh-reward interpolation grid selected the unchanged RIGHT head. One bounded
population search now changes only candidate fitness from the average phase rate
to the smaller phase rate. It keeps the same RIGHT theta6, legal features, native
physics, six LEFT plus six RIGHT contexts, perturbations, ranking and update cap.

For each candidate, compute six-start LEFT rate L and RIGHT rate R, then use
min(L,R) for direction ranking, spread normalization and the original mirrored update.
Save L, R, their mean and minimum. The minimum penalizes imbalance but is not a
retention constraint: improving the floor can still lower the stronger phase. Six-start
estimates are noisy and coarse. This fresh-seed experiment cannot by itself establish
causality, incompatible representation or reliable generalization.

Root: `/workspace/m1b-experiments/pong-minphase-interception-20261023-r1`.
The external PROTOCOL.md governs exact execution. Same eight directions, mirrored
0.25 perturbations, top four directions and 0.10 function-RMS update cap; max 12,
minimum five complete populations. Last complete endpoint only. Same 160,000
training transitions/1,200 seconds, 900-second evaluation and 2,100-second total.
Same 576 fresh jobs and eight registered learning/retention screens; unconditional
80% is separate and replication remains necessary. No BP, teacher, privileged policy
input, phase-label input, state rewrite, new architecture or automatic continuation.

Learned ancestry remains 558,832 training transitions: 532,296 used and 26,536 unused.
The failed balanced donor and scalar-grid result are not carried. Separately paid
recent research is 159,854 balanced transitions, 28,167 scalar-grid transitions and
14,879 interrupted transitions. No partial fitness or prior observations are reused.

Ten focused checks passed in 4.54 seconds. A prelaunch prose correction was recorded
without observations; final restage took 0.440 seconds and own review 0.063 seconds.
All 336 accepted canonical starts and 301 rejected draws are reserved, excluding
40,258 prior IDs. Parent independent review passed in 0.504 seconds with zero
observations, reconstructing schedules, candidates, maps and constants. Core source
remains 68d15a7. Frozen driver SHA256:
141a1928063e06cc8aafb763ab395a8853641d2ff02c0f8c5900f14aabb7a718;
protocol e1dfb3941c622c674440ed8f5a3bdec579d4c7296d75b311285e28f7b6d92856;
manifest 88c482f43645c171e7ca94084fccc35f4c0e2c48098d3492d5dc8f5446ace29a.

Launched exec20934 at 2026-10-23T07:03:27.304033303Z, original start
1792739007304033303 ns. The full budget ends by 07:38:27, before the 07:59 shutdown
reserve and 08:09 user deadline. Health and costs only until exit; source, driver and
protocol frozen through both result checks. No new candidate automatically follows.


## Verified result and closure

The coordinator exited successfully. The run was complete and evidence-eligible,
but it failed combined first-interception learning and five of eight required screens.

| First-return cohort | Immediate start | Endpoint | Paired lift and 95% interval |
| --- | ---: | ---: | --- |
| LEFT | 48/64 | 45/64 | −4.69 pp [−15.63, 6.25] |
| RIGHT | 44/64 | 42/64 | −3.13 pp [−15.63, 9.38] |
| Equal-weight primary | 71.88% | 67.97% | −3.91 pp [−12.50, 4.69] |
| FULL first return | 47/64 | 44/64 | −4.69 pp [−18.75, 7.81] |

The changes against the immediate start do not establish definite damage. The
endpoint was below DIRECT on LEFT, 45 versus 54/64: −14.06 points, interval
[−23.44, −4.69]. This does not attribute the whole DIRECT deficit to new training.
Only FULL first gain versus GLOBAL and FULL all-return preservation versus GLOBAL
and START passed. FULL all returns were 75/138, versus START63/127 and GLOBAL55/119.
Endpoint first rates were LEFT70.31%, RIGHT65.63%, FULL68.75%; no separate 80%
competence, combined promotion, replication or rally-completion claim.

A secondary FULL signal deserves preservation: at least three returns increased
from 2 to 9/64 versus START, +10.94 points [1.56, 20.31]. At least two returns and
mean hits/game had intervals crossing zero; GLOBAL already achieved 8/64 games
with at least three returns. This is one unreplicated result among several secondary
comparisons, not confirmed rally learning. Keep the head as a separate research artifact.

Nine complete populations all updated, with steps 0.0741–0.1000 function RMS,
net displacement 0.25267 and 253 cache action changes. There were 277/864 binary
paired outcome contrasts and 55/72 nonzero minimum-fitness direction contrasts.
This was a substantive test. Small-bank estimation, proposed-update generalization
and initialization remain plausible; incompatible representation is not established.

Actual cost was 159,830 training transitions: 153,978 used and 5,852 unused in an
incomplete population. Evaluation cost was 89,902 transitions (RIGHT25,653,
LEFT8,575, FULL55,674), plus 2,374 initial images overall. Wall time was 1,598.4459
seconds: 1,015.2461 training and 583.1998 evaluation/reporting, within all caps.
The selected checkpoint is the last complete theta_at9.npy, with processed ancestry
718,662 training transitions: 686,274 used and 32,388 unused. Other noncarried
research costs remain separate; the unchanged RIGHT starting head is preserved.

Own completion check passed in 16.7024 seconds; parent independent check passed
in 3.558 seconds. Both added zero observations or replay. Every minimum-driven
update, RNG/map/source/completion pin, native count and paired gate was verified.
Raw reads are complete, source freeze released. Archival follows sidecar EXIT0 then
cold-journal EXIT0 with full reconstruction; heads, projection and reports stay plain.

Stop after this bounded attempt. The preserved main experimental candidate remains
RIGHTtheta6, and GLOBAL ARtheta112 remains the earlier qualified reference. A future
fresh TRAIN center-versus-proposal confirmation rule is a research recommendation,
not selected implementation. It requires its own prospective budget and final fresh
held-out evaluation. Do not restart old clocks, reuse partial fitness or launch another
candidate automatically. The session is closing before the 08:09 UTC deadline.


Archival and process closure are complete. Sidecar preservation exited successfully
in 17.016 seconds before cold archival began. Cold archival then exited successfully
in 75.746 seconds (77.702 CPU seconds): 2,374 databases, 7,003,230,208 original
bytes to 946,106,022 compressed bytes. Every original byte was reconstruct-verified;
zero new observations or replay. Heads, projections and reports remain plain.
The final closure receipt confirms no active training coordinators and records exact
plain-file hashes. No additional experiment follows; the session is paused.
