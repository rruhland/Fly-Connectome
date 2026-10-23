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
