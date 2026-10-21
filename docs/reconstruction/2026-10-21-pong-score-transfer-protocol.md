# One frozen Pong score-transfer study

Both preselected qualified return-learning endpoints are evaluated once against
opponent speed **.35**, with their original generic visual prior, frozen raw
linear weights, transform and radial cap. No B-trained memory, policy selection,
training, coefficient change, opponent sweep or backprop occurs. This is transfer
from hit-reward learning, not acquisition from score reward or complete M1B.

There are **64 fresh shared development serves**, independent fresh camera phase
per serve, 10 s / 1,200 physics tick horizon, and nine cohorts: both endpoints,
both original restricted initial heads, down/no-op/up/random1/random10. All seven
controls participate in each endpoint's primary comparator. Frozen greedy trained
heads act every camera; random10 repeats its uniformly sampled action ten frames.
All evaluation rewards remain raw score; timeouts count as nonwins. The same
random-action seed is used by random1 and random10 on each serve.

The finite allocation is 576 jobs, at most 288,000 transition cameras, four
single-thread spawned workers and the existing imports-only GC freeze. The
original clock starts only at launch, with **18 min for rollout and 20 min overall
including annotation and bootstrap scoring**. Existing per-camera durability,
snapshot/replay bounds, full frozen mutation checks and completion pins remain.
No extension, replacement, early selection or hidden restart is permitted. Prior
failure costs survive resume; completed outcomes are retained with their original
wall eligibility and all raw partial work remains visible.

DEV root **1492093789**, action root **836917243**, tag **29115**, bootstrap root
**1236702941** are fixed prospectively. `SeedSequence([DEV, tag])` supplies
positive 31-bit serve seeds by rejection against all snapshotted prior declared
and actually expanded development/training/calibration/diagnostic IDs, new roots,
reserved final IDs and duplicates. Control actions use
`SeedSequence([action_root, tag, serve_index]).generate_state(1)`. A collision is a
prelaunch error, not permission to choose results. No final stream is expanded.
Preparation and tests use distinct test roots; staging expands the actual cohort
only after this recipe is frozen and records the complete exclusion metadata.

Primary metrics per endpoint are total-win rate and own-hit-followed-win rate
(raw positive terminal score with at least one prior own hit). The comparator
maxima are taken independently across both initial heads and all five simple
controls. A policy passes this practical transfer screen only when **both point
lifts are at least 10 percentage points and both paired 95% CI lower bounds are
strictly positive**. Replicated score transfer requires both endpoints passing;
the fixed-two-policy average is descriptive. Ten thousand bootstrap draws resample
the same 64 serve clusters, retaining all nine outcomes and recomputing each
control maximum within every draw. They are not 128 independent trained trials.

Also report natural-serve wins, losses, timeouts, first incoming returns, all
incoming opportunities, repeated returns, opponent reliability and physical time.
Incoming geometry before first contact is checked across all nine policies;
subsequent trajectories may differ. The fixed 10 s horizon constrains repeat
counts and is not a capacity ceiling. Continuous effects and uncertainty remain
reported if any screen fails, without changing the original qualification claims.

Qualification manifests, summaries, original source archives, readout mappings,
own/parent verification, current source and helper/protocol bytes are pinned. The
source bridge explicitly lists eight previously reviewed optional control files;
visual code, generic prior, physics, encoder, readout and action semantics remain
unchanged. Archived actual-policy default prefixes plus the byte-equivalent codec
proof anchor this bridge. New source signatures are retained honestly; full
session hashes with different source metadata are not claimed equal. Initial
images, diagnostic reexecutions, new evaluation cameras and posthoc physics audit
work are separately accounted. No part of this cohort selects a candidate.
