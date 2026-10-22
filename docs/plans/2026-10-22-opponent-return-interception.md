# Complete interception across native opening phases before claiming rally skill

## Selection and evidence

This follows the [population reassessment](2026-10-22-population-search-reassessment.md)
and the [continuous-learning scaffold plan](2026-10-21-continuous-learning-scaffolds.md).
Direct-opening acquisition replicated with the existing visual/linear policy class.
Two complete-rally configurations failed their rally-learning gates, although the
second further improved direct interception. General interception is unfinished:
in one recent ordinary cohort the designated direct policy returned only10/25
opponent-first openings versus17/25 for the original global policy.

Astra and reused Sol6.1XHigh independently proposed the same intermediate target:
learn first own contacts after the opponent returns the ball, while preserving the
new direct skill. Mixed full-rally training combined this opening phase with repeated
own contacts. Isolating the unfinished native phase is a genuine learning experiment,
not an oracle diagnostic or another bank/rate/threshold sweep. It does not by itself
identify whether vision, motor state or a learned control strategy changes.

## Prospective protocol

Start exactly from predesignated r1theta9 in
`/workspace/m1b-experiments/pong-leftward-population-20261022-r1`, retaining its
frozen generic vision and T projection. Keep DIRECT as a frozen copy and original
ARtheta112 as GLOBAL reference. Do not carry the failed integration or ALL8endpoint
because a secondary metric improved. No new architecture, backpropagation, teachers,
privileged observations, midgame resets or state rewrites.

Use ordinary native initial seeds whose initial ball direction is RIGHT, toward the
opponent. Uniform rejection by that direction only; preserve the complete correlated
native height/angle state. This is explicitly assisted training-distribution selection.
Each episode ends on first own hit or native win/loss/time limit,10s maximum. Binary
first-own-hit fitness is an allowed scalar scaffold reward; native scoring remains
recorded separately. The strong .65 opponent, greedy K1 policy and interval16research
execution contract remain unchanged.

Restore the demonstrated population rule:8function-RMS-normalized centered Gaussian
directions,+/−.25,12fresh shared RIGHT starts across all16candidates (192episodes),
top4stable maximum-sign ranking, selected-return spread normalization,.10function-RMS
update cap. Zero spread/contrast gives zero movement. Complete populations only;
all partial exposure charged, incomplete fitness unused. Fixed last complete endpoint,
no checkpoint selection or DEV peeking. No rate/noise/adaptive bank changes.

Budget160,000new training transitions/1200s,900s evaluation,2100s overall, one
immutable launch clock. Max12/min5complete populations. Expected24–26kframes per
population gives roughly6complete updates; report actual dose/nonzero movement.
This is a new prospective allocation, not an extension of either completed run.
Initial observations and inherited experience are reported separately.

Three frozen heads (ENDPOINT,DIRECT,GLOBAL) on64fresh RIGHT first-contact starts,
64separate LEFT first-contact starts and64separate unfiltered20s full games:
576jobs,max384k evaluation transitions plus576initial images. Fresh training and
DEV seeds exclude all prior raw/canonical used/unused/rejected schedules; final
streams remain untouched. All outcomes paired by start;10,000bootstrap samples
from a prospective root. Evaluation, annotation and statistics count in budgets.

## Separate learning, retention and competence

Required gates for `combined_first_progress`:

1. RIGHT first-return acquisition versus DIRECT: >=10pp lift and paired95%lower>0.
2. RIGHT first-return preservation versus GLOBAL: paired95%lower>−5pp.
3. LEFT first-return preservation versus DIRECT: paired95%lower>−5pp.
4. Unfiltered FULL first-return improvement versus GLOBAL: >=10pp and lower>0.
5. FULL all-return preservation versus GLOBAL: lower>−5pp.

Absolute competence is separate: all five gates plus endpoint first-return rates
>=80%on RIGHT,LEFT and FULL. This is provisional and requires independent training
replication before establishing a scaffold. A separate stronger RIGHT-superiority
screen versus GLOBAL (>=10pp/lower>0) is reported but not required: recovering old
RIGHT ability while retaining new LEFT ability can be genuine combined progress.
FULL>=2/3/4returns and mean own hits/game are secondary; no rally-completion claim.

RIGHT gain with LEFT loss identifies unresolved integration/retention. RIGHT gain
with retained LEFT and overall gain supports finishing broader interception before
rally training. RIGHT null despite substantive updates leaves information versus
credit/optimization unresolved; it does not prove vision or action history defective.
A partial/dose-limited result is separate. No automatic opponent-phase/hyperparameter
sweep after the outcome. Major architecture changes still require user review.

## Orchestration and outside evidence

Astra selected the question and criteria after independent hypothesis comparison;
Sol stages/executes and challenges, then both verify consequential evidence. Reuse
existing first-contact worker and original population math; add only focused tests
for changed schedule/cohort/gate contracts, no redundant execution replay.

The preceding reassessment records the primary literature actually read: Mania
et al.2018 for random search/seed variability, Kaplanis et al.2019 for behavioral
retention under within-task drift, and Ng et al.1999 for limits of policy-invariant
shaping in whole-episode search. None establishes this curriculum will succeed.
The decision is grounded in the observed native-phase performance gap. At selection, no new
allocation had launched; independent stage review came first.

## Independently staged

Root `/workspace/m1b-experiments/pong-rightward-interception-20261022-r1`, unchanged
core/worker/math source68d15a7. Eight focused checks PASS2.39s; no new execution or
cost replay. Sol stage PASS41ms after2.086s preparation. Astra independently
regenerated six-root native schedules, all336canonical starts/rejections, exact
maps/source pins and8directions/all192candidate cells in191ms; zero observations.

Roots: training2947108631, RIGHTDEV1596827437, LEFTDEV4103586919,
FULLDEV1179302843, optimizer3756142097, bootstrap2284095761; tag29131.
144RIGHTtrain+64RIGHTDEV+64LEFTDEV+64FULLDEV;293rejected candidates reserved,
37,940priorIDs excluded. Driver14b7fa99dad4f4a93c39206b88385a72f00c2257d2bc4a7ff40f090858b03b9f;
protocol544d2d577eb954a3a538b27adeb53951ab426c9d76b91850a6c9e1bc9e2ecf99;
prelaunch manifest016015758a3429232b7f27be3e5b42fd95f5acdb7a9e3857ae1abefdab66c940.
Launch authorized after both checks; same160k/1200train+900eval/2100overall,
min5/max12. Source/driver/protocol frozen through both postexit checks; DEV blind.

Prior pair-bank journals archived only after both checks:726DB2.1845GB→.3147GB,
20.134s; sidecar2.693s completed before archive. All bytes reconstruct-verified,
zero observations/replay, no overlap with this new training allocation.

Launched exec95719, originalstarted_wall_ns1792695554956184660
(2026-10-22T18:59:14.956184660Z). Same frozen caps; DEV blind, no extensions.

### Read-only representation review while evaluation remains blind

Astra checked `src/fly_connectome/vision/interface.py`: signed observed velocity is
bound to16×16current-position bins, and signed relative velocity is bound to9×9
relative-position bins. These interactions are already present; do not propose a
new architecture on the false premise that all velocity-position binding is absent.
The fixed projection can reduce resolution, and the policy does not receive an
explicit action-history feature. Neither observation establishes a causal failure.
No new observations, replay, diagnostic allocation or change to the running trial.


## Checked completion and next selection

Coordinator exited0. Six complete and nonzero original-rule updates used155,738
training transitions;4,095additional partial transitions were charged and unused.
Training total159,833; evaluation89,386 (RIGHT26,075/LEFT8,555/FULL54,756),1,759
initial images. Original-clock wall1467.965s =892.311training+575.654evaluation,
within all fixed caps. Net function RMS movement.211107,203cache greedy changes;
163/576direction-serve pairs and38/48mean direction contrasts were nonzero.

RIGHT DIRECT19→44/64: +39.0625pp,95%CI[26.5625,51.5625], acquisition passes.
RIGHT GLOBAL37→44/64: +10.9375pp,[−3.125,25], noninferiority passes;
optional superiority fails. LEFT DIRECT54→50/64: −6.25pp,[−17.1875,4.6875],
preservation fails. This does not establish definite degradation. FULL GLOBAL
first31→54/64: +35.9375pp,[23.4375,48.4375], gain passes; all54/118→81/144:
+10.4873pp,[2.2075,20.0293], preservation passes. Four of five gates pass;
combined_first_progress and provisional competence are false. Endpoint first-return
rates RIGHT68.75%,LEFT78.125%,FULL84.375% do not all meet80%. No rally-completion,
replication or combined/global-promotion claim. The existing class can learn the
opponent-return phase; preservation and higher reliability remain unresolved.

Sol completion verification PASS13.6918s reconstructs all six sequential updates,
optimizer RNG/maps/source/completion pins and raw greedy/native task boundaries.
Astra independent verification PASS3.2187s reproduces updates/counts/paired gates.
Both add zero candidate observations/replay. Source freeze released. Evidence is
in the experiment's summary.json/completion-verification.json and
`pong-mechanism-review-20261022/rightward-parent-result-check.json`.

After both checks and raw reads completed, sidecar preservation exited0 in3.4909s;
only then cold-journal archival ran and exited0 in56.6969s (56.8321s CPU).
1,759DBs5,248,086,016→689,020,049bytes, all original bytes reconstruct-verified,
zero observations/replay. Heads/T/metadata remain plain; raw DBs require restore.

Next selection is [balanced fresh practice](2026-10-22-balanced-practice-selection.md)
from the predesignated RIGHT r1theta6 experimental candidate. Exact RIGHT-only
replication is deferred; it has not occurred. Candidate accumulated training is
558,832transitions =532,296used+26,536unused. New balanced learning versus its own
start and required preservation gates must pass before any candidate promotion;
independent replication remains necessary. General plan/progress map are parent-owned.


## User pause after the selected balanced followup began

The user then asked to find a good pausing point and stop. The selected balanced
practice allocation was stopped promptly by coordinator SIGINT; workers settled,
exec9708 exited130 and no processes remain. Its immutable launch clock
1792697965537299559ns remains unchanged. Durable paid cost14,879training transitions+
161initials,161complete episodes, zero complete populations/updates/evaluation.
The partial population contributes no learning result. Last durable head is the
unchanged copy of this RIGHTtheta6. Exact costs/source/RNG/process checks are in
`pong-balanced-interception-20261022-r1/pause-receipt.json`; four settled jobs'
291transitions are included although their coordinator cost receipts were unwritten.
No new observations/replay were added by bookkeeping. The previous completed RIGHT
result above remains the last checked learning checkpoint.

Orchestration is PAUSED. Do not restart that old driver or reset its immutable clock.
Preserve journals/checkpoints; future learning requires user continuation and a new
prospective allocation/design/budget. No RIGHT replication or combined promotion.
