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
The decision is grounded in the observed native-phase performance gap. No new
allocation has launched; independent stage review comes first.
