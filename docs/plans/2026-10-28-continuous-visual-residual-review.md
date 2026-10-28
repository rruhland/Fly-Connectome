# Review proposal: continuous anonymous visual entity/pair residual

Status: USER APPROVED in chat on 2026-10-28. One fixed 600-second comparison only.
Implementation reviewed; native pilot launched with manifest SHA
30b97d9242071c30ca6f353603bc681559ddfb6f4983dfdc1ae020e97123923d.
No promotion or additional architecture variants are implied.

## Reason for the proposal

The current40069 research baseline has662TRAIN256 returns,233first returns and
161games reaching a second return; later429/615(69.76%). It is not reliable
throughout a match. Complete-prefix learning improved TRAIN but failed fresh.
The full-coordinate actor regressed662→566 on256games. Restricting corrections
to198smooth spatial coordinates still regressed287→237 on128games, including
151→126 on scheduled training starts and136→111 on other starts. Neither actor
was promoted. Stop actor-coordinate/step/temperature variants in this branch.

Hypothesis: retain joint continuous visual position/velocity and conditional pair
geometry before anonymous spatial bins discard information. This is a
representation hypothesis, NOT proof that existing features lack sufficient
capacity. Credit variance, limited experience, and stochastic-training versus
greedy-deployment mismatch remain plausible explanations for prior failures.

## Exact fixed feature construction

Keep frozen vision, tracking, its existing velocity calculation, sensor cadence,
the4389-bin transport, cap/scaler, actuator, and existing512TensorSketch unchanged.
The new branch reads only `latest_encoded['entities']`, which is already produced
from legal camera observations. Use its normalized_position, observed_velocity,
observed flag and velocity_known flag. No physical game state, actions, rewards,
object IDs, object roles, sizes inferred from game rules, or target coordinates
enter the features. Existing tracking IDs remain internal to frozen vision; they
are not feature values. No new velocity filter or learned vision is introduced.

Each entity has the six-coordinate descriptor
`[py,px,vy,vx,observed,velocity_known]`. Unknown velocity remains the existing zero
vector with velocity_known=0. Observed and predicted entities are both retained.
For EVERY ordered pair of distinct entity instances `(i,j)`, use twelve coordinates
`[pi_y,pi_x,vi_y,vi_x,pj_y-pi_y,pj_x-pi_x,vj_y-vi_y,vj_x-vi_x,oi,ki,oj,kj]`.
The anchor, signed relative geometry and velocity association remain together;
no object is assigned a ball/paddle/controlled-agent identity.

For each descriptor family, center and divide ONLY its continuous coordinates
by frozen TRAIN standard deviations. Flags remain0/1. Give episodes equal weight,
frames equal weight within episode, and entities/pairs equal weight within frame;
a crowded frame must not dominate calibration. Use denominator floor0.1 times
the median positive continuous-coordinate standard deviation. Record floors;
stop if an entire continuous descriptor family has no positive scale. Parent
review selected this substantial floor before any data or execution, rather than
a numerical epsilon that could amplify a nearly constant calibration coordinate
into rapidly varying Fourier phases after a small later change. No floor sweep. Do not
clip continuous descriptors or alter the already capped base-policy input.

Use ONE fixed RNG seed831496043. Draw, in order, entity matrix128×6 and pair
matrix128×12 with entries N(0,1/d), followed by their128uniform[0,2π) phases.
This fixes Gaussian-kernel length scale sqrt(d) in standardized descriptor
units; no bandwidth/frequency/seed/dimension sweep or reward-based map choice.
For a nonempty family compute `sqrt(2/128)*mean(cos(A*z+b))` over its descriptors.
Entity and ordered-pair blocks have128features each:256total.

Freeze pooled feature means and RMS denominators from the same TRAIN calibration,
again with equal episode/frame weights. Subtract each block's mean only when
that block is available; absent entity/pair blocks stay exactly zero. Use RMS
floor0.1 times median positive feature RMS; stop if all feature RMS are zero.
The pair block is zero for fewer than two entities. Pooling and calibration use
canonical descriptor-content order for reproducible floating reductions; ID or
list permutation must not change features. Compute all ordered pairs in chunks;
do not select a game-specific neighbor, entity subset, or privileged association.

## Preserved policy and training rule

Freeze40069W and its existing R and all existing map assets:
bundle SHA40069e2deae2b745b83b6c193389fa740450856b67443339b12f0dbb9bf3c438;
W SHA479472a37e9950246726cd90babf44f053467e3ae354783da2afbc35080ea15f.
Add only3×256 output weights C, initially zero. Compute
`Q=original40069_Q+C@new_features`. At exactly zero C, literally delegate to the
existing readout, preserving actual sparse arithmetic, legal actions and ties.
Do not recap the combined feature vector or modify base Q. Persist/hash actual
frequency matrices, phases, descriptor and pooled normalizations, C, base bundle,
source/scaler, calibration starts and records. Keep3ac/d7/eb/40069 immutable.

Use the SAME full-rally analytic actor objective: +1 per own hit over the complete
20s/loss horizon, two independent categorical attempts per start, two starts per
frozen four-episode batch. At each hit accumulate the prefix score eligibility;
subtract the independent same-start partner's TOTAL own hits times terminal
eligibility. Equal-total pairs can update. This estimator is unbiased before
caps but is not guaranteed lower variance than terminal paired credit. Only the
new output weights learn, using explicit categorical score algebra; no
backpropagation/autodiff, critic, teacher labels, or historical action locks.

Use fixed initial temperature .008161197682294006, previously calibrated to5%
expected nongreedy actions on the immutable2048-row TRAIN cache of40069scores.
Build the new256×256 empirical feature second moment once on the calibration,
equal episode/frame weighting, normalize mean diagonal to1 and add fixed .01I.
Precondition analytic output gradient with this matrix. Cap each aggregate step
at1% INITIAL centered40069-Q RMS=.008003329563010489 score RMS and mean
categorical total variation .01 on the saved new-feature calibration cache;
only fixed ray bisection enforces caps. Report actual greedy action dose, TV,
score movement and cumulative movement. Verify mapped/deployed score arithmetic.
Choose the FINAL completed batch endpoint, never best TRAIN checkpoint.

## Fixed collection, execution and outcome gates

One absolute600s experimental clock covers calibration, feature diagnostics,
training and native qualification. Implementation time is separate. Training ends
by300s from experimental start, at48,000NEW training transition cameras or24
four-episode batches, whichever comes first. Calibration is additional, at most
2,048cameras. Evaluation permits32TRAINscreen plus64paired-fresh native games
(up to96,000nominal20s transition cameras); total nominal upper bound146,048.
The600s clock dominates;64k was an earlier TRAINING ceiling, not a total-cost
claim. Count calibration/training/evaluation cameras and initials separately and
in total. Worker quotas enforce shared training-camera caps; partial training
batches are paid and never updated. No deadline reset, automatic extension,
replacement fit, feature retry or parameter sweep.

Calibration: at most2,048new cameras, eight predeclared TRAIN256starts, four per
phase, at most256transitions/start. Run frozen40069 and save the exact continuous
legal observations plus its scores. Ended scenes contribute fewer samples with
equal episode weighting. These observations calibrate only features and their
score metric; reward does not choose map/scales.

Training: at most48,000new cameras and24four-episode batches, within the300s
training subdeadline; uniform predeclared
TRAIN starts balanced by phase, independent declared action seeds. Final endpoint
then fixed32TRAIN greedy games, phase/depth-balanced using only40069baseline
receipts. Before collecting outcomes, persist calibration/training/screen starts
and genuinely unused32paired fresh starts, excluding ALL old TRAIN, fresh/probe,
actual receipts and reserved streams. No existing fresh admissions.

Screen requires completed net total AND later-return gains with retained first
and second-return counts and no lower later-return fraction. A tiny mixed result
is reported for parent review; it does not authorize a retry. Clear regression
ends the test. If useful, run the same frozen endpoint versus40069 on the32new
paired starts under the SAME remaining600s allocation. Save matched native
receipts and report total/first/second/later outcomes, losses, timeouts and paired
confidence intervals. Thirty-two pairs are a bounded pilot, not promotion proof.
Any incomplete stage is partial; never claim full or fresh qualification from it.
No automatic promotion. Larger fresh/full-bank and longer multi-rally/match
qualification remain necessary before claiming reliable match-long learning.

Focused correctness checks: permutation/ID invariance; empty/unknown handling;
continuous sub-bin position changes; distinguish two scenes with identical
pooled position and velocity marginals but different entity association; signed
ordered-pair descriptor check; batch/deployed features; zero-C exact actions/Q
including ties; analytic score finite differences; metric/step arithmetic;
unmodified base vision, W/R and map hashes. Rank/feature diversity/cost are
diagnostics, not nonlinear novelty or capacity proofs and not a map-sweep trigger.
Record entity/pair counts and inference latency to expose quadratic pooling cost.

## Why not bilinear pooling plus198DCT

Bilinear binning would preserve sub-bin first moments and smooth neighboring
coefficients, and is a reasonable smaller representation alternative. It still
aggregates entity positions and velocity channels into separate spatial sums;
different within-cell associations can remain aliased. Symmetric unconditioned
pair sums also lose which visual motion belongs to the pair's anchor. The
proposed entity/pair descriptors preserve those joint continuous variables before
pooling. The preceding198DCT test smoothed coefficient updates on existing HARD
bins; it did not test either continuous representation. RFF256 may still alias,
average away rare entities, generalize globally or fail credit learning. Its
selection is a concrete hypothesis, not a guaranteed improvement over bilinear.

This adds a feature source and residual branch: it is explicitly a MAJOR
architecture/input change, outside prior512TensorSketch and actor approvals.
Parent and independent review should resolve material corrections, then request
the user's approval for exactly this one bounded comparison before coding/running.


## Parent review and approval boundary

Parent reviewed the field whitelist, dimensions, anchored ordered-pair semantics,
empty-block handling, zero-correction delegation, learning rule and complete cost
budget. The only material pre-approval amendment was the normalization floor
above; feature construction and test remain unimplemented. This proposal is ready
for the user to approve or decline. Approval applies to this one bounded pilot,
not an open-ended representation, frequency or learner sweep.
