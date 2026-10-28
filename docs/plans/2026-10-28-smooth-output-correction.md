# Smooth corrections to the existing policy weights

## Decision and scope

The full-coordinate online actor regressed on both trained and other starts:
TRAIN256662→566returns, including151→141on its64training starts. Parent proposed
restricting updates to smooth spatial directions, and Sol revised its initial
preference for a new input representation in favor of this smaller intervention.

Keep40069 exactly as the warm reference, including its nonlinear residual. The
policy receives identical vision inputs and runs the identical raw4389+fixed512
readout. Only the TRAINING update coordinates change: the lowest3×3orthonormal
DCT modes on each of22existing spatial planes,198features/396contrast coefficients.
Keep nongrid coefficients and the512residual weights frozen. Map corrections back
into the existing linear weight array; no extra runtime branch is introduced.
This is an optimizer restriction within the existing architecture, not a new
architecture requiring user approval.

No novelty claim: older DCT actors used first-hit rewards and much weaker policies;
the spatial-neighbor prior used trajectory fitting. This pass tests smooth,
lower-dimensional corrections from full-rally rewards around the current stronger
policy. It cannot fix within-bin aliasing and does not remove stochastic-training
versus greedy-deployment mismatch. Failure is not a representation-capacity proof.

## One fixed comparison

Use the same full-rally analytic actor objective, initial temperature,2048cached
TRAIN observations,64TRAIN seed contexts,128action seeds and two step caps as the
completed actor. Choose the FINAL completed endpoint only. No rank, temperature,
margin or step-size sweep. Same maximum32batches/64k training cameras.

For fixed basis B and old raw optimizer denominator d, map the correction into
joint optimizer coordinates with J=[B*d,0]. Use H=J G Jᵀ, retaining exact units and
the existing regularized score geometry. Actual actor features are Bx. Test score
derivatives, mapped ΔQ, projected metric equality, residual/nongrid invariance,
and the deployed step caps. No post-hoc filter of an unrelated full-space update.

Before any native outcomes, select a fixed128game evaluation bank: all64scheduled
training starts plus64other bank starts balanced by phase and baseline return
depth. Report trained/other strata separately. This replaces repeated full256
checks to reduce evaluation time while improving coverage over the noisy32screen.
Use a new900s clock; training at most400s reserves500s for evaluation. Partial
attempts count, incomplete batches do not update. A useful frozen128result may
proceed to a new32start paired fresh evaluation; mixed results stop for parent
interpretation, without refitting or automatic promotion. Existing fresh sets
remain excluded from training.

Implementation target10minutes. Reuse the actor and qualification helpers, retain
completed experiment sources, and review concrete code/spec/tests before launch.
If this correction also regresses broadly, stop this family and prepare a concrete
generic continuous visual representation proposal for the user's required review.


## Reviewed and launched

Agent88e4ee8 integrated as main805a976.17focused tests, compilation and diff
checks passed. Parent review and the independent layout test caught an incorrect
nongrid cutoff before launch: future_outside3840:3848 andmetadata4381:4389 are
frozen; context4373:4381 must remain eligible for smooth updates. The corrected
code derives the mask from the basis, not a hand-entered cutoff.

Root `pong-rally-smooth-actor-20261028-r1`, specSHA
`e945f4ac8ca581cc440b33365df544e0971d4758a8326bec888fbb9a68a0ae9a`;
freshTAG63089. New900s clock,400s training bound. Source frozen. Firstbatch at
17.82s used2,039cameras and rewards[2,5,1,1], exactly matching the prior actor.
Parent independently verified all raw observations/actions/probabilities/hits
bit-for-bit and projected old full-space eligibility/credit into the new198basis;
maximum error9.1e−13. Zero extra observations,0.038s computation. Artifact
`parent-first-batch-projection-audit.json`. No performance claim before evaluation.


## Completed outcome — rejected; actor family stopped

All32batches completed in275.19s with59,420training cameras+128initials. Final
reference-cache movement was0.0245939RMS (3.07%initial),104/2048greedy choices
changed. R remained byte-identical, nongrid weights exact, all coordinate/cap
checks passed. No partial training batch.

The128screen regressed287→237returns,106→91first,67→61games≥2,
later181/262→146/222 (69.08%→65.77%). Each64-game stratum lost25returns:
trained151→126 (first56→49,second34unchanged), other136→111
(first50→42,second33→27). Seed104562292 improved5→6 despite the aggregate
regression, confirming why that single example should not control acceptance.

Normal completion435.28s; total114,966new transition cameras+250initials.
Screen122new games/55,546cameras+122initials, six exact episodes reused.
All128training and128screen receipts complete; final audit passed, no unknown
work. No fresh63089or full256 evaluation, no promotion. Candidate bundle
`dc33740b1c4a3471264df162772c596fa742fda0018f9ff295c43d7edc6c47f0`,
W `d56950cf629f03e3cf492ceefcf96645b0c43bdbdcf9072c9148436b12c90fab`.

Keep40069 and stop this actor-variant family. Next work is a concrete generic
continuous entity/pair representation proposal, requiring the user's major-change
approval before implementation. These failures do not prove a capacity limit:
credit variance and stochastic-training versus greedy-deployment mismatch remain
plausible. No architecture implementation or native experiment is authorized by
preparing that review document.
