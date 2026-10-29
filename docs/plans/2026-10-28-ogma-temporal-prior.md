# Learned temporal representation with a carried policy prior

Authorized under the user's temporary architecture-approval waiver. Parent and
independent Sol reviewed the implementation before launch. One600s experiment,
final endpoint only; no temperature or architecture sweep.

Existing camera, vision, capped transport and40069base remain frozen. Current
7×16×16visual planes feed an additional OgmaNeo2 image encoder8×8×16 and three
predictive temporal layers8/4/2×16 (temporal horizon4,ticks2). Older source pinned
c5be9ec41d26aeeb2fc805961140781c851542cf; inspected encoder/predictor/linearactor
use local updates, no hidden-layer error propagation. Do not substitute current
AOgmaNeo, whose nonlinearactor has derivative propagation. Vendored dependency
retains separate CC BY-NC-SA4.0 license and attribution.

Up to90s prediction-only warmup uses chronological savedTRAIN observations,
including actual past actions as temporal context, no actor reward updates or
expert action labels. Report actual rows/executions and any final partial sequence.
Then freeze representation weights and train one carried actor for300s using
only its actual sampled actions/probabilities and own-hit rewards. Runtimehistory
resets eachrally; learning weights persist. C++ local action output starts atzero,
added to fixed40069Q/T, T=.008161197682294006. Greedy zero-correction uses literal
float64baselinechoose, preserving ties/near-ties. Finalgreedyevaluation8→32→
conditional32freshpairs consumesremainingtime, atmost180s. Initial/intermediate/
finalcheckpoints retained; no best-checkpoint selection.

Approved adaptation: actor signedTD replay uses recorded(previous-action onehot
minus previous behavior probability), exact .95**actualdt discounts, terminal
bootstrapzero AFTER accumulated rewards, zeroTD giveszero policy update. This is
heuristic replay actor-critic, not unbiased current-policy gradient. alpha=beta=.02,
history128,minSteps8,historyIters8. No offpolicy actor warmup. Representation-only
and fullweight fingerprints distinguish learning from unintended mutation.

Focused checks include C++ hand-calculated reward/probability/state/terminal
alignment, x/y/channel packing against actual address3, zero-correction baseline
parity including near-ties, save/load/reset, and representationfreeze duringlocal
learning. First actual onlinegame will replay fromonlineinitialcheckpoint including
actualsuccessorflush; nativeevaluation replays exact chronologicalstate/action/Q.

Known limits before outcomes: saved2048calibrationstates have only1.30%average
stochastic nongreedy mass;414rows have exactlyone-hot float32probabilities andzero
score correction,1674have pmaxrounded1. StockminSteps8 leaves final8decisions
unreplayed before reset. These may limit credit; do not infer representation
impossibility from a null result. No adjustment selected afterseeingoutcomes.

Artifact `/workspace/m1b-experiments/pong-ogma-prior-20261028-r1`.
Approved spec SHA30ddfb996eec73511672b57a5ae0c070116c0ebb8e877023d1e66b23be184067.
Source/library/inputhashes pinned. No reliable-return claim before fresh first/later
>=95%, enough lateropportunities and independenttrainingreplication.
