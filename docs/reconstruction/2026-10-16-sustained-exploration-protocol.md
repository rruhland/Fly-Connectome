# Camera-rate hit learning with sustained exploration

This new, fixed exploratory study follows the user's instruction to prioritize
genuine reward learning. The preceding full hold10 intervention improved returns
relative to hold1 but also changed update frequency and macro credit geometry;
neither exceeded simple controls convincingly. This study changes only training
exploration correlation. It introduces no teacher labels, backprop, new policy
inputs, architectural replacement or final holdout use.

## Fixed behavior and learning comparison

Two policies start from zero, with the same new training seed **736187003**:

- **Independent:** at each camera, draw exploratory mode with probability .2.
  If exploratory, choose a uniformly random legal action; otherwise choose the
  maximum-Q action, resolving exact ties uniformly as before.
- **Sustained:** at each ten-camera block start, draw exploratory mode with
  probability .2. If exploratory, choose one uniform legal action and reuse it
  throughout the block. Otherwise recompute greedy action and exact tie choice
  at **every camera**, allowing immediate visual control.

This is a .2 probability per block, not a .2 per-camera probability of starting
a ten-frame burst (which would substantially increase exploratory occupancy).
Episode boundaries discard unused block tails. Truncation samples the final-state
bootstrap action before resetting; that unexecuted selection consumes RNG and
block phase, is logged separately, then is discarded with the old episode. Both
arms use the same rule. No block or random drive carries into a new episode.

The explicit block1 path preserves legacy gate/random/tie RNG call order and
identical update values. The unconfigured default retains its historical state
and record fields. Sustained exploration changes RNG consumption and later states,
so matched seeds do not imply matched trajectories after actions diverge.
State-dependent episode lengths can alter realized exploratory camera/time
fractions; report them rather than claim exactly 20% realized occupancy.

Both arms use hold1 and update true-online SARSA **every camera**, lambda .995,
alpha .001, epsilon .2 and physical gamma .95/second. Action/bootstrap selection
remains before the preceding TD update. Selection metadata is owned by its pending
action: record current executed mode/action/block phase and strictly non-greedy
status at its original pre-update selection time, plus separate bootstrap
metadata. An exploratory action can coincide with a greedy action; equal-valued
ties are not classified as non-greedy. Log actual mode and action run lengths.

Exploration has hidden mode/phase not present in the Q features. This behavior
memory is an explicit residual aliasing caveat, so the intervention does not prove
a pure Markov-factor or general exploration theorem. Frozen evaluation transfers
only mapped weights/cap and uses camera-rate greedy actions in both arms.

Physics and representation remain fixed: .65 opponent, 2400-tick/20-second training
episode limit, exact prior 214×4389 generic score-free DCT transform and cap256.
The visual prior's long banks remain empty, with four-step context fallback.
Learning reward is **+1 per own-paddle hit only**; game-score coefficient and
survival reward are zero. Raw terminal scores/end flags stay separate. Contact
ground truth enters only scalar reward, never policy inputs. No reward clipping,
hit thresholds, teacher targets or optimizer changes.

## Fixed allocation and fresh evaluation

Exactly **24,000 new camera transitions per arm / 48,000 aggregate**, from zero.
The budget doubles the previous arms' camera exposure; those arms produced only
22/29 hit events. Contact counts and adequate asymptotic exposure are not promised. Save/evaluate only the
24k endpoint. No intermediate DEV evaluation, checkpoint selection, seed replacement,
automatic continuation, replication or final evaluation.

Two isolated single-threaded training workers have **20 shared training minutes**;
the whole allocation has **30 minutes**, anchored at actual manifest creation and
never reset on resume. Four persistent fresh-evaluation workers; reserve the final
120 seconds for physical annotations/reporting. Retain failed/partial work with
no extension. Existing finite guards and max(|q|, |q_next|, q_spread)>1000 numerical
resource stop remain. Maximum fresh-evaluation allocation is 84,000 transitions
(seven policies × 24 serves × 500 cameras). Prior measured throughput suggests
roughly 18–23 minutes overall, not a guaranteed completion time.

NEW independent DEV serves: NumPy default_rng seeded by
`SeedSequence([691071257,29072])`, 24 integers in `[1,2147483647)`, in fixed order.
This recipe is fixed before expansion; no serve inspection or selection. Each job
starts its own one-episode evaluator, constructor serve, phase zero, 1200-tick /
ten-second limit. Both trained greedy policies use hold1, raw-score-only evaluation
and no learner update. Controls on the same serves are initial/constant down,
constant up, no-op, uniform random hold1 and uniform random hold10. Their per-serve
random seed is the first uint32 from `SeedSequence([447201113,691071257,i])`.
Observation-independent controls may use the validated perception-skip path;
its runtime is not inference-speed evidence. Final streams
308713321/1203078761/1983461927 remain unexpanded and unused.

## Outcomes, inference and integrity

Reuse deterministic raw-action physics replay: an own opportunity is an actual
incoming left collision-plane crossing; success is the own hit. Opponent-going
serves are not own opportunities until returned. Co-primary diagnostics are
first-incoming return success per serve and all returns/all incoming opportunities.
Verify exact first-incoming ball/velocity/tick equality across all policies.
Report ≥2 returns/rally, longest streak, returns/physical second, durations,
scores/losses/timeouts/censored approaches and opponent reliability, including
after own returns. No natural-win or survival substitution for return skill.

For each learner, retain the existing promising-screen rule: at least 20 percentage
points above the strongest of all five simple controls on **both** co-primary
metrics, with both paired 95% intervals above zero and zero invalid draws.
Bootstrap 10,000 times using seed193870291, resampling the 24 serve clusters and
recomputing the maximum control within each draw. Additionally report the direct
**sustained-minus-independent** paired difference and intervals on both metrics.
This intervention comparison is descriptive and not multiplicity-adjusted definitive
evidence. One training seed is exploratory regardless of the screen outcome.

Exploration block state, pending selections, RNG, traces and weights are durable.
Retain per-camera commits, bounded snapshots, full actual-recovery validation,
existing report/journal/endpoint completion pins and zero-work verification.
Narrow checks cover block1 legacy equivalence, held exploratory versus responsive
greedy actions, selection-time metadata, actual partial-block boundaries, exact
crash replay and worker→mapped greedy evaluator handoff. Archive source, protocol,
runtime, transform, manifests, all records and costs. Count diagnostic reexecution
separately. Parent independently audits physics, rewards, TD, exploration continuity,
first-incoming pairing and statistics. No further allocation follows automatically.
