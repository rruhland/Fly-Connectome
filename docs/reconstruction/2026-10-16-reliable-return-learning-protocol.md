# Reliable-opponent hit-only return-learning screen

User authorized the next return-learning/action-duration investigation. This is
a NEW task objective, not a claim to beat original score-only Pong. No backprop,
major architecture change or final holdout use. The original .45 scoring ceiling
and all prior negatives remain intact.

Parent's24-independent-seed physics-only probe found .45opponent224/239returns
(15misses, all same three initial serves across five policies). At.65 the five
policies saw258/258opponent returns and no natural wins. Legal tracking hold1
returned139/139own opportunities,24timeouts; hold10 returned60/81,21losses and
3timeouts. Scalar camera replay matched the batched probe. These finite samples
justify choosing.65, not claiming perfect reliability globally. The new main
cohort will measure opponent reliability, including after own returns. A separate
2048-camera .65vision check retained99.80–99.90%observed/predicted ball coverage
within3px; short samples do not rule out rare long-run corruption.

## Fixed training comparison

Two from-zero arms, SAME seed172900127: hold1 versus hold10, EXACT12,000 camera
observations each/24k aggregate. Both use the already frozen214×4389generic DCT
map, post-projection cap256, zero weights, true-online SARSA alpha.001, epsilon.2,
gamma_per_second.95, camera-lambda.995, .65opponent,2400-tick/20s training limit.
Hold10 uses macro-lambda.995^10, exact elapsed physical discounts/discounted reward,
and early score/truncation macro boundaries. Vision and durable commits remain
per camera; macro choices/updates are counted separately. Same camera exposure
means~10×fewer updates for hold10, an explicit confound alongside persistence and
exploration coherence. No pure one-factor causal claim.

Learning reward is ONLY+1per OWN-paddle hit; game-score coefficient0, survival0.
Raw±1game score and zero score contribution are still logged separately. A score
still ends the rally, but never pays or penalizes this return-count learner.
No clipping/fading or opponent-hit bonus. This isolates controllable return credit
from terminal scoring against a strong opponent. Ground-truth contact enters only
the declared scalar reward, NEVER policy features. Existing finite guards and
max(|q|,|q_next|,q_spread)>1000 resource stop remain. Partial/failed seeds are
retained, never replaced or extended. Save/evaluate only complete12k endpoints.

## Fresh evaluation and controls

24NEW independent fresh-phase development serves: NumPy default_rng seeded by
SeedSequence([104729113,29071]),24integers in[1,2147483647), fixed order, no scene
inspection/selection. Recipe is fixed in source before its first expansion.
All final streams308713321/1203078761/1983461927 remain undrawn. Each serve is
its OWN one-episode FreshPongEvaluator, phase0, max1200ticks/10s. Constructor serve
is used without an extra reset. Evaluate both frozen greedy endpoints at their
training hold duration, raw game score ONLY. No learner update or hit bonus in
evaluation. Owned mapped W=theta*T and cap hashes remain fixed.

Same24serves for down/initial,up,no-op and uniform-random hold1 AND hold10 controls.
Per-serve random-action seed is the first uint32 from
SeedSequence([447201113,104729113,serve_index]); both random cadences share it.
Constants need one physical trace each since holding a constant action is
physically identical. Observation-independent controls skip perception using the
already validated path; their timing is not inference-speed evidence. Legal
privileged tracking is feasibility only and never a candidate policy or weight
initializer. No teacher labels/coefficients enter this study.

All metrics are independently computed by deterministic physics replay of saved
actions; every raw reward, hit, elapsed time, physics tick and end flag must match.
An own opportunity is an ACTUAL incoming left collision-plane crossing, using
prestep x>=left_plane and float32(x+vx*dt)<=left_plane with vx<0. Success is the
own-hit indicator. A missed crossing counts immediately, before later−1score.
Opponent-going serves are NOT own opportunities unless subsequently returned to
the left plane. Right-plane opportunities and velocity reflections measure
opponent reliability, with a separate post-own-hit subset.

Co-primary diagnostics: (1) first incoming-ball return success per serve and
(2) own returns/all incoming opportunities. Before our first contact the ball
trajectory is identical across policies; verify its prestep ball/velocity/tick
exactly. Serves without an own opportunity are reported/excluded, not successes.
Also report≥2returns/rally, longest consecutive-return streak, hits/physicalsecond,
natural wins, scores/losses/timeouts/durations and unfinished left/right approaches.
Truncation before a plane crossing is censored and reported, not a fabricated
opportunity or successful return. No-op/random natural wins cannot inflate return
skill. Multiple contacts in a serve are correlated, not independent trials.

Paired serve-cluster bootstrap10,000draws, fixed RNG193870291; recompute strongest
of all five simple controls within every draw, separately for both co-primary
metrics. A promising development signal requires≥20percentage-point lift on BOTH
metrics and BOTH paired95%lower bounds above0, with no invalid draws. Report all
arms/ratios/intervals regardless. One training seed remains exploratory, not final
acquisition acceptance. No best-seed selection, replica/holdout or continuation is
automatic, even if a screen passes.

## Bounds, persistence and recoverability

12shared minutes training/22minutes overall, anchored at actual manifest creation,
never reset on resume. Two isolated single-threaded training processes; four
persistent spawned evaluation workers. Reserve final120seconds for physics-only
metric replay/reporting. Max24k NEW learning +84k evaluation camera transitions
(seven policies×24serves×500). Only complete training endpoints get evaluation;
other arms/controls still proceed. Wall exhaustion preserves partials, no extension.

Reuse≤200-camera chunks/≤5chunks per worker call,≤160s work, complete per-camera
commits and full verification on actual recovery. Report/semantic journal/
theta/W/cap completion pins remain. Verification-only recovery rejects an
incomplete actual session BEFORE any new transition. Backward-compatible optional
score coefficient leaves old reward specs/record fields unchanged. No new durable
class, policy input, tracker or digest optimization. Pin source/runtime/protocol/
helpers/transform/config; retain complete source snapshots, counters, records and
costs, including diagnostic/probe/replay work separately from candidate exposure.
