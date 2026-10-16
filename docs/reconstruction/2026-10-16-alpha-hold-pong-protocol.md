# Paired step-size versus control-cadence development study

Both fixed30k capped endpoints were negative: raw9/24wins(one own-hit), block-RMS
8/24(zero own-hit), versus strongest simple control10/24(two own-hit). The frozen
epsilon.2 diagnostic also gave8/24(zero own-hit) for both state-dependent raw and
calibration-mean-Q bias-only policies. Preserve these failures. Use the simpler
**raw+cap256** representation, not the one-win development difference as evidence
for a superior representation. No final holdout has been used.

Two prospective, from-zero arms share training seed172900127 and **30,000 camera
transitions each**, with the same frozen visual prior, raw generic transport,
fixed cap configuration, .25 opponent, legal actions, score-only±1, 50Hz camera
schedule and2400-tick rally limit:

| Arm | Alpha per learner update | Frames per action | Lambda per learner update |
| --- | --- | --- | --- |
| Lower step size | .0001 | 1 | .995 |
| Held action | .001 | 4 | .995^4 = .980149500625 |

Gamma is.95 per physical second. Epsilon.2 is drawn once per actual action choice,
not on held intermediate frames. All camera frames still update visual state and
receive **their own durable journal commit**. Keep camera, actual executed choice
and learner-update counters separate. Held action phases, pending features/action,
accumulated discounted reward/duration, traces and RNG survive every camera
snapshot/replay. Terminal or truncation ends a partial hold immediately; explicit
reset follows the genuine final camera. A final budget/checkpoint may land mid-hold;
save its actual valid weights and pending phase without taking extra cameras.

For a macro containing camera rewards r_i and elapsed times d_i, use
R=sum_i gamma^(sum_{j<i}d_j)*r_i and outgoing discount gamma^(sum_i d_i), zero on
true termination. Truncation bootstraps the genuine final observation and then
resets traces. The incoming trace discount spans the prior macro. Lambda is
camera-count matched via.995^4; physical discounts use actual2/3-tick durations.
No contact shaping or privileged labels enter either arm. This is a **whole
cadence intervention**: persistence, exploration/update frequency and Dutch-trace
geometry change together. A win would not isolate physical persistence alone.

The lower alpha tests an untried .25 step size and possible variance/common-bias
swings. The held arm tests stronger coherent consequences across the80ms motor
activation scale, despite initially subpixel displacement. Current bias and sparse
local credit, tracker corruption and limited exposure remain competing hypotheses.
This finite pair can expose large practical improvements; failure at this budget
does not establish asymptotic inability. No additional modifier arm is allocated.

Save6000/15000/30000-camera checkpoints descriptively. **Evaluate only30,000**,
with fresh histories on the same24development serves904271833 and20s limits.
Frozen evaluation is greedy at each action boundary; held evaluation retains50Hz
vision and chooses only every four cameras or after a rally reset. Initial/down,
constant-up and no-op are physically invariant to holding; reuse their existing
24-rally results only after exact action/physics/camera projection replay. Reuse
the old random control for hold1; newly evaluate random hold4 with seed447201113.
The same serve/reset semantics apply to every policy. Label skipped-perception
baseline timing separately from learned inference timing.

The total-win threshold is **max(14, maximum matched-control total wins +4)**.
Independently, the own-hit-win threshold is **max(6, maximum matched-control
own-hit wins +4)**. An arm must meet both. The two maxima can come from different
controls; a held-random control can raise either threshold alone. Truncations are nonwins; report
all losses/timeouts/completion, contacts, rally scores/durations and physical-time
returns. Compare complete allocations only. Among qualifying arms choose most
total wins; ties select the simpler **lower-alpha hold1**. No early checkpoint
selection or post-endpoint threshold adjustment is allowed. No automatic
independent replication follows; that would require its own frozen finite design.

Numerical abort remains nonfinite update or a committed record with
max(|q|,|q_next|,q_spread)>1000, ignoring absent target fields on nonupdate cameras.
Preserve failed/partial allocations and last valid journals. Do not lower alpha,
replace a seed or substitute a checkpoint after failure.

Maximum **60,000 learning camera transitions +86,400 fresh evaluation camera
transitions** (two trained24-rally policies and one random-hold4 control, each
conservatively bounded at1200cameras/rally), **35minutes shared elapsed wall time**.
Use two independent learning workers and up to three evaluation workers, each
Torch/BLAS threads1. Bounded200-camera chunks, at most five chunks/interpreter,
160s work/180s kill bounds and full per-camera SQLite durability remain unchanged.
Expected paired learning18–21minutes plus roughly2–3minutes evaluation; record
actual overlap, summed process costs and all counters. Source/runtime/config,
complete source archive, baseline proof, hold1 equivalence and fresh-process
mid-hold replay evidence must be pinned before launch. Default hold1 preserves
old records, weights and source-normalized complete state. Empty long banks and
four-step frozen forecast fallback remain explicit. Final seeds308713321,
1203078761,1983461927 remain unused.
