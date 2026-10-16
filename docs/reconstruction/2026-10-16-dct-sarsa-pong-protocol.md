# Frozen generic DCT coordinates with true-online SARSA: three-seed .25 screen

The raw/capped, RMS/capped, lower-alpha and hold4 SARSA studies failed their fixed
development screens. Score-only policy search and its fixed extension also failed;
the stationary→.25 curriculum ended at13/24wins(5own-hit), below14/6. Preserve
every failure. A quarantined oracle witness supports camera-only action control
in the already frozen restricted class, but does not establish Bellman-Q capacity.
This new study tests whether generic spatial sharing/conditioning helps the
existing SARSA learner, without changing its credit rule, reward or observation.
It is not another policy-search extension or an oracle-initialized candidate.

## Fixed representation and learner

Reuse the EXACT float64 `search-transform.npy` already fixed before policy search:
214×4389, file SHA256 `e2f2d132f553454fac7730d4ca98c683a16cab77c26b9c3b23eb1c1eb2d8b038`,
array SHA256 `a0b1872ba6432d6a4240fc233fadd567e9616c95d4d04d241c154c16f5b311e4`.
Lowest3×3DCT modes on every generic spatial plane and all nongrid coordinates,
score-free centering through the retained frame-present bias, supported-rank
projection and ridge.01 are unchanged. The fixed fit used1024 previously observed
generic feature rows, no reward, labels or oracle coefficients. The cache's
ORACLE-DIAGNOSTIC directory name does not make its separate generic sparse
indices/values an oracle input. No new calibration is collected or refit.
Supported rank183 gives549 supported scalar Q coefficients across three heads
and366 action contrasts; storage remains3×214. SARSA may learn a shared state
value, unlike the zero-head-mean policy-search parameterization.

At the learner boundary compute z=T*x and THEN apply fixed positive radial
squared-norm cap256. Preserve full raw transport, frozen tracking and forecasts.
Centering can fill an individually missing coordinate when frame-present is
nonzero; all-zero input maps exactly to zero. This is a restricted/conditioned
Q-function class and changed TD geometry, not an invertible full-class rebase.
The cap changes value approximation while preserving greedy argmax for fixed
weights. Calibration transformed norm² min/median/p90/p95/p99/max was
5.30/14.87/39.10/46.87/105.16/233.63;0/1024 rows activate cap256. No cap or step
size is selected against reward outcomes.

One frozen condition, **three prespecified independent from-zero training seeds**:
172900127,736187003,1597334677. Each receives EXACTLY30,000 camera transitions,
alpha.001, lambda.995, gamma_per_second.95, epsilon.2, hold1, zero initial weights,
opponent speed.25,2400physics-tick/20s episode limit. Alpha.001 is a fixed first
screen, not an effective-step-size match to raw coordinates. The existing analytic
true-online SARSA update and physical discount remain unchanged; no backprop,
hit reward, teacher label, controller state or physics input enters the learner.
Terminal scores are±1; truncations generate no score and retain existing bootstrap
semantics. No seed replacement or extra exposure follows weak/partial outcomes.

## Persistence and independent endpoint evaluation

Every camera still updates vision and commits one durable sample. Own the exact
T as an immutable FrozenCodec tensor anchor and scan its complete content in every
state fingerprint. Snapshots embed/resolve it; recovery verifies content/config
without needing the old file path. No factorization, cached digest or weaker
recovery check is introduced. Pending features/traces stay in214 learner
coordinates. Default unprojected session/checkpoint behavior remains unchanged.

Export only each30k endpoint: reduced theta, mapped full raw weights W=theta*T,
and exact T/theta/W/config hashes in the evaluation cap metadata. Same-boundary
missing exports can be rebuilt from the exact durable checkpoint; missing or
inconsistent past artifacts fail closed. Evaluation uses a fresh frozen ordinary
linear policy with W and a positive raw radial cap. Its Q scale differs from
training, but the two positive scales preserve greedy decisions; exact mapping,
blank input, mutation detection, crash replay and worker→fresh-evaluator handoff
are prelaunch checks. All meaningful live/frozen learned-hash checks remain.
Completion pins cover report bytes, all semantic journal tables and projected
endpoint artifacts. Exactly30k journal rows/report/consistent theta/W/cap are
required before DEV eligibility. A complete report lacking its pin must reopen
and verify the actual session with zero new samples; JSON alone never establishes
completion. Resumes validate existing pins. Timing-only recovery events do not
change the pinned semantic trajectory. One arm's failure does not replace its
seed or prevent other complete endpoints from evaluation.

After all training workers finish or reach their fixed bounds, independently
evaluate every completed30k endpoint on the SAME existing24-rally DEV stream
904271833, standard.25 opponent,20s limit, full vision, deterministic greedy
hold1. No earlier checkpoint receives DEV evaluation or selection. Reuse the
four verified simple controls (down/initial,up,random,no-op) after exact source-
bridged action/physics/reset replay. The zero-Q projected initial maps to W=0 and
chooses down, so the existing down/initial control is its exact initial baseline.
There is no random ARS initial-policy baseline in this study. Random action seed
447201113 remains fixed; matching controls are counted only at their original run.

Each endpoint must reach independently computed maxima:
total wins≥max(14,max control total wins+4)=14/24, and
own-hit-followed wins≥max(6,max control own-hit wins+4)=6/24.
Promote this CONDITION only if **ALL THREE** prespecified endpoints pass BOTH
gates. Report every seed/outcome/partial; never select a best seed. A failed arm
does not block evaluation of other complete endpoints. These shared24 serves
are24 paired clusters, not72 independent trials. This is a development screen,
not final acquisition acceptance. All final seed streams remain untouched.
No automatic further replica, blank-camera allocation or final evaluation follows.

Report all rally rows, wins/natural wins/own-hit wins, losses, timeouts as nonwins,
completion fractions, score/rally, return per physical second, actual durations
and contacts. Training feature/policy norms, finite weights/Q and learning curves
are descriptive; no outcome changes the fixed endpoint or budget. Preserve
nonfinite/resource stops: existing finite guards reject before commit; stop if
a committed max(|q|,|q_next|,q_spread)>1000. Discard a failed in-memory session and
recover the last valid journal; no in-place rollback is promised.

## Finite allocation and cost

Three single-threaded processes run concurrently within the four-CPU quota.
Maximum90,000 NEW learning cameras and86,400 fresh evaluation cameras (three
24-rally allocations at conservative1200cameras/rally). **30 shared minutes for
training,35 minutes overall**, anchored durably at manifest creation and never
reset on resume. No wall extension. If a seed cannot reach30k, retain its partial
allocation and do not substitute an earlier endpoint. Evaluation is capped by the
overall deadline; partial evaluations stay partial and cannot qualify.

Reuse the tested warm-session worker:≤200cameras/chunk, up to5chunks/process,
≤160s work/180s subprocess kill, complete per-camera commits and chunk snapshots,
full verification on actual opens. Failed writers remain poisoned. Source, runtime,
protocol, matrix, calibration/fit, baseline proof/report bytes and all checkpoint
artifacts are pinned, alongside full source archives and compute costs.

Bounded real-T preflight took27.17s: raw128-step mean3.740s vs projected4.705s,
1.258×cost/27.2 early-state cameras/s. Naive30k extrapolation18.4min; scaling the
historical full raw30k allocation gives25.5min, motivating30min training. Three
workers may contend on commits/history, so this is not a runtime guarantee.
The preflight executed640diagnostic cameras+23existing replay, zero candidate data;
raw256records matched archive, projected duplicate256records/full graphs and
fresh-process128records/state/chain/final graph matched. All preflight/test and
baseline physics replay costs are reported separately from candidate exposure.
