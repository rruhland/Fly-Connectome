# Proposed production integration: eight-step motion memory

Status: user approved integration and promotion on 2026-09-27, including the
two-seed retention exception. Integration is implemented as an explicit upgrade,
but default promotion is blocked by an unexpected online context-memory regression.
See the [integration findings](../experiments/2026-09-27-context-integration-findings.md) and
[execution ledger](../experiments/2026-09-27-context-integration-protocol.md).
The user requires approval before production architecture promotion. The completed
[five-seed findings](../experiments/2026-09-27-long-context-findings.md) support this
as the next integration candidate. Implementation/promotion is contingent on review
of those results and the retention exception below: two of five seeds exceed the
0.2-nat limit, despite better absolute prediction and useful shared-dynamics transfer.

## Concrete change

Use the tested eight-step conditional forecaster with a fully retained four-step
fallback. Both banks append actual observed endpoints; neither deletes examples
during online learning. This changes the current production reservoir policy as
well as adding longer context. It does not add a recent-bank weighting scheme.

The long bank starts empty, never with invented eight-step histories derived from
old four-step keys. The existing bundled prior initializes the short bank. After
32 long-bank outcomes, nine consecutive observed positions enable the eight-step
forecast. Otherwise, use four-step prediction after five consecutive observations.
Missing evidence cannot become a new motion measurement. Retain cached issued
forecasts and their evidence ages through outages as in the approved baseline.

Use the same local normalized-history nearest-neighbor outcome mixtures, Gaussian
components and calibration rule. Credit retains the actual issue-time history and
distribution. The fallback keeps learning all eligible real endpoints; the long
bank learns only endpoints for which eight-step issue-time history existed.
No backpropagation, game identities, collision labels, controller semantics, or
neural replacement of the working observation system.

## Approved integration work

1. Move the tested conditional predictor into the installed vision package without
   research-script imports. Retain the legacy predictor for old checkpoints.
2. Make live history collection support nine positions and use the same contiguous
   history suffix for issuing a forecast and exporting its mixture/intervals.
   Preserve identity routing, outage behavior, scene reset and pending credit.
3. Add a versioned checkpoint representation for both banks and issued-mixture
   calibration. Version-1 checkpoints must remain readable with their original
   behavior; an explicit upgrade initializes an empty long bank. Following approval,
   `load_default()` can apply that upgrade to the existing bundled prior. Do not
   choose a pretrained candidate by its held-out seed score.
4. Keep the structured and sparse central interfaces unchanged. Preserve every
   mixture component; only forecast quality and underlying temporal memory change.
5. Verify full camera-loop versus experiment parity after the long bank is active,
   including noise, multiple entities, outages and resets. Test save/load of learned
   short/long banks and calibration into a fresh scene. Run the existing context,
   recovery and probabilistic-regression checks on an actively trained candidate,
   not only the trivial cold state in which long-bank prediction is inactive.
6. Measure online two/eight-entity latency and memory growth. Appending forever has
   growing retrieval/storage cost; record it as an optimization target. Do not
   substitute pruning or a different retrieval rule simply to meet a hard budget.

## Acceptance and remaining limitation

The longer-context experiment measures a useful learned forecast, not a general
causal/action-conditioned world model. Dense grayscale frames remain part of the
declared sensor. The pretrained credibility observer is not retrained online.

The original <=0.2-nat immediate-retention screen remains binding unless the user
explicitly accepts its failures as a tracked limitation for this production
upgrade. Better absolute prediction is not a pass on that separate screen.
If the user accepts that tradeoff, integrated regression checks still must pass
before default promotion. Do not declare lifelong retention, cross-game competence
or M1B motor learning established by this experiment.

After integration, proceed with the existing M1B roadmap and its simplest online
reward learner. Preserve this visual substrate rather than restarting a large
latent-world-model search. Further memory/context work should answer a concrete
failure exposed by the integrated system, not continue a history-length sweep.
