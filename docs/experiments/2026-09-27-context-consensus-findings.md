# Context repair and eight-step production promotion

The user authorized the investigative fix and promotion. All registered active
integration gates now pass. `load_default()` uses the eight-step predictor,
append-all four-step fallback, and consensus context corrections. The long bank
starts empty from the unchanged bundled prior; no held-out-trained checkpoint is
chosen as a new shipped prior. This is the M1A.5 substrate for the saved M1B plan.

## What caused the failure

The fixed 192-scene online run produced only four context-correction updates.
All were early period-four motion transitions; their key and timestamp routing
were consistent. Only one came from an unconfirmed track, so merely filtering
tentative tracks would not explain or address all updates.

The four new outcomes, ranging up to a correction norm of 3.162, become the
nearest examples for ordinary motion. They disagree about direction, but the old
rule averages them into a large definite shift. This is what corrupted carried
outage support, while identity routing and future mixtures remained correct.
The credit and retrieval audits are saved beside this report.

## Bounded fix

Keep the same local neighbors, weights, memory capacity and online observation
updates. Apply a weighted correction on an axis only when all selected outcomes
are nonnegative or all are nonpositive. When both signs occur, retain uncorrected
motion on that axis. Zeros are neutral. No magnitudes are clipped; agreed large
corrections remain learnable. No task labels, hidden targets or backpropagation
are introduced.

This is conservative local agreement, not calibrated statistical confidence or
arbitrary rotation equivariance. It can abstain on legitimate but conflicting
contexts. The forecast service continues to retain mixtures of possible futures.

## Evidence

| Check | Result |
|---|---|
| Outage support after online training | **172/180 (95.56%)**, up from 29/180; >=95% gate passes |
| Noisy / outage identities retained | 36/36 each; zero final extras |
| Cached outage forecasts | 900 evaluations; zero missing |
| Four/eight-sample forecast NLL | 2.485 / 3.009; exactly unchanged by repair |
| Nominal 90% marginal coverage | 93.54% / 89.85% |
| Aggregate, each-family, observation and calibration gates | Pass |
| Existing spatial / appearance contexts | 64/64 each |
| Ambiguity, visible contradiction, noisy blank input | Pass |
| Frozen held-out evaluation | Learned parameters unchanged |

The complete 192-scene training schedule was repeated with the repaired production
implementation, then saved/reloaded and tested on the original 216-scene transfer
split plus recovery/context/blank checks. No seed selection or parameter sweep.

Crucially, a separate context memory started **empty**. Appearance performance
went from 4/64 to 64/64 after 8,192 samples. After another 8,192 samples of the
spatial task, spatial performance rose from 0/64 initially to 64/64, while
appearance stayed at 64/64. Memory keys changed in both stages, ending with 256
examples. The repair preserves acquisition and sequential retention on these
tasks, rather than merely retaining a pretrained answer or freezing learning.

## Compatibility and delivery

Version-3 checkpoints explicitly identify the consensus rule; unknown rules are
rejected. Version-1 and original version-2 files retain their old predictor/context
rules. The explicit version-1 upgrade preserves learned tensors and resets only
the scene. The original artifact remains unchanged and accessible through
`load_legacy_default()`. Historical comparison scripts are pinned to that loader.

Independent review found no actionable correctness issues. An isolated installed
wheel loads the promoted default, updates active banks, processes a sensor sample,
and round-trips version-3 state without importing research scripts.
The final repository suite passes: **721 tests, four skipped**.

The integration JSON records the candidate's pre-promotion evaluation stage;
the default factory was switched only after those gates and fresh acquisition
checks passed.

## Online runtime and growth

One CPU thread, 64x64 frames plus events, learning enabled, full state plus sparse
interface encoding. Each arm runs 240 samples, timing the final 200 after warm-up.
No concurrent test workloads ran during measurement; camera acquisition is excluded.

| Visible entities | Median / p95 milliseconds per sample | Forecast-bank tensors, initial -> final |
|---|---|---|
| 2 | 76.6 / 165.8 | 203,760 -> 356,368 bytes |
| 8 | 300.5 / 462.0 | 203,760 -> 814,192 bytes |

This is **not 50 Hz real-time performance** on this CPU. Median encoding alone
is 5.9 / 25.2 ms. Storage figures count only short/long key and value tensors,
not total process memory, context, calibration or transient state. All three long
banks become active during measurement. Append-all storage and exact retrieval
continue growing; this short run does not establish indefinite scalability.
These remain optimization targets under the user's soft compute policy, not a
reason to discard the measured learning gains. Raw timing and bank counts are
in `2026-09-27-context-consensus-timing.json`.

## Limits and next step

Eight of 180 outage locations remain below the support criterion, even though
the aggregate gate passes. The earlier two of five strict retention failures
remain an accepted limitation; this repair does not waive or solve them. The
tests establish synthetic motion/context transfer, not lifelong cross-game or
natural-camera competence. Dense grayscale frames remain declared sensor input.

Keep this visual service and proceed to the saved M1B plan: the smallest online
reward learner over its generic structured/sparse interface. No new latent model,
neuralized replacement, or additional context-mechanism search is needed now.

Reproduction: `scripts/run_context_integration.py`,
`scripts/validate_consensus_learning.py`, and the `context-consensus-*` /
`consensus-context-learning` JSON results beside this report. The original
failed checkpoint remains preserved separately from the repaired integration run.
The repaired generated checkpoint is `checkpoints/m1a5/context-consensus-integration.pt`,
SHA256 `2d3428bc7621fa10aa2d1286a963ca5430d80560ee05898256b830ffd20abff6`.
