# Joint output learning: diverse rewarded batch

## Question and decision

The first joint update preserved all 192 practiced outcomes and improved total
returns from 493 to 523, but fresh returns fell from 73 to 65. It establishes
integration, not transferable skill. Parent and reused Sol selected one small
batch to test whether several complementary rewarded trajectories improve on
the single-target update. Repeated training-bank gains alone will not establish
success.

Use the 40069 joint checkpoint as a TRAIN-proven scaffold, not a fresh-qualified
replacement. Keep eb and all previous checkpoints. Choose three distinct targets
from existing TRAIN own-reward experience, spanning serve phase and return depth
where available. Do not select against the last fresh set or admit it into TRAIN.
The exact targets and compatibility evidence must be persisted before fitting.

## Unchanged model and learning contract

Frozen vision, fixed 512-feature pairwise map and existing input transport.
Learn only the two existing output paths, without backpropagation. Reuse the
first joint pass's fixed score metric and optimizer denominators by verified
hashes; no new metric copy, feature recalibration, or architecture change.

Retain all complete currently earned prefixes except each target's old prefix,
which is replaced with its strictly better, actually rewarded source prefix.
Validate the source policy and rewards. Reject incompatible exact input/action
contracts before fitting. Failure suffixes do not become action labels.

One bounded fit, no target/guard/ridge retry. Verify every original deployed
choice, then actual target games and a representative training comparison.
Advance useful candidates to all 192 games; any loss of a certified earned
outcome is a correctness failure. Reuse whole-game receipts where exact policy
equivalence proves validity. No active evaluator changes during execution.

## Decision after testing

Test an improved candidate on unused ordinary paired games against both the
original eb carry and the first joint checkpoint. Report first returns, second
return coverage, later opportunities and returns, and per-game changes. Preserve
uncertainty: a small fresh set cannot establish reliable match-long play.

If this batch again only improves practiced outcomes, stop treating additional
small replay fits as evidence of transferable progress. Reconsider training
coverage and on-policy experience collection before another such fit.

All preparation, fitting and native work share one 900-second execution budget;
implementation time is separate. A reusable prefix-state cache remains an
efficiency option, but does not delay this learning test unless its implementation
cost is justified. Longer continuous-match qualification follows strong short-game
return rates, not merely successful constraint fitting.

## Outside perspective

Parent checked the [Go-Explore primary paper](https://www.nature.com/articles/s41586-020-03157-9)
and [authors' repository](https://github.com/uber-research/go-explore). Remembering
and revisiting successful states can aid exploration; the repository separately
identifies deterministic exploration and robustification. Our existing branch
scaffold already uses a related idea. This is a reason to distinguish archived
success from transferable policy skill, not a new mechanism or permission to
import their backpropagation-based training.

## Selected batch and execution authorization

Zero-camera compatibility screen is saved at
`pong-rally-joint-output-20261028-r1/proposed-next-batch.json`:

| Seed | Serve phase | Current → rewarded target | Prefix rows |
| --- | --- | --- | --- |
| 2667357792 | left | 1 → 2 | 205 |
| 478956070 | right | 1 → 2 | 321 |
| 1212947804 | left | 2 → 3 | 375 |

Selection: shortest compatible new second-return prefix per phase, followed by
the shortest compatible third-return prefix improving on both the current policy
and named 3ac baseline. These are recorded model-earned trajectories, not expert
labels. Full source receipts are validated before slicing. Combined raw-input
alias screen found no conflicting actions.

Current complete earned memory has 80,333 rows. Replacing 395 old target rows
with 901 new rows gives 80,839 hard rows across 175 cases. Parent approved
implementation with one 60-second fit, the fixed saved metric, and a shared
900-second execution clock. Review code and focused tests before launch. The
fresh comparison uses the same 32 unused starts for all three policies (96 games)
if the remaining budget permits, with partial work reported honestly.

Defer production prefix caching: an eager build would reexecute 80,333 frames
and needs roughly 2.1 comparable passes just to repay camera work, before code
and storage costs. Whole-episode reuse remains enabled.
