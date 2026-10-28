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

## Completed outcome

Source agentdf9c09e/main3a5600a reviewed;27focused tests, compile and diff checks
passed. Root `pong-rally-joint-batch-20261028-r1`, implementation estimate420s
separate. Preparation14.14s; fit33.26s verified all80,839original choices with
maximum actual margin violation4.04e-15. No fallback or retry.

Actual targets earned3,6,6returns versus1,1,2; trained prefixes required2,2,3.
Representative32 improved64→80total,24→25first,16→19games≥2,40/59→55/72later.
Six gains, no losses, including three non-target gains. Full192 completed:
523→546total,175→176first,123→129games≥2,348/482→370/501later (73.85%).
All192earned-outcome invariants passed;11gains and no losses, original104five-hit
skill retained. Native full stage145newgames/90,604cameras+145initials,47reuses
including10eb alternatives. Cumulative before fresh108,049cameras+175initials.
Three-way fresh63009 completed96games: eb72returns/28first/18games≥2/44of68later;
40069scaffold86/30/23/56of80later; candidate73/28/21/45of71later.
Candidate versus scaffold−13returns, mean−.40625/game paired95%CI[−.96875,+.15625],
later−6.62pp CI[−16.95,+3.80]. Against eb+1return but later fraction−1.33pp.
This does not establish transferable improvement. Do not promote the batch as
a better general player. Stop more old small replay batches pending a review
of broader on-policy experience and selective action credit. No fresh admission.

Entire pass589.36s/900,160,146actual transition cameras+271initials. No interruption,
unknown work, refit or active source change; source/maps/weights frozen checks pass.

New bundle SHA4c0b6374f3e35fad8260d5bd07161f780383de99a8c536b67437169d12200811;
new W SHA3f2e4958de960c32d0ba30b61d25449e1d60d1344fe9c5bf669b12d08f1419aa.
