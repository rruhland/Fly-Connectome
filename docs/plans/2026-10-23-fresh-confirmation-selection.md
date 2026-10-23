# Confirm proposed updates on fresh phase-balanced training games

## Research question and selection

Can the existing reward-trained search improve the carried interceptor when its combined update must first demonstrate improvement without an observed phase loss on fresh training games? Mean-balanced practice previously improved LEFT while significantly damaging RIGHT. Minimum-phase fitness did not improve joint first returns; its exploratory longer-rally signal remains unreplicated. A weight interpolation selected the unchanged carried policy. These results motivate testing the proposed update itself rather than inferring its value from perturbed candidates.

Astra posed the mechanism question and compared outside evidence; Sol independently challenged the proposal, located a prior negative confirmation experiment, and derived alternatives. Both selected the unchanged mean-balanced proposer plus a fresh empirical acceptance check. This is a bounded mechanism-informed experiment, not a controlled causal attribution against the earlier trajectory. No new architecture, backpropagation, privileged policy inputs or vision changes.

The earlier `pong-validated-win-children-20261021-r1` already used fresh confirmation: seven complete cycles, two accepted children, and inconclusive frozen win results. That interrupted historical allocation remains a negative precedent, not a completed confirmation success. The present experiment differs in dense first-hit reward, strong opponent, explicit two-phase constraints, and confirmation of a combined ES update instead of a directly evaluated child. Confirmation is not a wholly new idea here.

## Fixed experiment

Start again from RIGHTtheta6, preserving the successful LEFT-then-RIGHT learning ancestry. Use eight mirrored directions at noise .25, top four by maximum sign fitness, step .10 and existing function-RMS cap. Discovery uses six LEFT and six RIGHT ordinary starts shared across sixteen candidates: 192 first-contact episodes per cycle. Fitness is their mean binary own-hit reward.

Evaluate the resulting combined proposal and its incumbent on 24 fresh LEFT plus 24 fresh RIGHT starts: 96 additional episodes. Let D_LEFT and D_RIGHT be the proposed-minus-incumbent hit counts. Accept exactly when D_LEFT >= 0, D_RIGHT >= 0, and D_LEFT + D_RIGHT >= 2. This is empirical optimizer selection, not a significance test or retention guarantee. Every confirmation observation is TRAIN and charged to the budget; heldout games never select updates.

A complete rejection keeps weights byte-exact while advancing proposal RNG. A partial confirmation commits neither weights nor RNG. An exact-byte zero proposal may skip redundant confirmation after all discovery cells complete; its unused confirmation schedules remain reserved. Save proposed and applied steps, phase paired wins/losses/ties, attempts and acceptances separately. At least five and at most twelve complete cycles; 160k training transitions / 1200 seconds, then at most 900 seconds evaluation / 2100 seconds total from immutable launch. Resume cannot reset the clock. At roughly 89 transitions per episode this permits about six cycles, potentially fewer accepted updates.

If eligible training ends with zero accepted updates and an unchanged policy, skip the 576 heldout jobs under prospective futility: report no changed learner and no new heldout claim. Otherwise retain the existing independent 64-start LEFT/RIGHT/FULL evaluation, equal-phase first-return primary, all eight retention/improvement screens, and separate 80% competence criterion. No best-development checkpoint selection or automatic global promotion. All previous accepted/rejected/unused schedules and canonical environment IDs remain excluded; final streams stay untouched.

## Why 24 confirmation starts per phase

An exact paired-multinomial toy calculation compared 16/24/32 starts. Under independent phases, discordance .15 and true +5 percentage points in each phase, the selected 24-start rule accepts about 56.7%; under zero lift and discordance .20 it accepts about 25.6%; for a +15/-10 point tradeoff with discordances .25/.20 it still accepts about 15.9%. These are illustrative assumptions, not measured Pong power or guaranteed protection. More confirmation reduces proposal count; fewer starts admit more phase tradeoffs. Artifact: `/workspace/m1b-experiments/pong-confirmation-research-20261023/acceptance_power.json`.

Astra read the abstract, introduction and section 2.1 of [Mahsereci and Hennig, Probabilistic Line Searches for Stochastic Optimization, NeurIPS 2015](https://papers.neurips.cc/paper_files/paper/2015/file/812b4ba287f5ee0bc9d43bbf5bbe87fb-Paper.pdf). Noisy objective comparisons make deterministic decisions uncertain. Their gradient/GP/Wolfe machinery is not adopted, and their analysis does not establish this Pong rule. Existing ARS evidence likewise does not guarantee that a combined normalized top-four proposal improves the unsmoothed task.

## Interpretation and stop rules

A passing joint heldout result warrants replication, not M1B completion. Zero/few acceptances demonstrate limited effective update dose, not representation failure. Frequent rejected phase tradeoffs implicate the proposer; noisy marginal duels implicate resolution. Accepted TRAIN gains that disappear on heldout games expose selection noise/generalization. Report each possibility against actual evidence, including longer-rally secondary outcomes without post-hoc promotion.

No threshold, batch size, reward, step or noise tuning during this allocation. Reassess before any next experiment. This continuation began 16:49:30 UTC on October 23 and must stop by 18:49:30 UTC, reserving ten minutes for closure; a full 35-minute allocation cannot launch after 18:04:30 UTC.

## Prelaunch review

Sol's ten focused fixtures passed in 3.30 seconds; staging/validation passed with zero observations. Parent independently reconstructed all schedules, exclusions, first population directions and mapped candidates in 0.384 seconds: PASS. There are 912 distinct canonical accepted start IDs, 952 rejected candidates and 40,903 prior excluded IDs. Frozen driver d6616be2585464fa334cd74812586e3aaaf97aaec3da6f8d47081d3e7880483b; protocol 3daef4c016acf8686c5da7f9730d07dbae3bf4aff734344b994c36b13e129954; unchanged source 68d15a7. The single allocation was authorized after both reviews.

## Blind alternatives considered during execution

Before reading this allocation's outcomes, Sol favored a balanced acquisition run from the earlier DIRECTtheta9 if several accepted updates still fail to combine skills. This asks whether joint practice before RIGHT specialization follows a more useful path. It would be a separate lineage, not silent carry of the RIGHT branch, and a single fresh-seed comparison cannot isolate initialization causality. If almost no proposals are accepted, Sol instead considers confirming the best actually executed perturbation child, retaining the prior win-child negative precedent. Neither follow-up is automatically selected.

Astra independently considered preserving observed behavior through localized/constrained perturbations, and combining phase-specific reward-derived search directions before proposing an update. [Sener and Koltun, NeurIPS 2018, sections 3.1–3.2](https://papers.nips.cc/paper_files/paper/2018/file/432aca3a1e345e339f35a30c8f65edce-Paper.pdf) describes finding a minimum-norm convex combination of task gradients to obtain a common descent direction under its assumptions. A no-backprop analogue could use only reward-based perturbation estimates. Our tiny noisy banks and normalized finite directions do not meet the exact-gradient premise; apparent conflicts or a zero combination would not prove Pareto optimality. Their backpropagation/encoder machinery is excluded. This is an unselected idea, with a serious risk of starving updates and requiring fresh confirmation anyway. Behavioral constraints also risk preserving mistakes and need legal observed-feature bookkeeping. These alternatives deserve comparison rather than an automatic parameter sweep, but a simpler path test may be more informative per implementation cost.

## Result and next decision

Completed in981.600 seconds: six cycles, zero accepted changes, exact carried head retained;159824 training transitions,1755initial images,0DEV. Both raw checks passed. Confirmation phase nets were (+2,-1),(0,0),(+2,-9),(-1,+2),(+2,-2),(+1,-2). None meets total>=2 even without phase protection; no threshold relaxation follows. No heldout learning claim. [Selected final follow-up and interpretation](2026-10-23-executed-child-selection.md). Visual artifact: `/workspace/m1b-experiments/pong-mechanism-review-20261022/confirmation-20261023-results.png`.
