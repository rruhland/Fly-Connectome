# Bounded orchestration results — 23 October 2026

The session produced additional direct-interception learning, but no new policy that demonstrably improves and retains both interception phases. M1B and sustained rally learning remain incomplete. No backpropagation or architecture change was introduced. The preserved RIGHTtheta6 remains the main experimental carry; none of this session's endpoints replaces it.

![Checked session results](/workspace/m1b-experiments/pong-mechanism-review-20261022/session-20261023-results.png)

Each experiment has its own fresh paired cohorts. Absolute counts across rows are not a longitudinal learning curve.

| Experiment | Fresh result | Decision |
|---|---|---|
| Balanced mean-fitness practice | Direct47→59/64; opponent-return51→37/64. Balanced−1.56pp, paired95%CI[−9.38,6.25] | Direct learning and opponent-return harm are individually supported; joint progress fails |
| Five fixed learned-weight mixtures | TRAIN hits50,47,47,43,43/64 for alpha0,.25,.5,.75,1 | Unchanged start wins; registered futility skips576 evaluation games. No changed policy or heldout claim |
| Weaker-phase fitness | Direct48→45/64; opponent-return44→42/64. Balanced−3.91pp, CI[−12.50,4.69] | No demonstrated first-interception improvement;3/8 joint screens pass, combined/80% gates fail |

The weaker-phase endpoint's ordinary first returns were44/64 versus47/64 at its own start. Ordinary all-opportunity returns were75/138 versus63/127; preservation passed, improvement did not. Compared with the older GLOBAL reference it retained a first-return advantage, but that does not establish new learning from the carried start. Direct performance was below the designated DIRECT reference45versus54/64, difference−14.06pp CI[−23.44,−4.69]. Differences against its own start were inconclusive, not established harm.

There is an exploratory rally signal: three-or-more-return games rose2→9/64 versus the carried start (+10.94pp, CI[1.56,20.31]). Two-or-more rose14→19/64 with an interval spanning zero, and mean returns/game also remained inconclusive. Three-or-more versus GLOBAL was8→9/64, inconclusive. These are multiple secondary comparisons from one training run, not independently replicated sustained-rally learning or a reason to override the primary failures. Preserve the endpoint and signal for a prospectively designed future test.

## What the failures narrow down

Both full training attempts made nine substantial nonzero updates: net generic-cache function-RMS movement0.26638 for mean fitness and0.25267 for weaker-phase fitness. The latter changed253 calibration greedy decisions and had55/72 nonzero directional minimum-fitness contrasts. Absent/tiny updates do not explain these runs. Equal practice, the tested parameter segment and this weaker-phase objective were insufficient within their budgets. None proves the existing visual/linear class cannot combine skills.

Small practice banks and an unconfirmed combined update remain plausible optimization explanations. The search evaluates perturbed candidates but creates an untested center by combining directions. Astra proposed, and Sol independently supported, a future fresh-TRAIN comparison of proposed versus current center before accepting updates. This is a recommendation, not another staged allocation. It costs extra episodes, needs a fixed acceptance criterion, and cannot guarantee retention from a small bank. See [competing hypotheses and limits](../plans/2026-10-23-integration-hypotheses.md). Avoid another automatic noise/rate/mixture sweep. Architecture changes still require user review.

## Execution, checks and cost

All three allocations completed as registered, including interpolation's deliberate no-change stop. Both independent result checks passed for each; all core source remained68d15a7. The failed prior interrupted allocation was not resumed, its clock was not reset and its14879transitions/161initial images remain separately recorded. Current results exclude every prior used, unused and rejected schedule; reserved final streams remain untouched.

| Allocation | Training transitions | Evaluation transitions | Initial images | Allocation wall |
|---|---:|---:|---:|---:|
| Balanced |159854 (155015used,4839partial)|90374|2360|1746.956s|
| Mixture |28167 (all used)|0|320|189.085s|
| Weaker phase |159830 (153978used,5852partial)|89902|2374|1598.446s|
| **Total** |**347851 (337160used,10691partial)**|**180276**|**5054**|**3534.487s (58.91min)**|

Allocation wall excludes orchestration, staging and post-exit checks; it is not total session elapsed time. Training ancestry and failed research costs are retained separately in each protocol. Historical and completed current journals were losslessly compressed only after raw checks and reader closure, with byte-exact reconstruction verified; weights, metadata and source remain available. All check/archival operations added zero observations or physics replay.

Astra selected and challenged directions, consulted primary sources, independently reconstructed results and wrote the synthesis. The same Sol6.1XHigh agent implemented, challenged, verified and interpreted the tests. It was paused during routine monitoring and resumed for final verification. No replacement agent or model substitution was used; no unsupported token-cost percentages are claimed.

Sources actually consulted for the relevant decisions: [WiSE-FT abstract/introduction](https://arxiv.org/html/2109.01903v3), [group DRO abstract/introduction/setup](https://arxiv.org/html/1911.08731v2), and [ARS section2.1/Algorithm2](https://arxiv.org/html/1803.07055v1). They motivate hypotheses, not guarantees for this Pong learner. Their supervised/backpropagation methods were not adopted.

Detailed records: [balanced result](../plans/2026-10-23-balanced-practice.md), [mixture decision/result](../plans/2026-10-23-interpolation-selection.md), [weaker-phase protocol/result](../plans/2026-10-23-minphase-interception.md), and [ongoing scaffold plan](../plans/2026-10-21-continuous-learning-scaffolds.md).
