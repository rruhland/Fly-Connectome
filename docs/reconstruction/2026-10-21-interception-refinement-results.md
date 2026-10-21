# First-return scaffold refinement: no detectable improvement

The bounded continuation did not improve first-return reliability: the carried
start and refined endpoint both returned 34/64 fresh first incoming balls (53.1%).
The paired lift was 0 percentage points, 95% bootstrap interval [-4.69, +4.69].
Neither the registered learning screen (at least +10pp with positive lower bound)
nor the provisional 80% reliability target passed. M1B remains incomplete.

## Research decision and orchestration

Follow the [continuous-learning scaffold plan](../plans/2026-10-21-continuous-learning-scaffolds.md).
Astra independently considered exploration scale, first-versus-later exposure,
and visual/motor-state sufficiency. The existing Sol 6.1 XHigh agent was reused
to challenge explanations, examine prior results and primary research, and execute.
Sol initially favored a two-hit curriculum; Astra challenged withholding credit
for successful first interceptions when the second failed. Sol revised its
recommendation to first-return refinement before harder chaining.

[Salimans et al. (2017)](https://arxiv.org/pdf/1703.03864) motivated distinguishing
parameter-smoothed search fitness from the executed discrete policy. The
curriculum papers [Florensa et al. (2017)](https://proceedings.mlr.press/v78/florensa17a/florensa17a.pdf)
and [Florensa et al. (2018)](https://arxiv.org/pdf/1705.06366) supplied useful
curriculum concepts, but their reset/goal-generation machinery does not establish
applicability to this pixel-only, non-backpropagation learner. No oracle resets,
teacher-fitted weights, or backpropagation were introduced.

## Frozen experiment

Start from the final return head of the first qualifying lineage, selected by
chronology rather than evaluation ranking. Carry its optimizer RNG and learned
state. Frozen vision and the existing visual readout were retained; scalar first
own hit was declared scaffold reward. Opponent strength remained .65.

Eight mirrored search directions, top four, used twelve common fresh serves per
population. Fixed perturbation RMS was .20 times initial head function RMS;
step cap was .05 times that RMS. Training stopped at the original 80k-camera/
15-minute cap, taking the last complete endpoint. Evaluation used 64 fresh paired
20-second full games per policy, with no intermediate evaluation or best selection.
This jointly changes task emphasis and search scale; it does not isolate noise
scale causally. Frozen vision and episodic population search remain scaffold
assistance, not a demonstrated continuously adapting general learner.

| Fresh evaluation | Carried start | Refined endpoint |
|---|---:|---:|
| First balls returned | 34/64 | 34/64 |
| All incoming opportunities returned | 58/122 (47.5%) | 58/121 (47.9%) |
| Games with at least two returns | 15/64 | 17/64 |
| Games with at least three returns | 7/64 | 4/64 |
| Games with at least four returns | 2/64 | 1/64 |
| Maximum returns in one game | 4 | 6 |

All-return lift was +0.39pp, 95% interval [-5.83, +6.64]. The separate -5pp
retention margin was inconclusive: neither preservation nor damage is established.
The isolated six-return game is not evidence of reliable rally improvement.
The opponent returned every opportunity; wins were descriptive and not a gate.

## Costs and verification

Four complete populations; 79,835 training transitions (66,181 used and 13,654
unused partial-population work), 33,907 evaluation transitions, and 1,055 initial
images. Inherited experience was 199,676 training transitions, not reset to zero.
Training took 785.49s; all-in execution/report recovery took 1171.06s (19.5min).
No new visual-learning images or reserved final-stream use.

All games finished. The coordinator then failed while constructing the summary
because two dictionaries supplied the same field. An external reporting-only
recovery used existing records, preserved original code/traceback/clocks, and
added zero observations (1.006s recovery computation). The parent independently
checked raw counts, annotations, frozen source identities and paired statistics:
PASS in 2.435s, zero observations. This report defect did not change the policy.
Sol's focused verification also passed in 2.54s: 896 complete used-training/eval
pins, all 64 candidate weight maps, source/input/RNG identity, and reconstruction
of all four optimizer updates. All four updates were nonzero; 6, 7, 8 and 6 of
eight mirrored directions had unequal mean fitness. The null is therefore not
explained by absent reward variation or an optimizer that never updated. Only
two first-return outcomes changed: one improvement and one regression.

## Interpretation and next decision

Smaller local steps with first-hit credit did not produce useful incremental
learning within this allocation. This does not prove a representation ceiling
or that longer training would never work. Do not automatically launch another
noise-size variation or promote this endpoint as an improved learner.

The next research decision should distinguish weak/local search credit from
insufficient visual precision or action-history information. Use retained traces
and candidate contrasts first to select a discriminating intervention; any new
learning trial must have a separate budget and fresh evaluation. Major
architecture changes still require user review. No follow-up was allocated here.

Sol proposed a generic previous-action trace, initialized to preserve the exact
starting policy, against a matched unchanged-head continuation control. Astra
regards actuator-history insufficiency as a plausible competing hypothesis, not
the conclusion of this null result: small policy changes and limited local search
also remain explanations. Inspect existing failure traces and settle intervention
scope before choosing that experiment; no representation change is approved by
this report alone.

## Retained artifacts

- Trial: `/workspace/m1b-experiments/pong-interception-refinement-20261021-r1/`
- Research briefs, decision, parent check and plot:
  `/workspace/m1b-experiments/pong-scaffold-research-20261021/`
- Exact protocol: trial `PROTOCOL.md`; driver: `run_refinement.py`.
- Driver SHA256: `16988938b8f857495fc883a983cb6cb113d48b8e8f1e9c443ce358ca8b32c89a`.
- Runtime source: `6df746f7e24989fd0e2f52ca166590d104258d3a`.
- Endpoint: `theta_at4.npy`; source and reporting recovery remain separately pinned.
