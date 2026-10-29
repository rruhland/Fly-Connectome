# Independent challenge: compact symbolic policies

Research only, 2026-10-29. No code, fitting, native cameras, or checkpoint changes.
Existing HANDOFF and independent-learning-directions were read; the user has since
waived architecture approvals. This note adds no permission requirement.

## Recommendation

A **small arithmetic/conditional program with structural search** is the strongest
remaining compact-policy alternative, but confidence is low. It is worth one
bounded test after the current learned-model experiment, not an unbounded CGP,
TPG, or NEAT restart. Small deployment cost does not imply cheap learning.

Rank among distinct compact-policy routes:

1. **Grammar/CGP arithmetic policy over anonymous objects and ordered pairs.**
   Its transferable bias is compositional arithmetic and conditional reuse:
   a few learned products, ratios, differences and tests may express a relation
   that a frozen random-feature linear head approximates poorly. No prewritten
   intercept/bounce expression or semantic object selection is acceptable.
2. **Tiny stateful register program / TPG.** Only promote this if reproducible
   visual aliasing remains after conditioning on present anonymous velocities.
   Otherwise its state, graph evolution and reward variance enlarge a search
   already short of fitness evaluations. The failed Ogma context result weakens
   a memory-only justification.
3. **NEAT/topology evolution: do not allocate now.** Mutating hidden connectivity
   is structurally different, but nonlinear neural-weight search still inherits
   much of the failed ARS/CEM credit and sample burden. There is no concrete
   reason this encoding should be cheaper than arithmetic programs here.

This is a change of hypothesis class, not a new source of supervision. It does
not repair missing action coverage or ambiguous observations by itself. A
residual grammar still risks repeating the earlier rescue-without-transfer
pattern. A failed bounded pilot would stop this family under the current budget;
there would be no node-count, primitive, seed or grammar sweep.

## Primary project-source evidence

Wilson, Cully, Luga and Miller, *Evolving simple programs for playing Atari games*,
GECCO 2018, https://arxiv.org/abs/1806.05667 . The author repository README pins
its published Mixed-Type CGP results to commit
`096e84316c0c265e17ea75de7b5fcaa6427fa5dc`:
https://github.com/d9w/CartesianGeneticProgramming.jl . I inspected that original
project source; I did not independently retrieve the complete paper text.

At that pinned commit, `experiments/atari.jl` sends raw RGB to a program, uses
argmax outputs for actions, and accumulates scalar environment reward. Its
`src/EAs/oneplus.jl` clones a champion and mutates programs. `src/mutation.jl`
and `src/chromosomes/cgp.jl` show that genes decode to operator/connection/output
choices and constants. Thus floating-point genomes are not merely dense neural
weight noise: mutations can change the computation graph itself.

The same source's `cfg/base.yaml` defaults to **10,000 fitness evaluations**, nine
mutant offspring, 40 fixed nodes, and optional recurrence. A fitness evaluation
can run up to 18,000 decision loops, with an additional repeat on 25% of loops.
These are inspected source defaults, not a verified assertion that every reported
Pong run used them or consumed the maximum. They establish the important scale:
compact program execution coexists with a large native interaction search.
The source also exposes many image/list/math operators. Its strong generic image
processing prior should not be mistaken for a tiny unstructured search.

A separate Python TPG project inspected was
https://github.com/Ryan-Amaral/general-game-playing-tpg . Its `sgp-train.py` uses
360 teams, image subsampling, 1,000/5,000/18,000-frame schedules and an open-ended
generation loop. It is an implementation example, not the original Kelly/Heywood
paper or independent evidence of cheap Pong acquisition. It strengthens no claim
that TPG can reach the present 95% sustained-return goal in minutes.

## One concrete bounded pilot

Keep frozen vision, raw-feature output and original40069 scores available. Feed
only legal six-field anonymous rows to the new program. Evaluate the same shared
scalar subprogram on every object or ordered object pair, then use declared
permutation-invariant mean/max reductions to form three action residual scores.
Empty inputs yield zero residual; no object gets a ball/paddle label or semantic
selector. Existing public phase and past scalar hit may be ordinary terminals.
Use a typed DAG with at most16 active arithmetic/conditional nodes, generic
`+`, `-`, `*`, protected division, absolute value, min/max and comparison/select,
and a small fixed constant alphabet. No reflection primitive, contact target,
analytic extrapolator, shaped distance reward, simulator-state field, or
expert/baseline action target. A zero-residual incumbent reproduces40069.

Mutate operator, input edges, output edge and constants; require at least one
active structural change per challenger. Deduplicate identical action signatures
on saved TRAIN observations only to avoid paying for exact duplicates. Do not
score a new action using another policy's saved return. Use actual scalar-hit
returns from complete native TRAIN games, with unchanged primitive camera-rate
actuation and timeout rules. Baseline scores are deployable scaffold inputs,
not labels. The objective is actual total own hits, with first/later counts
reported and smaller programs winning exact reward ties.

Suggested ceiling: **900s and160,000 new cameras, whichever expires first**, at
most16 generations of eight challengers. Use four phase-balanced TRAIN starts per
generation, rotate through a fixed32-start bank, and reevaluate the incumbent on
the same current starts. Cache exact unchanged trajectories where legitimate.
The maximum128 challengers is an upper bound, not a promise that the camera cap
will permit them. Freeze grammar, RNG, starts and selection rule before launch.
No successful-action labels, reward-filled prefixes, teacher imitation or
selective branch rescues enter this search.

Reserve the final150s/30,000 cameras inside the same ceiling. Freeze one selected
program; evaluate complete matched32 TRAIN games against40069. Proceed to32
pre-reserved fresh pairs only if total and later returns both improve without
losing first returns, and only if the original cap permits completion. A larger
qualification and independent training replicate are still needed before the
95% milestone. If the search cannot produce and evaluate a fixed endpoint within
its cap, report an incomplete acquisition test, not proof that symbolic policies
cannot solve Pong. Do not promote on selected four-game fitness.

The useful discrimination is whether a few learned explicit relations transfer
beyond the selected fitness games. More rewarded starts and a short program are
not sufficient evidence. The current planner should face the same demand: real
full-game outcomes, not rollout error or value-tail agreement.
