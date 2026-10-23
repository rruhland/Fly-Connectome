# Test weaker-phase fitness without changing the policy

## Evidence and decision

Fresh balanced mean-fitness practice strengthened direct returns but damaged opponent-return interception; the joint gate failed. A subsequent fresh-reward five-coefficient interpolation selected the unchanged RIGHTtheta6: balanced training hits50,47,47,43,43 of64 for alpha0,.25,.5,.75,1. Direct hits26,26,28,28,28 of32; opponent-return hits24,21,19,15,15. The registered futility rule stopped with zero evaluation, no changed policy and no heldout claim. Parent independent raw reconstruction passed in0.451s. This rejects the bounded segment/selection attempt, not the function class.

Before interpolation results, Astra challenged Sol's proposed direct-ancestor restart with an objective-mismatch alternative. Sol independently favored weaker-phase fitness: a restart changes history but still lets mean fitness sacrifice a phase. Both agreed to keep all other search mechanics fixed. The current observed tradeoff supplies a concrete mechanism; a fresh run remains a mechanism-informed attempt, not a clean causal comparison with the earlier mean-fitness run.

## Selected intervention

Start from preserved RIGHTtheta6 again, not the failed balanced donor. Keep six ordinary LEFT and six ordinary RIGHT starts per population, eight mirrored directions, noise.25, top4 and update.10 with existing function-RMS cap. Replace candidate fitness `(mean(LEFT)+mean(RIGHT))/2` with `min(mean(LEFT),mean(RIGHT))`. Keep fresh starts each population, native first-contact cutoff, strong opponent, greedy execution, frozen generic vision and existing linear head. No backpropagation or architecture change. Training phase information is explicit curriculum assistance and never a policy input.

The minimum equals the average minus half the phase imbalance. It discourages sacrificing the already weaker phase but is not a formal retention constraint: a stronger phase can still decline. Six starts per phase make estimates coarse and the minimum noisy/downward-biased. Do not bundle larger batches, new noise, new step sizes or adaptive weights with this test. Report phase-specific contrasts and actual movement.

Retain the existing eight heldout screens, including at least10pp balanced improvement against the actual carried start, phase controls and ordinary-game retention against that start. Report actual effect/interval separately from gate status. Absolute80% competence is separate. All prior schedules and rejected candidates excluded, fresh roots fixed before launch, no development selection or parameter sweep. The carried head's learned ancestry remains558832 training transitions; failed balanced/interpolation/paused attempts remain separately paid research exposure, not inherited updates.

Bounded allocation160k training transitions/1200s plus900s evaluation,2100s overall, same minimum5/max12 complete populations. No launch after07:24UTC, user hard stop08:09 with ten-minute closure reserve. Source/protocol frozen through both independent checks. No automatic further training allocation after this attempt; synthesize whether objective, sample noise, optimization path or representation deserves the next intervention.

## Outside evidence

Astra read abstract/introduction and setup§2 of [Sagawa et al., ICLR2020](https://arxiv.org/html/1911.08731v2). Their worst-group formulation helps specify the alternative, including training group information without inference access. Their warning that worst-group training performance need not generalize reinforces fresh evaluation. Their supervised gradient optimizer and convergence claims do not transfer to this Pong search; none is adopted.
