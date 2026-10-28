# Sustained return learning

User authorizes ongoing orchestration until ball returns are learnably consistent throughout a match, or user stop. Bridge work belongs to another thread. No backpropagation, privileged policy inputs, or expert action labels. Major architecture changes still require user review; smaller changes and experiments are authorized.

## Starting point and objective

Carry the exported October25 final head (SHA256 `184dd67a67999224dbe4eb5ff3ba3b86ebdf821d15f1486eef2432a56e99fb15`) with its exact scaler and current frozen vision. Preserve r4 and the final export as immutable baselines. The previous evidence established first-return gains, not rally improvement: both heads had 11/32 repeated-return games and the final head had 42 total own hits versus r4's45. All160 phase probes gave154 direct and142 opponent-return successes. Do not overstate that as uniformly reliable first interception.

The practical target is a carried model that learns to return reliably at both opening and later contacts. Working acceptance target: at least95% successful returns among actual incoming opportunities, separately for first and subsequent contacts, on fixed fresh full-game cohorts with enough later contacts to expose weaknesses (aim200+ later opportunities). Report agent misses per game, opponent-ended games, timeout censoring, contact-depth buckets, total hits and wall time. Sparse later opportunities or easy opponent wins cannot establish rally consistency. Confirm improvement against the unchanged carry on paired starts and demonstrate that learning gains recur with another training allocation; this is still a reset-based learning scaffold, not proof of general continuous online RL.

Native benchmark remains deterministic opponent speed.65, K1, ordinary full games, current camera/vision path. Longer rallies or stronger opponents may be used as explicitly labeled training/stress conditions when ordinary games provide too little later-contact coverage. Keep native matched evidence separate. Do not optimize primarily for wins.

## Orchestration and first pass

Root owns hypotheses, sources, experiment choice, evidence review and pivots. Reuse Sol6.1XHigh `/root/pong_one_pass_research` for independent mechanism analysis, alternatives, code and execution. Do not let successive fitting variants replace a causal explanation of failure.

1. Collect a bounded fresh TRAIN full-match baseline (32–64 starts). Separate missing later-state coverage, visual tracking failures around rebounds, late motor correction and update interference. Hidden physics may diagnose failure but never supply policy features or action targets.
2. Extend the successful reward-based one-action intervention method to later failed returns. Explore earlier decision points or bounded action sequences if one-camera interventions produce no reward contrast. Preserve actual earned behavior across complete trajectories; do not merely preserve labels at isolated opening states.
3. Fit and deploy a concrete candidate quickly, then measure actual full-game returns and first-return retention. Keep TRAIN selection distinct from fresh qualification. All replayed/branched training observations count toward cost.
4. After each bounded allocation, root reviews: did we discover successful behavior, did fitting reproduce it, did it survive closed-loop deployment, and did it generalize? Change the mechanism at the failed stage rather than endlessly sweeping the same fit. An uninformative discovery pass should change exploration/time horizon; successful labels with failed deployment should trigger interference/coverage analysis.

Use previous successful full trajectories as memory, revalidated under the carried head. Reclassify any prior evaluation reused for learning explicitly as TRAIN; leave the saved larger validation untouched. Never silently replace the model with a scripted interception controller.

## Outside ideas to examine

- [Ross, Gordon and Bagnell2011, DAgger](https://proceedings.mlr.press/v15/ross11a.html): motivation to aggregate states reached by the learner itself, since actions change future observations. We do not have or introduce its expert-label assumption.
- [Chang et al.2015, LOLS](https://proceedings.mlr.press/v37/changb15.html): motivation for local action deviations and rollouts under the learned policy, rather than measuring only agreement with a reference. Its structured-prediction guarantees do not establish guarantees for our partial visual state.
- Existing local-action credit remains related to classifier-based policy iteration. Combine useful mechanisms only when they produce actual native reward improvements without backpropagation.

These sources motivate hypotheses and mechanisms; none establishes that this Pong learner already works. Save concrete accepted/rejected candidates, failures, costs and the next decision in the reconstruction log.


## October28 consequence-sensitive correction

After one-witness soft fitting and target-only projection lost broad native returns, root and Sol selected a different credit mechanism. Choose eight harmed TRAIN games deterministically, prioritizing lost later returns. At the identical first divergent decision compare one carried action followed by the candidate policy against the candidate action followed by that same candidate policy. Use actual remaining own-hit differences as context-specific credit; ties remain unordered. These are learner-earned preferences, not optimal-action labels. A null continuation must match saved native records. All prefix replay, branches and verification cameras are charged.

If at least two independent decisions have positive credit, minimally project the candidate onto those verified choices plus its already successful complete target trajectory. Keep other old commands unrestricted. Evaluate actual160TRAIN outcomes; useful total/later improvement earns an untouched paired fresh comparison. If causal differences are scarce or the correction does not deploy, reconsider coordinated changes; do not extend a one-action fishing expedition. d7 remains immutable. First block bounded8–12min; this is a review cadence, not a user stop limit.

Root reviewed [Lazaric, Ghavamzadeh and Munos2016](https://www.jmlr.org/papers/volume17/10-364/10-364.pdf), abstract and regret discussion: action errors should be weighted by their consequences. Our deterministic branch screen adapts that motivation; it is not their algorithm or a transfer of its Markov-policy guarantees. Sol independently recommended the same selective-credit mechanism after reviewing failed37-witness aggregation and historical8D global search. Both chose this over another global penalty or blind search.
