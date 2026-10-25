# Build a working interception learner

The user renewed ongoing orchestration on October25: continue until interception learning is working, reliable and efficient, or the user stops us. Earlier two-hour ceilings and final-two-experiment stopping rules are historical, not current limits. No backpropagation. Major architecture alterations still require user review; practical optimizer, reward, curriculum and runtime changes are authorized. Reuse the established Sol6.1XHigh agent; Astra owns direction, outside evidence and independent review.

The deliverable is improved behavior, not passed audit gates. Preserve promising partial improvements and compare them honestly with the carried learner. Retain enough checks to catch bugs and overfitting; do not require eight simultaneous statistical screens before continuing a useful learner. Record actual work, including unsuccessful rollouts, without recreating large protocol documents for every adjustment.

## Immediate engineering direction

First make the rollout loop cheaper without changing the legal camera, vision, policy, actuator or simulator semantics. Use the existing evaluator directly, with start/end frozen-state checks and compact episode accounting rather than per-frame SQLite and full-state fingerprints. Check a few exact trajectories against retained reference records and measure speed. This is an explicit lightweight research execution path; it does not promise the old crash-recovery/within-episode mutation-detection contract.

Then give the learner more consistent practice. Six fresh starts per phase and sixteen candidate policies produced optimistic selection: the last selected children averaged83.33% discovery hits but68.75% on new training games, versus72.22% for the unchanged incumbent. This suggests testing a persistent, diverse training bank and repeated missed situations, with periodic fresh games to detect memorization. Deterministic seed reuse is an explicit simulator curriculum scaffold; all repeated camera processing is counted, and no hidden state enters the learner. Do not automatically rerun the previously suggested DIRECT restart or relax a confirmation threshold.

Sol independently examines concrete visual/motor failure evidence and implements the lean loop. Astra examines optimizer/data-allocation alternatives. Combine useful ideas when justified; do not turn every component into a separate full experiment. Select the next training implementation after speed/equivalence and failure review.

## What counts as useful progress

Track direct-opening and opponent-return-opening first contacts separately, with ordinary-game return opportunities alongside them. Aim for at least90% first interception in each opening cohort on two fresh cohorts, a clear gain over the starting policy, and useful transfer to ordinary games. Treat this as an engineering target, not a confidence-interval veto; keep intermediate gains and continue fixing misses. Do not claim sustained rally competence from first-contact-only tasks. Check that improvement is reward-learned rather than an inserted hand-coded interception controller. Record time and observations to improvement, and repeat a promising learning recipe before claiming dependable acquisition.

## Outside inputs considered

[Ng and Jordan, PEGASUS, UAI2000](https://arxiv.org/pdf/1301.3878), introduction and section3, motivates comparing policies on shared fixed simulation scenarios to stabilize the objective. Its simulator access is a scaffold, and its generalization guarantees do not establish that a small Pong bank suffices. We adopt neither hidden-state policy access nor differentiating through the simulator.

The posted [Pong CEM project](https://sandeepreddybaddam.github.io/pingpong/) uses six handcrafted semantic features, so it is not evidence that our generic visual representation will work equally well. It does motivate compact black-box optimization as a practical option. [pytorch-es](https://github.com/atgambardella/pytorch-es) reports Atari Pong training on a12-core machine over roughly24hours; it is not a fast drop-in answer to this project. No foreign code has been executed or imported.
