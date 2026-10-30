# Scaffolding a continuous non-backpropagation learner

## Current authorization — 2026-10-29

The user now approves continued work without further architecture approval until
the current reliable-return milestone is complete or they stop work. This
supersedes the older approval requirements below, while retaining no backpropagation,
no privileged policy inputs and no expert action labels. The broader continuous
learner and connectome goals are not satisfied by a successful Pong controller alone.

Current research favors actual learned behavior over repeated surrogate fits.
Following failed value, rollout and broad program-search pilots, the active test
uses a compact generic feedback prior whose object binding and parameters are
selected by actual rewards. Treat a successful result as a scaffold: document the
prior explicitly, qualify it on unseen play, then test learning replication and
longer rallies. Preserve useful learning and investigate concrete remaining failure
mechanisms instead of automatically switching algorithms or sweeping parameters.

## Current direction agreed with the user

Discover what can learn each component, then integrate the successful mechanisms
and demonstrate the whole progression in one continuously adapting learner. Pong
is the initial environment, not the definition of the eventual general learner.
On 2026-10-22 the user explicitly requested continued orchestration until told to
stop. Continue bounded research cycles autonomously, reviewing evidence and
reassessing direction between them. This supersedes the previous single-batch
stopping point; it does not remove resource discipline or architecture review.

No backpropagation, privileged policy observations, oracle-fitted teachers, or
hidden-state aiming instructions. Scalar environment rewards such as own hits may
be declared as scaffold assistance. Major architecture changes require user
review; small/medium changes and bounded experimentation are authorized. Preserve
the existing reserved final streams for a prospectively frozen procedure.

## Progression

1. **Reliable interception:** learn to return incoming balls across varied fresh
   trajectories. Establish useful improvement over initialization, the carried
   starting policy, and appropriate simple controls. Existing two-seed return
   acquisition is a foundation, not evidence that interception is dependable.
2. **Sustained rallying:** learn repeated exchanges while preserving interception.
   Use a competent returning opponent and a horizon that permits enough exchanges.
   Record survival of successive opportunities and per-opportunity return success;
   report censored rallies separately. Do not make winning the acceptance gate.
3. **Effective placement:** learn contacts that induce opponent misses while
   retaining interception and rallying. Wins become a primary task measure here.
4. **Integration:** specify one procedure and demonstrate acquisition through the
   entire progression without manual checkpoint selection or failure-driven repairs.
   Learning must remain enabled in the demonstrated continuous system.
5. **Generalization and retention:** learn substantially different games using the
   shared learning machinery, revisit prior skills, and measure retained competence.
   Keep broader M1B/connectome requirements explicit rather than silently declaring
   the milestone complete after a Pong scaffold succeeds.

Carry forward successful learned parameters/memory by default; choose the starting
artifact prospectively, not by searching evaluation results. A scaffold may change
its reward, exploration, credit assignment, representation or training environment
when evidence warrants it. Record what changed, what was retained/reset, what
assistance was supplied, and whether earlier skills survive. Distinguish a carried
policy's existing skill from new learning. Successful individually repaired stages
do not establish a single successful learner.

## What remains a hypothesis

Time-varying rewards are one possible integration mechanism, not a committed plan.
Other candidates include exploration adaptation, improved credit assignment,
memory/state interfaces, and skill retention mechanisms. First find what learns
each stage, then use the evidence to select an integration design. A Pong-specific
hit schedule is useful scaffolding but not proof of an environment-general rule.
Any human-scheduled stage changes must later be replaced by a declared procedure
and tested from its beginning; preserve all interventions in the research record.

## Astra / Sol research workflow

- **Astra owns direction:** independently form competing mechanistic explanations,
  inspect relevant primary research, identify what evidence would change the plan,
  select the next experiment and interpret results. Do not reduce this role to
  provenance review or approving implementation details.
- **Sol 6.1 XHigh explores and challenges:** reuse the established agent by default;
  ask for independent alternatives before sharing Astra's preferred explanation
  when useful. Give goals and examples without prematurely fixing the solution.
  Use its deeper reasoning for synthesis, difficult mechanisms and debugging.
- **Joint selection:** compare hypotheses, predicted outcomes, outside-source
  applicability, plausible learning progress, and observation/wall/usage costs.
  Prefer an actual learning experiment with a discriminating outcome over an
  expanding chain of privileged diagnostics or optimizer variants.
- **Sol executes; Astra reviews:** scope the implementation, use focused tests and
  existing verification tools, then independently check the decisive evidence.
  Reserve expensive duplicate audits for consequential changes or findings.
- **Update and decide:** report which hypotheses gained/lost support, what remains
  unknown, and whether to continue, pivot, or seek a strategic user checkpoint.
  Both agents generate ideas; record actual attribution and sources read.

Reuse context and existing infrastructure. A fresh agent needs a concrete benefit
such as an independent critique; do not spawn one for every experiment. Reasoning
effort is not a model intelligence ranking: use XHigh for hard work, avoid invoking
it repeatedly for routine status and bookkeeping. Do not invent model-cost splits.

## Evidence and resource discipline

Before each allocation, freeze the target skill, starting artifact, assistance,
primary outcome, competence threshold, retention criterion, independent evaluation
distribution, and finite observation/wall budget. Include initial images, partial
or failed work and inherited experience. No result-dependent extensions, best
checkpoint selection, final-stream tuning, or resetting interrupted clocks.

For interception/rallying, report first and later incoming opportunities, misses,
return fractions, and success at multiple consecutive exchanges. Episode-level
pairing/uncertainty must account for dependent hits and policy-dependent exposure;
more total hits or a favorable conditional subset alone is not competence. A longer
horizon or opponent change is a declared scaffold intervention, not a free gain.
Require absolute reliability as well as learning improvement, then independent
training replication before calling a scaffold established.

Keep each research batch to one primary experiment and at most one justified
follow-up. Two credible failures within the same mechanism trigger a hypothesis
reassessment and direction change, not an automatic adjacent variation. Continue
within existing authorization; seek user review for genuinely major architecture
changes or decisions that require it, rather than stopping after every batch.
Optimization is valuable when measured and amortizable; do not turn it into an
unbounded parallel project. Publish negative outcomes without relabeling them as
learning success. The ultimate measure is dependable learned behavior per budget.

## Immediate research target

Reliable interception across direct and opponent-return openings, then sustained
rallying. Direct acquisition has replicated and opponent-return acquisition has
succeeded once, but combining and retaining both remains unresolved. Carry the
preserved RIGHTtheta6 experimental head while testing explicit integration
scaffolds; report fresh phase rates and ordinary-game transfer separately.
Distinguish optimization/objective/sampling limitations from visual or motor-state
limitations before proposing architecture changes. Ten-second first-contact tasks
do not demonstrate repeated exchanges. Winning remains descriptive at this stage;
autonomous reward scheduling and a new architecture are not selected.

### 2026-10-22 evidence update

Original-scale population search now replicates strong direct-interception learning:
29→56/64 and27→49/64 on independent fresh cohorts from the same carried policy.
The80% absolute threshold passed only the first run; reliable interception across
ordinary games is not complete. Both specialized endpoints fail to establish
full-rally preservation. This updates the optimization-versus-representation diagnosis:
the existing frozen visual/linear class supports much stronger direct interception
than the preceding local actor experiments demonstrated.

Two full-rally continuation variants failed their primary rally gates. A subsequent
opponent-return first-contact curriculum from the predesignated direct endpoint
produced strong acquisition:19→44/64, paired gain39.06pp CI[26.56,51.56]. Ordinary
first returns improved31→54/64 versus the original general policy. Direct retention
54→50/64 failed noninferiority, so combined progress is not yet established.
Both checks passed; opponent-return acquisition has not yet been independently
replicated. The next research question is acquiring and preserving both opening
phases in one policy before escalating the rally objective. Balanced first-contact practice from the learned opponent-return endpoint is
selected next, requiring new learning and retention against its own start. This
experimental carry preserves acquired updates without claiming qualified promotion;
exact replication is deferred until a viable combined recipe. No new architecture
is implied. Original ARtheta112 remains the global reference. Detailed criteria:
[population reassessment](2026-10-22-population-search-reassessment.md),
[opponent-return study](2026-10-22-opponent-return-interception.md).

Historical evidence and artifact locations: [HANDOFF](../reconstruction/HANDOFF.md).
The [M1B roadmap](2026-09-27-m1a5-to-m1b-roadmap.md) remains the broader milestone
reference; this plan governs the current scaffold research and orchestration.

### 2026-10-23 bounded integration update

Fresh balanced practice produced real direct-skill improvement and real opponent-return harm; combined progress failed. Effective update magnitude was substantial. The preserved opponent-return head remains an experimental carry; the new endpoint is not promoted. A bounded fresh-reward coefficient test now asks whether their changes can combine in one existing head. This is an explicit integration scaffold with no architecture change, not the completed continuous learner. See [decision and conditional alternatives](2026-10-23-interpolation-selection.md). Next allocations require interpretation rather than automatic parameter sweeps.

The bounded2026-10-23 session then rejected the fixed mixture via unchanged-alpha0 selection and found no joint first-learning from weaker-phase fitness despite substantial movement. A secondary three-return signal is unreplicated. Retain RIGHTtheta6; next consider independent fresh-TRAIN confirmation of proposed updates, with a complete prospective design, rather than another automatic objective/rate sweep. [Session evidence](../reconstruction/2026-10-23-bounded-session-results.md).

### 2026-10-23 resumed research batch

The selected next trial confirms each combined mean-balanced search proposal on fresh training games from both phases before accepting it. This preserves the existing policy class and original search scale; it explicitly tests whether unconfirmed updates cause avoidable phase sacrifice. Every rejected/partial observation remains charged. A no-accepted-update run stops without redundant heldout evaluation. The current batch allows one primary experiment and, only after interpreting it, at most one justified follow-up inside the user's two-hour ceiling. [Selection, outside evidence and blind alternatives](2026-10-23-fresh-confirmation-selection.md). Reuse the established Sol XHigh agent for implementation and independent challenges; Astra owns selection, source synthesis and separate result reconstruction.

The resumed batch completed both weighted-proposal and executed-child confirmation trials: six cycles each, zero accepted updates in either, unchanged RIGHTtheta6,319648new TRAIN transitions and0DEV under registered futility. No new learning gain. Stop acceptance-rule variations for this batch. Next recommended question: joint balanced acquisition from the earlier DIRECT scaffold before RIGHT specialization, requiring genuine RIGHT acquisition plus LEFT and ordinary-game retention and explicit comparison to the preserved RIGHT candidate. This separate lineage is not silently substituted for accumulated learning. No next allocation or architecture change is selected by this note. [Evidence and costs](../reconstruction/2026-10-23-continuation-session-results.md).

### 2026-10-30 carried-learning scaffold completed

The inherited visual relational-controller family now has a qualified persistent
online selector. Two fresh streams acquired reliable returns, preserved learned
state across game resets, and adapted through hidden normal/reversed/normal
controls. Fixed late windows:1,063/1,071 returns; full live streams with exploration:
1,719/1,815. Matched reset and frozen controls support the learning/adaptation claim.
[Results, scope and curves](../reconstruction/2026-10-30-carried-learning-and-adaptation-results.md).
This is reward-based structural selection among16existing feedback primitives,
not general game learning or fullM1B. Next extension should increase what is
learned online (numeric feedback or sensorimotor association), not endlessly
polish frozen return rates. No additional run is launched after qualification.
