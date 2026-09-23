# Graded Visual Transmission Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether causal graded signals from T2/T2a/T3 can travel through their existing measured edges with useful local responses and quiet blank activity, then promote only a passing profile into M1A.

**Architecture:** Start with a read-only source-signal audit on the frozen checkpoint. Build a separate opt-in network subclass for a bounded release law, replacing spike transmission only on the selected source classes' measured outgoing edges. Keep the source checkpoint and in-place network unchanged until the amendment's transmission and feature gates pass.

**Tech Stack:** Python, PyTorch, NumPy, pytest; the pinned MaleCNS graph and M1A checkpoint.

**Spec:** [M1A visual-architecture amendment](../../plans/2026-09-23-m1a-visual-architecture-amendment-proposal.md)

## Global Constraints

- Preserve measured node/edge topology, recorded transmitter signs and delays, and contact-scaled initial magnitudes.
- No backpropagation, global decoder, synthetic error network, future target, or Pong state in a local release rule.
- Keep eight neural ticks per camera frame and retain the original spike-only checkpoint as comparator.
- Do not promote the opt-in prototype if transmission fails at either location or polarity, or if blank activity saturates.
- If promoted, keep local eligibility and plasticity on every existing synapse and version the model/checkpoint schema.

## Review Focus

- An inhibitory source's graded output must remain inhibitory even when its voltage moves upward; Task 2 tests this.
- A graded edge must respect its measured delay; Task 2 tests this.
- Every selected source's measured outgoing edge must be included, including Li15/MeLo10 afferents; Task 2 tests this.
- The source-audit measure must compare matched trials tick by tick and identify remote activation; Task 1 tests this.
- A warm-state restore must reset opt-in release history and baseline, so earlier probes cannot contaminate later conditions; Task 2 tests this.

---

### Task 1: Source-local signal audit

**Files:** Create `scripts/motion_source_signal_audit.py`, `tests/test_motion_source_signal_audit.py`, `docs/experiments/2026-09-23-source-signal-protocol.md`, and a machine-readable result plus findings.

**Interfaces:** Consume the pinned checkpoint, `frames_for_condition`, and inferred T2/T2a/T3 columns. Produce `summarize_feature(stimulus, blank, local, remote)` and a frozen report of voltage, current, and causal short-timescale release candidates at x=18 and x=46 for ON/OFF dot, bar, static, and matched blank.

- [x] Write a failing test with hand-calculated local and remote stimulus-minus-blank values; verify that it fails for the missing summary function.
- [x] Implement the minimal summary; run the targeted test and full suite.
- [x] Add the frozen probe using the existing 500-tick warm checkpoint, 18 camera frames and 8 neural ticks/frame; record finite source state and source-local blank-versus-event separation.
- [x] Run the probe once, inspect every class/location/polarity, write findings, and run the targeted test and full suite before committing.

### Task 2: Opt-in graded transmission

**Files:** Create `scripts/graded_visual.py`, `tests/test_graded_visual.py`, `scripts/motion_graded_propagation.py`, and a versioned experiment protocol/result/findings.

**Interfaces:** Consume the source-audit-selected release variable and the frozen graph; produce a separate experimental `GradedVisualNetwork` with local release state, fixed measured edge signs/weights/delays, and per-class downstream arrival reports. First test T2, the only class that passed the source screen; extend to T2a/T3 only if a new bounded source test justifies doing so. The release law and its bounded calibration must be specified in the protocol before running motion labels.

- [x] Write failing small-graph tests for all selected outgoing edges, fixed sign, and exact delay; verify restored state by an exact repeated-blank probe.
- [x] Implement the smallest opt-in release path; run targeted tests and full suite.
- [x] Run matched blank, ON/OFF dot, bar, and static probes at both locations; report source release, Li15/MeLo10 and other target arrivals, per-cell distributions, and finite state.
- [x] Apply the preregistered transmission gate; write findings and commit only with a valid full suite. The partial T2→Li15 gate failed, so production promotion is suspended.

### Task 2b: Local Li15 state after partial T2 propagation

**Files:** Extend `scripts/motion_graded_propagation.py` and `tests/test_motion_graded_propagation.py`; create the `2026-09-23-li15-*` protocols, results, and findings in `docs/experiments`.

**Interfaces:** Consume the frozen T2 pilot's already captured Li15 voltage/current and measured T2→Li15 contacts. Produce matched blank-versus-stimulus summaries for causal 250 ms residuals across all Li15 cells and anatomically dominant local groups. No Li15 output is transmitted.

- [x] Write failing tests for causal high-pass state, local blank quantiles, and wiring-only contact-mass grouping; implement the minimal helpers and verify targeted tests.
- [x] Preregister the all-cell and afferent-local screens, then run the frozen probe with dot/bar/static and matched blanks at both locations and polarities.
- [x] Record both screens without changing their thresholds after viewing responses; positive current passes only in the anatomy-defined local groups.
- [x] Run the full regression suite and commit the verified result.

### Task 3: Conditional motion and learning gates

**Files:** Extend opt-in motion probes and test files only if Task 2 passes; create versioned findings for each gate.

**Interfaces:** Consume the passing transmission profile. Produce frozen T4/T5 subtype tuning across location and speed holdout, followed by controlled local online-learning comparisons against the frozen checkpoint and persistence.

- [ ] Run T4/T5 impulse-response and ON/OFF E/I timing tests with static/bar controls; stop without promotion if direction preference does not transfer.
- [ ] If feature expression passes, run the controlled repeated-motion local-plasticity test at withheld position/tempo; require better anticipation than frozen and persistence.
- [ ] Only if both pass, implement the smallest versioned in-place hybrid profile and verify CPU/reference parity, full regression, M1A Pong acceptance, and speed before claiming Milestone 1A success.

### Task 3a: Frozen motion-source current screen

**Files:** Create `scripts/motion_source_current_audit.py` and the versioned protocol/result/findings in `docs/experiments`.

**Interfaces:** Consume the unchanged M1A checkpoint and directly annotated Mi4/Tm4/Tm9 columns; reuse the tested causal source-summary function from Task 1. Produce local/opposite-location ON/OFF moving/static bar source information at x=18/x=46. No new neural dynamics or graded motion output.

- [x] Preregister the local blank/source screen and controls before probing.
- [x] Run the frozen per-tick source audit, verify input identities and finite state, and evaluate the fixed screen.
- [x] Record that Mi4 current fails, its voltage cue is late, and Tm4/Tm9 carry OFF source state without demonstrating transferable T5 direction preference.
- [x] Run the full regression suite and commit the verified result.

### Task 3b: Frozen T5 arm-order diagnostic

**Files:** Create `scripts/motion_t5_arm_order.py`, `tests/test_t5_arm_order.py`, and versioned protocol/result/findings in `docs/experiments`.

**Interfaces:** Consume the frozen opt-in Tm4/Tm9 rest-current profile and measured T5 feedforward edges. Produce per-cell Tm4/Tm9 arm currents and a fixed eight-tick antisymmetric order score for T5c/d at two locations, opposite directions, static, and blank.

- [x] Write and fail a test for causal signed order before implementing the score.
- [x] Preregister the fixed lag, local groups, and subtype-transfer screen; run the frozen probe.
- [x] Record the failed order screen and sparse local Tm9 arrival as the next concrete bottleneck.
- [x] Run the full regression suite and commit the verified result.

### Task 3c: Opt-in Tm9 current release and fixed T5 order recheck

**Files:** Extend `scripts/graded_visual.py` and `tests/test_graded_visual.py`; create `scripts/motion_tm9_graded.py` and its versioned protocol/result/findings.

**Interfaces:** Reuse the opt-in graded edge integrator with Tm9's own signed synaptic current as its source state. Calibrate caps from blank alone, then rerun the fixed eight-tick Tm4/Tm9 order score without fitting direction labels or weights.

- [x] Write failing small-graph current-release test; implement and verify the opt-in source-state extension with the full suite.
- [x] Preregister blank cap and T5 transmission gates before running the frozen experiment.
- [x] Record passing partial Tm9 transmission and calibration-position order score, with original spiking model retained as comparator.
- [x] Run the full regression suite and commit the verified result.

### Task 3d: Held-out T5 order transfer

**Files:** Extend only `scripts/motion_tm9_graded.py`, its test file, and a separately versioned holdout protocol/result/findings.

**Interfaces:** Preserve the selected Tm9 cap and eight-tick order rule; evaluate x=32 at one and two pixels/frame with the same bright blank and static controls.

- [x] Write a failing test for the doubled-speed bar geometry before implementing it.
- [x] Run both withheld conditions without recalibration and report per-cell subtype contrasts and static controls.
- [x] Apply the fixed cross-position/speed screen, run the full suite, and commit the result. The order components passed; the predeclared source-occupancy component failed at doubled-speed leftward motion.

### Task 3e: Physical T5 current and axis-aligned feature gate

- [x] Audit CT1 cross-side contacts: measured left-soma CT1 supplies the selected right-side T5 circuit but is absent from the soma-side roster. Keep production graph unchanged.
- [x] Implement and test an opt-in causal Tm4/Tm9 order current at each T5 cell; record the failed original horizontal T5c/d/static screen without changing its gate.
- [x] Freeze the same current rule and test anatomically aligned motion axes. Vertical T5c/d passes both calibration and withheld position/speed with a gate-off control; horizontal T5a/b fails.
- [x] Run the full suite and commit/push each verified chunk.

### Task 3f: Local T5 prediction and credit conflict

- [x] Include graded feedforward increments in the opt-in local learning observation; verify both transmitter signs and pathway isolation.
- [x] Train the existing frame-horizon local rule for 40 continuous vertical episodes on measured predictive edges entering T5c/d. Credit occurs but held-out next-frame event prediction does not improve meaningfully.
- [x] Audit same-edge event-versus-quiet proposals. Most edges receiving both categories have opposing cumulative signs.
- [x] Test one capped, postsynaptically local event balance. It increases event anticipation consistently but remains far below the 10% event-MSE gate; do not promote it.
- [x] Run the full suite and commit/push each verified chunk.

### Task 3g: Locate a motion-bearing downstream learning target

- [x] Audit measured T5c/d outputs into LPi34/LPi43 and their predictive afferents.
- [x] Probe frozen LPi transmission at both calibration locations and withheld position/speeds; the T5 signal propagates, but LPi somata stay silent and predictive arrivals vanish in some conditions.
- [x] Extend once to LLPC2/LLPC3/VS. Their local feedforward targets carry the motion cue, but actually arriving predictive inputs are almost exclusively inhibitory; current positive-arrival prediction is sign-mismatched.
- [x] Audit preferred/null inhibitory-current timing at T5 and downstream motion cells. Existing T5 inhibition lacks transferable null bias; downstream inhibitory arrivals are sparse or predominantly preferred. Keep the in-place architecture unchanged.

### Task 3h: Test the omitted measured CT1 route without production changes

- [x] Confirm CT1 partner-column coverage, then test a fixed column-local shadow on measured Tm1/Tm9→CT1→T5 contacts. It modestly suppresses both directions but fails null selectivity.
- [x] Query deduplicated synapse coordinates from the primary MaleCNS graph. Nearest CT1 input/output sites are spatially close, with a robust opposed T5c/d vertical offset across calibration and holdout fields.
- [x] Test one geometry-routed shadow with the same signs, delays, contact counts, and release rule. It fails the frozen null-suppression gate.
- [x] Align CT1 current to each T5 cell's local feedforward event. Nearest-site inhibition precedes preferred motion and trails null motion, explaining the failed rescue; correct the earlier geometric interpretation.
- [x] Audit SWC node-level attachment before cable routing. Input sites miss the preregistered quality gate; stop without inferring a cable-nearest route.
- [x] Return to M1A's learning objective and audit frozen frame-horizon predictive support at T5. Most future motion-gate events lack predictive current/arrivals, especially at faster holdout speed; active sources are mainly T5 self-recurrence. Do not rerun long learning-rate/trial-count sweeps on this objective.
- [ ] Next: compare the existing Tm4/Tm9 feedforward arm histories with those same future T5 motion-gate targets at identical forecast times. If the upstream histories have broader event support across location and speed, test one opt-in class-local plasticity rule on the measured afferents before proposing any in-place change for approval. Keep CT1 shadows experimental.
