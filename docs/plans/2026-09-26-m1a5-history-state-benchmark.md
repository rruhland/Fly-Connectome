# M1A.5 benchmark: useful history beyond present observation

**Status:** Opt-in research protocol. This defines the test before adding a new state or dynamics learner. Production M1A is unchanged. M1A.5 is not complete until a learned visual state passes the representation gate and a downstream locally learned dynamics model passes the forecast gate.

## Observable input and claim

Primary evaluation receives events only. A separately reported hybrid arm may receive the same sparse visible intensity frames used in earlier research. The existing learned observer is frozen in both arms, but its original **every-frame rendered-intensity training teacher remains privileged**. Thus event-only *inference* is not yet proof of event-only *online learning*. Report the observer's full fast/slow/contrast state and the causal coincidence trace as the sensory control; no test may compare a learned memory only to the current raw event frame.

No model receives hidden masks, identity, direction, speed, object count, occlusion flag, disappearance flag, or future image during inference or representation learning. Simulator truth is for scoring and for an oracle ceiling only. All learned candidate weights use local/no-backprop rules until a different rule is explicitly reviewed.

## Data and controlled ambiguity

Render generic signed moving patterns on a 32×64 field behind a fixed opaque band, with deterministic ON/OFF event generation. Pair histories that have different approaches and hidden locations but the **same current visible image and zero current event** at a declared decision frame. Record the numerical distance between their *entire* frozen-observer/coincidence states; call a pair truly aliased only when that control cannot separate it with the registered frozen probe. Include one and two movers; matched trajectory pairs that either continue or disappear while hidden; varied gap duration; changed position, speed, motif, polarity, and camera cadence. Some indistinguishable continue/disappear pairs have no correct deterministic answer before reappearance: score probabilities and false confidence, not forced identity.

Build development and held-out scene families before training a new latent. Hold out positions, speeds, shapes, and mixed-entity combinations jointly. Keep a simple no-occluder stream as a regression check. The test must include both real missing observations and a later contradictory event that should correct stale state. The visible renderer and any sparse intensity frame must contain the occluder, never the hidden object.

## Strong controls and readouts

Compare at the same input and decision frames:

1. Frozen observer **full state** plus causal local coincidence primitive.
2. A fixed multi-timescale retinotopic memory with at least the candidate's temporal span and a matched state-size budget.
3. A generic deterministic detect/associate/constant-velocity tracker that uses only observed visual evidence, with no Pong rules. It is a strong engineered baseline rather than an allowed learned world-state.
4. Time-shuffled history and an oracle hidden-state ceiling as negative/sanity controls.

Use one frozen, spatially shared local readout calibrated only on *visible* objects in separate development scenes, then apply it without oracle crops to hidden-state localization. An additional flexible offline probe may measure latent capacity, but does not count as autonomous discovery. Score hidden occupancy localization, identity continuity/reacquisition, false persistence after contradiction, clean/noisy robustness, and calibrated probability on observationally ambiguous outcomes. Report every split and per-case counts; avoid collapsing a large failure into one aggregate score.

## Sequential gates

**Benchmark validity:** At least one held-out ambiguous split must leave the complete sensory control materially below the oracle, while the generic tracker or equal-memory control demonstrates that a causal history can help. If controls already saturate, enlarge the ambiguity instead of testing another latent. If no allowed history can distinguish outcomes, mark that split as uncertainty-only.

**M1A state:** A locally learned state must exceed both the full sensory and matched fixed-memory controls on hidden localization and reacquisition by a predeclared material margin, preserve visible position/speed/shape transfer, and avoid confident false persistence. This is the prerequisite to a forecast head. Connectome motion input, including the opt-in measured T5 signal, is evaluated by paired ablation after a generic state works; the graph is not assumed to contain a complete world-state.

**M1A.5 dynamics:** Freeze an accepted state, then train a separate local/no-backprop transition on observed sequences. Evaluate future spatial belief at multiple horizons and next visible reappearance, not only next event timing. It must beat persistence, the fixed tracker, and the same transition trained on the sensory control under held-out scene changes, with calibrated uncertainty on unknowable outcomes. Training throughput and state memory cost are reported. Only after both gates pass is a concrete production promotion prepared for user review.

One development pass may fix benchmark-generation or scoring bugs before candidate training. Held-out results do not authorize threshold tuning or a chain of narrow candidate variants.

## Frozen first candidate gate (after control development, before candidate training)

The clean constant-motion scenes are regressions: 16/16 single-mover top-32 sites and at least 14/16 hidden entities in the two-mover top-64 test. The discriminating split uses a repeated 1/2/3 displacement cadence and scores the second-to-last fully hidden frame; its 16 held-out combinations change position, motif, starting cadence phase, polarity, and direction from the visible-only development cases. The fixed constant-velocity tracker reaches 12/16 at top-8 and top-32, while the constant-acceleration extrapolator reaches 0/16. A learned state or dynamics scaffold must reach at least **14/16 top-8 and top-32** here without hidden-state targets or per-scene tuning, while meeting the constant-motion regressions. This is a minimum promising signal, not full M1A.5 acceptance. Full acceptance additionally requires identity/reacquisition, contradiction correction, uncertainty calibration, teacher feasibility, and changed dynamics beyond the training cadence.

**Result of the one registered candidate:** [The learned cadence scaffold failed](../experiments/2026-09-26-local-cadence-memory-findings.md). Its stronger frozen multi-target control also saturated the changing-motion top-32 score, so that split cannot be used as a final acceptance gate. Preserve these results and replace the discriminating split with a context-dependent hidden change before another architecture experiment.

## Registered context-dependent successor

A stationary two-pixel visual mark near an entrant's path predicts a vertical path deflection behind an opaque band. The mark is visible before the gap, but the deflected entity is not. The relation is learned from later visible reacquisition; the learner receives neither the mark's sign nor hidden path truth as a label. Held-out scenes change shape, position, speed, direction and polarity. In the 16-case event-only control run, complete sensory, fixed leaky memory, constant velocity, literal constant acceleration, and fixed multi-target tracking each scored 0/16 top-8 and top-32 at the late hidden frame. Hybrid sparse frames gave the same results. The oracle ceiling is 16/16. Shuffled mark/outcome association is a required negative learning control.

Before evaluating the local context candidate, freeze a minimum promising criterion of **at least 12/16 top-8 and 14/16 top-32** on this contextual split, with at least 14/16 top-32 constant-motion and 14/16 top-64 two-entity regression hits. The candidate must exceed its own frozen and shuffled-association controls. This synthetic relation tests whether online local association can influence hidden world-state; it alone cannot establish generic environmental transfer or full M1A.5 completion. Post-reveal contradiction and uncertainty on true disappearances, teacher feasibility, and unrelated dynamics remain mandatory for full acceptance.

**Outcome:** The [aligned local-context association](../experiments/2026-09-26-local-context-memory-findings.md) passed that clean promising gate at 16/16 context top-8/top-32 and 16/16 on both regressions, while frozen/shuffled context controls scored 0/16. The [teacher and noise audit](../experiments/2026-09-26-event-only-observer-teacher-findings.md) found only 1–5/16 noisy context top-32 and incomplete post-reveal reacquisition. M1A.5 therefore remains unaccepted. The next acceptance candidate must use a confidence-weighted multi-entity belief/association state and meet the same clean gates plus a declared noisy and contradiction gate; the original success cannot be promoted on clean accuracy alone.

For that next candidate, retain event-only sensing and event-only observer teaching as the primary arm; sparse visible frames are a separately reported additional-sensor arm. Before training, require at least **14/16 clean context top-32**, **12/16 familiar-noise context top-32**, **8/16 heavy-noise context top-32**, **14/16 constant-motion top-32**, and **14/16 two-mover top-64**. After expected reveal, require at least **14/16 continued cases above .5** at the true location and at most **2/16 vanished cases above .5** at the counterfactual location, with identical pre-reveal predictions for observationally matched pairs. These thresholds are a bounded experimental screen. Full M1A.5 acceptance also requires broader rotated motion, unrelated contextual relations, camera-domain transfer, calibrated uncertainty, and throughput measurement before any production architecture revision is proposed.

**Second candidate outcome:** [Causal event corroboration before tracklet formation](../experiments/2026-09-26-corroborated-context-memory-findings.md) retained 16/16 clean contextual hits but reached only 4/16 familiar-noise and 2/16 heavy-noise top-32 with event-only training/inference; sparse-frame inference gave 7/16 familiar-noise. It also reacquired only 10/16 continuing paths above .5. It fails the registered noisy and contradiction gates. Further threshold/decay variations of this family are out of scope; a genuinely confidence-weighted multi-hypothesis visual state is the next substantial architecture experiment.
