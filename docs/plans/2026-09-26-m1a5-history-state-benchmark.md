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
