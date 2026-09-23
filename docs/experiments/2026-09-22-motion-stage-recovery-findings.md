# Bounded motion-pathway rest-current recovery: partial input activity, no direction code

Status: approved opt-in experiment; **no production dynamics, checkpoint weights, topology, transmitter signs, delays, or plasticity changed**. The frozen initial full-graph checkpoint, matched blanks, 8 ticks/frame, and warm-state reset follow the [proposal](2026-09-22-motion-stage-recovery-proposal.md) and [baseline audit](2026-09-22-motion-stage-audit-findings.md). Scripts and compact data are versioned beside this report; per-neuron files remain in ignored `runs/motion-stage-recovery-v1/`. The source checksum in every report matches the baseline audit.

## Preregistered calibration

Each class was varied *alone* at rest currents `0, 0.85, 0.95, 1.05, 1.15` on a right-moving ON/OFF bar at x=18, with matched blanks. Selection required at least five local cells changed versus blank, an arrival at a measured T4/T5 target, blank activity below 0.01 spikes/neuron/tick, and finite state. The current class roster and all other parameters stayed fixed.

| Class | First passing current | Local cells changed | Arrivals to target | Blank class rate | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| Mi4 | none | at most 2 / 132 | 128 at 1.05, 63 at 1.15 | 0.00034 at 1.05 | Current can cause sparse firing but does not make it visually responsive |
| Tm4 | 0.85 | 107 / 132 on OFF bar | 864 | 0 | Measured T5 arm can carry stimulus-dependent spikes |
| Tm9 | 0.85 | 7 / 133 on OFF bar | 99 | 0 | Measured T5 arm becomes weakly active |

The maximum whole-network spike fraction stayed below 0.014 per tick and states were finite. Tm4/Tm9 activity is therefore not merely broad autonomous firing. Mi4 is a decisive failure of the **rest-current-only** idea: pushing its current above threshold does not create the needed stimulus-driven class response. The selected combination was consequently only Tm4=0.85 and Tm9=0.85; Mi4 remained at baseline zero. It was tested as a partial OFF-pathway diagnostic, **not promoted as a model profile**.

## Frozen combined follow-up

The original two-location, two-direction, ON/OFF moving bar was rerun with those two selected rest currents and no learning. Local T4a-d stayed at **zero spikes** for all conditions, exactly as in baseline. T5 OFF responses increased, but direction preference did not become stable. Local right-minus-left OFF spike contrasts were:

| T5 subtype | Baseline x=18 / x=46 | Selected x=18 / x=46 |
| --- | --- | --- |
| a | -1 / +3 | -2 / +6 |
| b | +3 / +2 | 0 / +3 |
| c | +3 / +3 | -4 / +1 |
| d | +1 / +3 | -3 / +6 |

The combined setting fails the proposed advance checks before held-out speed or location probes: Mi4 remains inactive, T4 ON is silent, and T5 OFF signs reverse or vanish between the two calibration positions. Running the dot, static, and holdout panel would not rescue that preregistered failure. No full-Pong training run was made. M1A remains unmet.

## What the measured graph retains, and what the model discards

The signed edge inventory into Mi4 points to a pathway-level issue, not missing T4 edges. Its largest measured incoming contact totals include inhibitory Dm4 (20,372; no spikes in the bar audit), TmY16 (16,969; no spikes), and Mi9 (16,852; active), alongside excitatory Mi1 (13,939; active). This inventory does **not** by itself establish the net conductance or causal contribution of any class. It explains why increasing Mi4's intrinsic current may be insufficient: visually patterned upstream drive and sign balance are also needed.

Mi4 carries a small, local *subthreshold* stimulus signal despite emitting no spikes. Across 132 local cells at x=18, the ON-right bar changed mean transit voltage from matched blank by -0.00144 on average (57 cells exceeded 0.001 absolute change); OFF-right changed it by -0.00543 (101 cells). At x=46, the corresponding means were -0.00265 and -0.00531 across 139 cells. The sign and small size matter: a graded-release mechanism could in principle convey a **decrease** in inhibitory Mi4 output, but this averaged voltage is not a measured release function and does not prove that such a mechanism would produce motion tuning. Tm4/Tm9 also showed subthreshold stimulus modulation before their rest-current rescue.

The anatomical spatial motif is present in the pinned graph. Using contact-weighted Mi1 input as a T4 reference, Mi9 and Mi4 presynaptic assigned-hex centroids lie on opposite sides of T4a (`[+0.49,-0.61]` versus `[-0.40,+0.56]` axial columns); the offsets reverse for T4b and rotate for T4c/d. T5 Tm4/Tm9 inputs likewise have nonidentical offsets. This is an **inferred column-level receptive geometry**, not a synapse-coordinate dendritic reconstruction. Tm3 has 2,394 anatomical edges into T4a but no `assignedOlHex` annotations, so its column offset is absent from this audit, not its connection. The current single-compartment point neuron sums these pathways before any spatially localized conductance interaction.

Primary physiology places motion selectivity in T4/T5 dendrites and reports that T4's spatially separated excitatory and inhibitory inputs can interact supralinearly through conductance and disinhibition ([T4 input physiology](https://www.sciencedirect.com/science/article/pii/S0896627317301939), [T4 dendritic biophysics](https://www.nature.com/articles/s41586-022-04428-3), [T5 contrast-opponent inputs](https://www.nature.com/articles/s41467-021-24986-w)). The simulation results are consistent with missing signal expression and dendritic integration, but they do not prove which omission is decisive. The next test should isolate those two possibilities in a small frozen, opt-in experiment before revising M1A plasticity. See [graded/compartment proposal](2026-09-22-motion-stage-signal-integration-proposal.md).
