# Proposed M1 visual operating-point correction

Status: proposal for review, not implemented or a replacement for the approved design.
Requested after the real-data M1A pilot exposed downstream silence. Preserve the
failed-pilot checkpoints and results as the zero-background control.

## Evidence and diagnosis

The real T5 visual graph has 47,413 neurons and 1,377,103 edges. After 3,000 training
steps, held-out scripted sequences activated only L1-L3. A frozen ON moving edge
with global current scale from .005 to 1.0 still activated no downstream population.
These observations identify an operating-point problem; they do not prove that the
entire selected circuit cannot learn. See the committed pilot JSON.

The current implementation assigns positive ON impulses to inhibitory L1, starts
all cells at zero, and supplies no intrinsic background drive. More L1 spikes then
increase inhibition onto otherwise quiet targets. Increasing synaptic gain cannot
turn that inhibition into excitation.

Primary experiments show L1 and L2 hyperpolarizing to light increments and
depolarizing to decrements. Mi1/Tm3 responses reverse the L1 sign, whereas Tm1/Tm2
responses preserve L2's sign. Signals also acquire different temporal responses
across these synapses. [Yang et al., 2016](https://pmc.ncbi.nlm.nih.gov/articles/PMC5606228/)

Glutamate imaging and receptor perturbations support inhibitory sign inversion in
the ON pathway, with additional GABAergic circuit contributions; it is not simply
a positive feedforward relay. [Molina-Obando et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC6845231/)

Electrophysiology reports distinct temporal responses for Mi1/Tm3 and Tm1/Tm2.
Measured response latencies are compound circuit responses, not direct estimates
of a model membrane time constant. The paper also excludes a particular unstable
tonic-response recording class as likely nonphysiological: that is not evidence
for assigning all modeled cells persistent high firing.
[Behnia et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4243710/)

Behavioral state changes baseline activity and temporal tuning in T4 and its input
populations, supporting attention to operating points rather than assuming a
universal silent baseline. This does not supply numerical LIF currents or authorize
a new arousal network. [Strother et al., 2018](https://pubmed.ncbi.nlm.nih.gov/29255026/)

## Recommended bounded change

Retain the same adaptive current-based LIF equation, with fixed class parameters:

```
tau_m[class(j)] dv_j/dt = -v_j + I_rest[class(j)]
                         + I_existing_synapses[j] + I_events[j]
```

1. Add fixed, explicitly versioned intrinsic current and membrane/synaptic time
   constants selected by annotated cell class. Defaults retain the old model.
   Start with the lamina and documented first-order medulla classes; use a small
   shared parameter table, not independent fitted parameters for each body.
   No simulator state, reward, motor state or task-dependent signal controls these
   parameters. Do not add direct sensory drive to medulla or L4.
2. Replace positive-only L1 ON injection with a fixed sensory transduction profile
   for L1-L3: ON events cause hyperpolarizing current and OFF events depolarizing
   current around their operating point. Each event still has exactly one emitted
   polarity; both channels are not emitted for a single transition. Anatomical
   projection and event-only observations remain fixed. A short decaying sensory
   current state carries the impulse between neural ticks; it is not an image
   encoder, latent object state or additional synapse.
3. Allow basal discharge only where needed for the spiking approximation to
   represent reductions in transmission. ON-induced reduction of inhibitory L1
   discharge can then disinhibit its measured targets. There is no firing-rate
   floor, automatic upscaling to maintain firing, or additional homeostatic rule.
   Individual cells can remain silent under inhibition and adaptation.
4. Keep the existing predictive and reward updates unchanged, including the
   anatomy-defined current partition and fixed observation edges. Intrinsic current
   is not counted as a sensory observation or prediction. Check whether the
   existing rectified current prediction variables remain informative under this
   operating point; do not silently change their meaning if they fail.

This is a mechanistically motivated spiking approximation to largely graded early
visual circuitry, not a claim that these cells normally emit these spike trains.
Changing all synaptic signs or using graded transmission is outside this proposal.
Changing rest voltage numerically without changing the equilibrium relative to
threshold would not solve the lack of baseline transmission.

## Calibration and verification before adopting a canonical profile

First write a hand-checkable inhibitory-chain test: suppressing a tonically active
presynaptic cell releases its existing target; the same suppression cannot excite
an unconnected cell. Test both event polarities, L4 exclusion, absence of task input,
and exact checkpoint resume of sensory currents. Preserve existing invariant tests.

Preregister a small candidate table in normalized model units, with explicit source
or approximation labels, before running calibration. Do not label any numerical
baseline Hz, resting current or time constant as measured without a matching source.
Do not choose parameters using Pong scores or reward. Compare candidates using
uniform backgrounds, isolated ON/OFF flashes, moving edges at multiple directions
and speeds, and post-stimulus recovery with learning disabled. Save response traces,
baseline/stimulus firing, peak activity and decay for both old and candidate models.

Use one fixed warm-up protocol independent of the game for each frozen probe.
Report baseline-subtracted responses alongside absolute firing, so adding tonic
activity cannot masquerade as visual processing. Reject numerical instability or
persistent runaway activity; physiological tuning remains exploratory, as agreed.
If no small current-based profile gives useful propagation, report that failure
rather than broadening the architecture or hand-coding motion detectors.

Once the profile is chosen, freeze it identically across T1/T3/T5, version the full
configuration and new checkpoint schema, and rerun the paired M1A experiment. Include
a zero-event predictor alongside persistence to expose the sparse-event trivial
baseline. Resume M1B learning only with honest M1A results. No existing trained
artifact is overwritten or silently reinterpreted under the new dynamics.

## Approval boundary

The user authorized investigation and a proposal. Implementation of this correction
still needs their decision because it changes the canonical sensory-current polarity
and intrinsic operating point. The approved topology, transmitter signs, local
plasticity, discrete-spike and no-backprop requirements remain unchanged.
