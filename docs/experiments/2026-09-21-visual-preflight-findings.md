# Frozen precursor-response preflight

The fixed mixed-sign motif fails the frozen precursor-response gate: **ON0/10,
OFF1/10**, with8/10 per polarity required. Execution took6.76seconds. No training
was started, and no neuron, learning, sign, topology or weight parameters changed.
This shows insufficient current precursor response in the selected motif and
stimulus, not inability of the architecture or local rule to learn.

## Selection and controls

The [protocol](2026-09-21-visual-preflight-protocol.md) keeps previous target
L1 38366, the original rightward dot (row16,pixels28..36) and8.33ms forecast lead.
The63-neuron,274-edge induced crop contains all three measured prediction sources
into that L1 and every measured direct input to those sources, plus the trajectory
sensory neurons. This repairs omission of immediate source inputs in the previous
crop; it does not reproduce the full network's operating state. There remain
6,274 incoming and6,969 outgoing boundary edges omitted from the selected roster.

The measured forecast inputs are:

| Source | Fixed sign | Magnitude | Delay |
|---|---:|---:|---:|
| Mi1 33462 | + | 0.095 | 1 tick |
| Tm3 85937 | + | 0.060 | 1 tick |
| C2 332251 | - | 0.180 | 1 tick |

Ten paired probes use seed9011 and variable blank intervals. Every full stimulus
and its ON/OFF blank controls start from identical warmed neural state. Each
control blanks images before its target event; forecasts are read eight ticks
before that event. Both conditions therefore have the same initial state and
neither can use the future target to generate its scored forecast.

## What the actual spikes show

**C2 never fires in the ten full probes.** Its negative physical trace stays zero,
so it supplies no negative forecast for the ON event. The other two forecast
inputs are excitatory. Their tiny ON-forecast residuals in two trials equal the
blank controls and reflect background activity, not spatial anticipation.

All five measured direct inputs to C2 are retained:

| C2 input | Signed magnitude | Arrivals from recorded full-probe spikes |
|---|---:|---:|
| Pm12 10531 | -0.025 | 0 |
| Mi1 36533 | +0.035 | 10 |
| L1 38366 | -0.265 | 51 |
| Mi1 55976 | +0.035 | 10 |
| Tm4 70997 | +0.030 | 0 |

Across these probes the summed positive arrival impulses are+0.700 versus
-13.515 inhibitory, a19.3-fold difference in magnitude. These totals are not
mean current, membrane voltage, or proof of a voltage-level causal explanation.
They show that C2 receives weak excitation relative to inhibition in this frozen
operating state. Its approved resting current equals the baseline threshold1.0;
it does not supply autonomous firing by itself. Upstream boundary omissions and
the point-neuron dynamics remain possible contributors. No tonic-current tuning
or synthetic drive was introduced to make it fire.

**Only one OFF probe has sufficient stimulus-dependent positive anticipation.**
With14blank frames, Mi1 and Tm3 spike11 and9ticks before the OFF event, respectively.
Their one-tick delays let both arrivals contribute by the issuing tick8ticks
before the event. The full forecast is0.122628 versus0.111344 in its blank control.
It passes the stated per-trial response threshold. The other nine probes do not;
these ten deterministic probes are not independent statistical evidence of
generalization, and this is not a trained result.

The signed currents reconstructed from actual spikes, fixed delays and retained
weights match every scored forecast within8.38e-9 (required tolerance1e-6).
This validates the timing reconstruction, including negligible residual error
from omitted pre-recording warmup history. All checked raw neural states remain
finite and all frozen weight comparisons match exactly. The source checkpoint's
checksum is unchanged.

## Why a directly driven temporal motif is the next useful test

Before selecting this fixed motif, a read-only inventory found137 measured
predictive L2->L1 edges:136 connect neurons mapped to the same retinal column.
The sole cross-column pair is22312->24823. None spans the tested interior-row
rightward2-4pixel gaps. This inventory does not exclude indirect spatial pathways,
other directions, or other mappings. It does make most of these direct edges
natural candidates for a local temporal-prediction test.

Recommend a repeating two-position dot sequence centered on a measured L2->L1
motif, with actual event input driving the L2 precursor. Keep the fixed signs,
delays, neuron dynamics and local update; retain positive feedback support if
scoring both polarities. Verify the precursor response in frozen paired probes
before any training. This would test whether a directly available sensory
precursor can be learned without depending on the currently silent C2 route.
The revised stimulus/motif must be specified before examining its learning score.

Stop this preflight here. Do not launch a speed, horizon, crop or excitability
sweep. This result does not yet justify changing the learning rule. A frozen
net-current response is a conservative operational gate, not a mathematical
requirement for eventual learning: plasticity could unmask opposed contributions.
M1A remains unmet; there is no M1B advancement or new full-Pong experiment.

## Reproduction and verification

Run `.venv/Scripts/python scripts/visual_preflight.py --output runs/visual-preflight-v1`
using a fresh output directory. The [results](2026-09-21-visual-preflight-results.json)
contain the checkpoint checksum, graph identity, retained parent-edge indices,
individual paired forecasts and source spike timing. Thirty complete small
frozen-probe artifacts are retained locally under `runs/visual-preflight-v1`.
The [C2 input diagnosis](2026-09-21-visual-preflight-c2-inputs.json) contains the
per-trial recorded spike-arrival totals; initial warmup-history arrivals are
excluded from those totals.

207 tests pass;4CUDA tests skip. New tests cover exact delay/sign/decay and rejection
of tonic-only, wrong-sign and canceled responses. Focused review found no code
blockers and clarified the limits of the frozen-response gate above.
