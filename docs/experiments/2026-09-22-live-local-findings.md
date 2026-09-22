# Live local learning improves prediction but exposes a slow-tempo sign conflict

The approved shared-magnitude timing rule has now learned *inside* the neural
simulation. Updating its existing predictive magnitudes changed future spikes
and subsequent eligibility, while keeping topology and transmitter signs
fixed. The selected network reduces held-out prediction error below its
initial state and persistence across every tested tempo and both continuous
challenges. It does **not** yet meet the small motif's robust anticipation
criterion: OFF remains below .1 at tempos2 and6, and ON has the wrong sign in
the steady tempo6 segment of the continuous challenge. Pong M1A remains unmet.

## Selection and held-out results

The [fixed protocol](2026-09-22-live-local-timing-protocol.md) trained one
pass across 42 trials at rates .1 and1.0. Nine validation trials selected .1
because its ON/OFF anticipation was .271/.125 with 2.35% quiet alarms. Rate
1.0 yielded .600/.035 and failed OFF. No held-out score influenced selection.

| Held-out condition | Initial MSE -> learned MSE | Persistence MSE | Learned ON | Learned OFF | Quiet alarms |
|---|---:|---:|---:|---:|---:|
| Dwell2 | .1738 -> .1470 | .3516 | .297 | **.068** | 3.56% |
| Unseen dwell3 | .1459 -> .0871 | .2991 | .667 | .297 | 2.93% |
| Dwell4 | .1282 -> .0840 | .2602 | .533 | .231 | 2.49% |
| Dwell6 | .1023 -> .0759 | .2065 | .303 | **.078** | 1.92% |
| Continuous switches | .2288 -> .1627 | .4683 | .489 | .165 | .96% |
| Continuous with omissions | .2239 -> .1551 | .4585 | .510 | .169 | 1.27% |

Whole-stream means hide the central failure. In steady dwell6 of the standard
continuous stream, ON anticipation is **-.291** and OFF only .057; the timing
gate was open on all eight ON and all eight OFF events. In the omitted stream,
the first omitted event causes the expected false alarm, the next is suppressed,
and the first two returning events are missed. The third returning ON event
has the wrong sign (+.337 forecast against a -1 target).

## Local cause, checked against the actual neural run

The final live weights on target L3's L2 and L1 incoming edges are1.720 and
2.714, versus original .025 and .045. Replaying the trained neural weights
with the independently validated frozen collector exactly reproduces every
saved spike, raw forecast, local target and sensory state. The sum of its
signed edge traces, clipped to [-1,1], reconstructs the forecast after
warmup. Its steady dwell6 event contributions are:

| Event due | L2 signed contribution | L1 signed contribution | Net signed forecast |
|---|---:|---:|---:|
| ON | **-.609** | +.319 | **-.291** |
| OFF | +.057 | approximately 0 | +.057 |

Increasing the shared L2 magnitude strengthens weak OFF but makes ON more
negative at slow tempo. Decreasing it repairs ON but weakens OFF. The live
training updates show the same conflict: across ON confirmations L2 sums
to -12.234 and L1 to +8.807, while across OFF confirmations L2 sums to
+10.087 and L1 to -3.063. Quiet confirmations add smaller opposing changes.

The context needed to separate these demands is available locally at issue:
167/168 ON examples have positive sensory state and all168 OFF examples have
negative sensory state (the exceptional ON is startup with zero state). The
timing gate itself is not causing the steady slow-tempo error. A single
anatomical edge may need two locally selected efficacy states, as the prior
split frozen-history replay suggested. Relative normalization of one scalar
magnitude would rescale these conflicting effects but cannot select a
different magnitude for the two local contexts.

## Coupling, stability and limits

Initial-versus-trained evaluation changes 152,173,206 and261 spike entries
at dwells2,3,4,6, and297/285 entries on the two continuous streams. The
neural feedback path is active. All recorded states and weights stay finite;
incoming magnitudes remain in [0,10] and non-target weights are unchanged.
Per-frame proposed updates reconstruct saved synchronized target magnitudes
within 1.2e-7. The eta.1 training stream ran 2453 frames in61.05 seconds,
40.18 camera frames/s with eight neural ticks each; mean cell rate was3.86Hz
and maximum individual rate30.62Hz. This shows real online plasticity on the
small motif, not full Pong throughput.

The model still trains on repeated two-position motion rather than event-camera
Pong sequences. Its first-pass *pre-update* training OFF anticipation was only
.037, despite the final snapshot's .125 validation OFF. Closed-gate surprises
receive no magnitude update and omission recovery remains imperfect. These
limits require further evidence before claiming effective M1A learning.

The next architectural experiment should use two magnitude components on
**each existing predictive edge**, selected by the target neuron's local
sensory-state sign, while retaining the gate-aligned local update. The selected
component must also determine the physical recurrent current, so closed-loop
neural and plasticity semantics stay aligned. A concrete, bounded design is
in [the context-efficacy proposal](2026-09-22-context-efficacy-proposal.md).
This is a model change requiring review under the previously agreed process.

## Evidence and reproduction

- [Training selection and hashes](2026-09-22-live-local-training-results.json)
- [All held-out conditions, firing and update audit](2026-09-22-live-local-evaluation-results.json)
- [Signed edge/current audit](2026-09-22-live-local-credit-audit.json)
- Local full traces: `runs/live-local-timing-v1/*.npz`

```powershell
.venv/Scripts/python scripts/live_local_timing.py train
.venv/Scripts/python scripts/live_local_evaluate.py
.venv/Scripts/python scripts/live_local_credit_audit.py
```

Saved run files are protected against overwrite or source changes; a fresh
reproduction should use an empty run directory while retaining the original
source artifacts.
