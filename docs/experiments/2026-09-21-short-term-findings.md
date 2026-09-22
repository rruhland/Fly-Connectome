# STP preflight stops on an overly broad target-equality requirement

Both approved release variants are implemented and verified. **No training was
run.** Preflight found changed internal observation targets and obeyed the
approved stop rule. This is not a negative learning result: the experiment has
not yet tested whether STP improves prediction.

The important distinction is that all directly injected sensory targets,
including the scored L3 body82450 target, remain identical. Internal observations
change because physical transmission changes downstream spike timing. Requiring
every internal target to remain identical was too broad for this architecture
experiment. A [narrow protocol correction](../plans/2026-09-21-short-term-target-comparison-amendment.md)
is proposed, not applied; the model formulas themselves do not need changing.

## Implementation and controls

Depression and facilitation use the approved utilization .5 and100ms recovery
on existing predictive edges. Release state is private per environment and
recovers analytically only at arrivals. One cached gain per arrival feeds physical
current, signed E/I current, and the arriving local eligibility contribution.
Old eligibility is not rescaled. Fixed signs, measured topology, delays, eight
ticks/frame, +8tick local learning, homeostasis and no-backprop are preserved.

A small default environment-aware impulse hook delegates to the original impulse
function for existing models. STP classes are opt-in reference experiments; no
native/production checkpoint support is claimed.

Tests cover paired impulses at10/30/100/300ms, bursts, recovery, first-impulse
waveform identity, independently reconstructed current and eligibility, repeated
queries, private batch state, unchanged nonpredictive transmission, and runner
integration. Real-crop unit-release controls reproduce predictions, targets,
spikes, eligibility, updates and final learned weights bit-for-bit for both D/F.
Independent code review found no blocking defects.
Final full suite:261 passed,4 skipped; focused release/integration suite:18 passed.

## Real-crop preflight

Five trials per dwell2/3/4/6, blank seed9080, fixed original weights and standard
warmup. We audited every neuron's local observation, separately retaining the
scored L3 comparison. Differences count neuron-tick entries, not visual events.

| Dwell | All local target entries | D differences | F differences | Direct sensory / scored L3 differences |
|---|---:|---:|---:|---:|
| 2 | 412,000 | 291 | 181 | 0 / 0 |
| 3 | 492,000 | 298 | 215 | 0 / 0 |
| 4 | 572,000 | 439 | 206 | 0 / 0 |
| 6 | 732,000 | 444 | 394 | 0 / 0 |

Affected classes include L4, C2/C3, Mi1 and several Dm/Tm types. Some differences
are on neurons receiving predictive plasticity: D has194/176/288/277 and F has
90/102/82/218 such changed entries at2/3/4/6. They cannot all be dismissed as
outside the learning circuit, but they are not changes to the scored sensory task.

The offline audit independently sums recorded presynaptic spikes with original
signed feedforward magnitudes and fixed delays, then applies the unchanged
observation normalization. For every condition it matches all noninjected targets
with **zero maximum error** after the maximum feedforward delay. The first few
ticks are excluded because their pre-sequence spike history was not recorded.
Thus the evidence supports changed physical observations, not an encoder or
transmitter-sign bug. Target-equality failure alone says nothing about subsequent
learning quality.

## Release has activity-dependent variation

Arrival-conditioned median gains on the main L2 body20655 -> L3 pathway:

| Dwell | D gain | F gain |
|---|---:|---:|
| 2 | 0.32769 | 1.67240 |
| 3 | 0.24098 | 1.75903 |
| 4 | 0.25204 | 1.75046 |
| 6 | 0.20712 | 1.79298 |

At dwell6 D spans0.13114-0.97899 and F1.02101-1.86942, so this pathway is not
fixed at a saturation bound. The L1 -> L3 pathway behaves similarly. These are
realized arrival gains; stored states are measured immediately after arrivals.
They do not show that a useful timing representation exists. Current-event
arrival subsets are particularly small (only1-3 L2 arrivals per condition), so
event/quiet quantiles are descriptive, not reliable evidence of separation.
No classifier or gate was fitted and no local-learning success is claimed.

All runs remain finite and within weight bounds. Release state, timestamps and
the edge index add27,052 persistent bytes on this crop, plus transient arrival
buffers. The12 frozen sequences take55.83s total, excluding warmup/neutral checks
and file compression. STP runs include release logging that baseline lacks;
these times are not a clean overhead benchmark or a full-connectome throughput
claim. No optimized kernel was changed.

## Decision and reproduction

Honor the original protocol: no200-trial training, frozen learning evaluation,
confirmation or ablation was started. Preserve both implemented variants and raw
preflight artifacts. M1A remains unmet and M1B blocked.

The proposed correction keeps every external/scored target identical and permits
internal observations to follow actual physical arrivals. It retains the same
local rule and all original training/evaluation settings. Approval is required
because it changes the explicitly approved comparison/stop criterion, not because
another model mechanism is needed.

Run `.venv/Scripts/python scripts/short_term_experiment.py` for preflight (refuses
an existing runs/short-term-v1 directory), then
`.venv/Scripts/python scripts/short_term_audit.py` to reconstruct targets and
export the report. Raw prediction/target/spike/eligibility/release arrays remain
in ignored runs/short-term-v1. No source checkpoint is modified.

[Preflight metrics](2026-09-21-short-term-preflight-results.json),
[stop decision and exact neutral controls](2026-09-21-short-term-preflight-decision.json),
[target reconstruction and pathway gains](2026-09-21-short-term-audit.json),
[raw artifact checksums](2026-09-21-short-term-artifacts.json).
