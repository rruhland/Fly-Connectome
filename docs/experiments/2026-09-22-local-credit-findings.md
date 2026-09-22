# Causal gate-aligned credit learns useful amplitudes on frozen histories

The replay found a local credit rule that learns both event signs on the
recorded eligibility histories and transfers to unseen tempo 3 and untouched
continuous histories. The decisive change is to use the **same causal timing
gate** in the issued prediction and the eligibility receiving credit. A
single magnitude per anatomical edge suffices for an aggregate pass, though
two locally selected magnitudes are materially stronger across individual
tempos. These are offline weight updates on fixed neural histories; the next
required test is live neural feedback.

## Protocol and causal boundaries

Each forecast uses recorded presynaptic eligibility and postsynaptic local
sensory state at issue, and its local event target eight neural ticks later.
Updates use only that issued state and the then-observed target. Initial
magnitudes are the original anatomical values, not the offline optimum.
Weight magnitudes remain nonnegative and at most 10; transmitter signs and
the twelve incoming anatomical edges are fixed. There is no backpropagation.

Training uses trials1..14 from the tempo2/4/6 recordings. Trials15..19 alone
select learning rate; trials20..29 and all tempo3 recordings are held out.
The standard and omitted continuous neural histories are also untouched
tests. All gate histories come from chronological local observations, not
tempo/phase labels or future inputs. Five passes reuse the same training
samples as repeated presentations; first-pass and pre-update scores are
reported separately. Six variants and the complete .01/.03/.1/.3/1 rate grid
are retained in the machine-readable results.

## Selected-rule comparison

All scores below apply the same causal timing gate at evaluation. The original
and ungated-credit controls used ungated prediction in their own updates;
their ordinary ungated output scores are also retained in the results.

| Local amplitude rule | Validation ON/OFF | Known test ON/OFF | Unseen 3 ON/OFF | Standard continuous ON/OFF | Standard quiet alarms |
|---|---:|---:|---:|---:|---:|
| Original shared scalar | .110/.036 | .104/.038 | .273/.085 | .256/.050 | .64% |
| Shared, gate-consistent | .456/.124 | .420/.132 | .667/.324 | .494/.177 | .96% |
| Shared, gate-consistent + causal balance | .557/.316 | .503/.338 | .667/.857 | .358/.444 | .96% |
| Split context, ungated credit | .054/.048 | .052/.049 | .087/.089 | .089/.063 | .64% |
| Split context, gate-consistent | .468/.701 | .443/.706 | .666/1.000 | .581/.798 | .64% |
| Split context, gate-consistent + causal balance | .632/.699 | .611/.705 | .667/1.000 | .639/.797 | .64% |

The selected shared balanced rule also scores .393/.455 ON/OFF on the omitted
continuous stream, with 1.27% quiet alarms. The selected split balanced rule
scores .641/.816 there, with .95% quiet alarms. All gated rules stay below the
5% quiet-alarm reference on these particular tests. The common gate provides
most of that suppression; it should not be credited to learned weights.

The original scalar and split-with-ungated-credit controls miss at least one
polarity threshold. Gate-consistent credit passes both with or without split
weights. Causal event/quiet balancing uses only counts observed before each
update, capped at gain 8. It improves the shared model's unseen OFF response
from .324 to .857. This is a bounded diagnostic candidate, not proof that the
gain formula is biologically correct.

## Learning speed and nonuniform performance

With the chosen rate, the shared balanced rule has first-pass *pre-update*
ON/OFF anticipation .525/.423 on its training presentation. Its first-pass
validation scores are .559/.311, so the result does not require five passes
to emerge. The split balanced rule has pre-update .545/.345 and first-pass
validation .629/.534. These are sample counts from fixed histories, not wall
time claims about live neural training.

The shared balanced rule has known-tempo OFF anticipation .080 at tempo2,
despite passing pooled tests. It has .700 at tempo4 and .234 at tempo6. The
split balanced rule has OFF .667, 1.000 and .447 at tempos2,4,6 respectively;
its ON values are .334, .667 and .831. This is why a future split architecture
remains plausible. For a first live test, the shared rule is the smaller
intervention. Its transfer limitation should be an explicit acceptance check,
not hidden in the pooled mean.

The split unbalanced fit drives one magnitude to its bound of10. The selected
split balanced model reaches9.905, still close to the bound. The shared
balanced model reaches9.322 on one active edge. Clipping, bound proximity and
the fixed basis may limit further gains; these are recorded rather than
interpreted as convergence to a biological optimum.

## What is and is not learned here

The weight update is local and causal *given the recorded traces*. Those traces
came from a network with different frozen weights. During a real run, the
learned weights will alter recurrent currents, spikes, eligibility and future
sensory predictions. The replay cannot establish that the coupled network
learns the same way. It also cannot establish M1A on Pong; this remains a small
motif with repeated dot motion. M1B remains blocked.

The next implementation should therefore be an experimental live **shared
gate-consistent** rule: one magnitude per existing edge, a local past-event
reference, and gate-aligned error/eligibility. Test exact frozen behavior at
zero learning rate, then train and evaluate continuously with a true coupled
network. If the shared rule underperforms by polarity or tempo, the measured
split advantage justifies a separate design review for two magnitudes per
anatomical edge. The detailed first-step proposal is in
[the live-rule proposal](2026-09-22-live-local-timing-proposal.md).

## Reproduction

```powershell
.venv/Scripts/python scripts/local_credit_probe.py
.venv/Scripts/python -m pytest tests/test_local_credit_probe.py -q
```

- [Protocol and follow-up control](2026-09-22-local-credit-protocol.md)
- [All candidates, source hashes, first-pass and challenge scores](2026-09-22-local-credit-results.json)
