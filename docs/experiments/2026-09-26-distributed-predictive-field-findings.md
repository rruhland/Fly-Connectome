# Distributed local prediction learns clean dynamics but amplifies noise

**Decision:** Reject this simple distributed local-kernel field as the M1A latent. It passed the clean-forecast criterion but failed the [registered](2026-09-26-distributed-predictive-field-protocol.md) corrupted-forecast criterion. Do not tune trace decay, recurrence gain, learning rate, local radius, or intensity cadence on these held-out cases. The experiment remains opt-in; production M1A is unchanged.

The field had no entity proposals or identity slots. Eight continuous local channels held fast/slow signed event traces, signed contrast belief, and a recurrent copy of the previous prediction. A shared 5×5 kernel learned by source activity times local next-event error, with no gradient through time. It trained once on the same 64 unlabeled generic streams and was frozen for four held-out clean/corrupted scenes plus stationary noise. All arms were scored against the same clean next event; independent intensity arrived every eight frames in the aligned and shuffled-intensity arms.

| Stream/arm | Top-32 | Top-8 | False contrast mass / active frame | Non-target forecast mass / scored event |
|---|---:|---:|---:|---:|
| Clean, aligned learned field | **.681** | **.522** | .00 | 4.87 |
| Clean, shuffled temporal credit | .182 | .042 | .00 | 3.39 |
| Clean, graded aligned files | .466 | — | — | — |
| Corrupted, aligned learned field | .231 | .104 | 56.51 | 102.69 |
| Corrupted, no intensity | .097 | .038 | 154.61 | 198.86 |
| Corrupted, shuffled intensity | .209 | .076 | 63.13 | 101.85 |
| Corrupted, shuffled temporal credit | .116 | .025 | 56.51 | 10.25 |
| Corrupted, graded aligned files | **.331** | — | — | — |

The aligned versus shuffled-credit result demonstrates that local temporal credit learned a real spatial-temporal regularity. The distributed state also achieved clean performance without brittle entity identity. Independent intensity reduced false contrast mass by 63% compared with event-only, and corrupted top-32 improved from .097 to .231. But the field still forecast more than 100 units of non-target mass per scored corrupted event versus 1.40 units on target. Aligned and unrelated intensity produced similar corrupted forecast scores (.231/.209), so the signal from occasional true intensity did not reliably govern prediction. The stationary noisy case had zero hits and 70.69 non-target forecast mass even with aligned intensity. The field's visible state remained contaminated by events between independent observations.

The failed criterion is not evidence that distributed recurrence is a dead end. This implementation learned *transitions* but did not learn which incoming events deserve entry into persistent state. The next genuinely different architecture would learn a local **observation model**: predict both future events and current signed appearance, compare incoming events with persistence/appearance evidence, and use the local mismatch to update state and source reliability. That comparison should train using paired event and intensity observations from the same camera stream, never clean evaluation targets or object labels. It must be tested under clean and corrupted input with an independent withheld future, against this field and the graded files. A hand-selected event-amplitude gate or decay sweep would repeat the narrow diagnostic cycle and is not justified.

Reproduce with `.venv/Scripts/python scripts/run_distributed_predictive_field.py` (about 11 seconds here); exact measurements are in [the result JSON](2026-09-26-distributed-predictive-field-results.json).
