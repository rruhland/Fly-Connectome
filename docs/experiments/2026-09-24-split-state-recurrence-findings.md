# Split sensory access plus the existing recurrent rule still fails causal reappearance

The [registered opt-in experiment](2026-09-24-split-state-recurrence-protocol.md) joined the separately learned raw and coincidence encoders to the prior `SeparatedVisualState` transition. Local updates trained a hidden transition and separate observed/imagined event-emission banks on 96 clean plus 32 interruption/noise episodes. The controls used the same split sensory input with no recurrence, and the trained recurrent model with state reset just before reappearance. No production model changed; this run took 222 seconds.

| Held-out online next-event F1 | Archived coincidence learner | Split recurrent | Split no recurrence | Split state reset |
| --- | ---: | ---: | ---: | ---: |
| Unseen single patterns | **0.669** | 0.588 | 0.509 | 0.588 |
| Independent movers | **0.681** | 0.623 | 0.545 | 0.623 |
| Crossings | **0.583** | 0.493 | 0.430 | 0.493 |
| Disappearance, all scored frames | **0.430** | 0.312 | 0.221 | 0.312 |
| Speed changes | **0.634** | 0.521 | 0.459 | 0.521 |
| Sensor-bit noise | **0.487** | 0.398 | 0.257 | 0.398 |
| Exact reappearance t=10→11 | 0.000 | **0.067** | **0.067** | **0.067** |

Recurrence improved pooled scores over its matched no-recurrence emitter, so the hidden transition has some effect during ordinary activity. It did **not** contribute to the exact transition requiring pre-gap history: all three split variants had the same six true-positive and 50 false-positive pixels against 124 target events. State reset caused no change at that transition. The recurrent model had imagined activity on 192 of 368 blank-input frames, but mere hidden activity did not become useful context. On matched opposite-motion histories with identical reappearance input, the recurrent forecasts differed by only 0.189 in total absolute pixel probability across the entire 2×32×64 output; no-recurrence and reset forecasts were identical, and all three had the same 0.296 pair F1. The recurrent variant produced 122 false-alarm pixels on 16 quiet-target frames, versus 97 for the archived learner; this stayed below the registered twofold cap but did not buy a reappearance gain.

The mechanism fails the registered reappearance and clean-transfer gates. Do not tune its learning rates or emission threshold through a narrow sweep, and do not promote it. The combined evidence is now specific: first-sighting input is accessible; the preserved learned motion code contains ordinary motion information; a paired counterfactual requires history; this local recurrent rule can maintain activity but does not make past direction causally useful at reappearance. The next high-value question is whether a **different local temporal-context code or learning objective** can bind pre-gap motion to new evidence, rather than simply keep prior activity alive. A short-term event trace with learned context-dependent local credit, and an explicit state-reset/paired-history test, is one candidate. The temporal code should remain a learned representation, not a fixed direction bank or Pong tracker.
