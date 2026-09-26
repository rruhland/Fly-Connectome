# Diverse motion experience teaches faster dynamics but overwrites slow behavior

**Decision:** Do not promote either 256-episode field. The [equal-budget protocol](2026-09-26-multicadence-experience-protocol.md) shows that visual tempo diversity is necessary for changed-speed transfer, but this single shared transition kernel cannot retain slow/native-cadence performance while learning fast motion. More repetitions of the original distribution worsened transfer. Production M1A is unchanged.

The locally learned observer was frozen from the prior 64 generic training scenes. Two new distributed transition fields used identical state, local prediction-error updates, and 256 episodes: four repeats of each original scene, or each original scene viewed at camera strides 1/2/4/8. Both conditions used the same seeds and episode length; no labels or Pong data entered learning. They were evaluated on exactly the same transfer streams as the prior frozen audit.

| Full-frame top-32 next-event recall | Original 64 | Repeated 256 | Multi-cadence 256 |
|---|---:|---:|---:|
| Changed position | .173 | .112 | **.406** |
| Changed speed | .089 | .062 | **.418** |
| Changed shape | .344 | .208 | **.459** |
| Two separated entities | .170 | .111 | **.341** |
| Crossing entities | .172 | .109 | **.271** |
| Pong, native cadence | **.347** | .262 | .134 |
| Pong, four-step cadence | **.314** | .196 | .276 |
| Clean generic held-out | **.685** | .604 | .583 |
| Corrupted generic, trained noise rate | **.635** | .558 | .503 |

The diverse field beat equal-budget repetition on every changed-motion group, often by several times, so the earlier speed failure was not an immutable limit of the local kernel. The raw-input field showed the same pattern (e.g. changed-speed .425 multi versus .057 repeated), confirming that the fixed observer did not create the gain. Repetition itself degraded the original field, consistent with online overwriting/overfitting. But adding faster experience did not preserve the original slow or noisy task, and native Pong worsened dramatically even relative to the repeated control. This is a context conflict in one learned transition state, not a reason to sweep the observer gate or local step size. A separate domain shift in native Pong sensing may also contribute; this experiment cannot allocate the whole regression to tempo alone.

The next substantial architecture candidate should let local predictive transition state express **multiple temporal regimes simultaneously** with locally learned credit and a local context selector, while keeping the observer and generic sensory primitives fixed. A winner-take-all hand-coded speed bank would violate the representation goal. The meaningful test is joint retention: changed-speed and multiple-entity gains **and** original generic/noise and native Pong, with shuffled temporal-context credit as a control. If a compact multi-timescale recurrent representation still trades one regime for another, this latent family should be reconsidered rather than extended through more narrow variants.

Training the two 256-episode fields took 36 and 48 seconds respectively; the full run took 123 seconds, so the limit here was model behavior rather than real-time event waiting. Reproduce with `.venv/Scripts/python scripts/run_multicadence_experience.py`; [raw results](2026-09-26-multicadence-experience-results.json) retain per-case scores and state metrics.
